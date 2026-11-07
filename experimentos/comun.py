"""Utilidades compartidas por los notebooks de experimentos (capítulos 8 y 9).

Se importa desde los notebooks de `experimentos/` con `import comun`. No importa CrewAI ni el
código pesado del sistema (YOLO, ResNet): localiza rutas, lee la configuración y las anotaciones,
gestiona la caché de percepción (que se construye en un proceso aparte) y la congelación.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ETIQUETAS_SISTEMA = ["animal", "arquitectura", "paisaje", "producto-still_life", "retrato-humano"]
ETIQUETAS_A1 = ETIQUETAS_SISTEMA + ["otro"]

PAQUETES = ["crewai", "ultralytics", "torch", "torchvision", "opencv-contrib-python",
            "scikit-learn", "numpy", "pandas", "grad-cam", "pillow", "matplotlib"]

# --- Anotaciones (ver experimentos/anotaciones/GUIA_ANOTACION.md) ---
SI_NO_DUDOSO = {"si", "no", "dudoso"}
ESQUEMA_A1 = {"contexto": set(ETIQUETAS_A1)}
# Versión acotada (2026-09-27): `horizontal_estructural` y `vegetacion_dominante` se retiraron
# para reducir la anotación (experimentos.md §15).
ESQUEMA_A2 = {
    "color_analizable": SI_NO_DUDOSO,
    "horizonte_natural": SI_NO_DUDOSO,
    "perspectiva": SI_NO_DUDOSO,
    "sin_sujeto": SI_NO_DUDOSO,
    "segmentacion": {"ok", "parcial", "mal"},
    "protagonista": SI_NO_DUDOSO,
}
# Columnas que se rellenan desde `anotaciones/ANOTAR_AQUI.xlsx` (importar_anotaciones.py).
COLUMNAS_HOJA = {"A1_contexto": ["contexto"],
                 "A2_aplicabilidad": ["color_analizable", "horizonte_natural", "perspectiva", "sin_sujeto"]}
ESQUEMA_PATRON = {"patron_percibido": {"centrada", "tercios", "otro"}}
# Columnas de A2 que solo son obligatorias cuando la percepción actual tiene fuente = "yolo"
# (evalúan la máscara o la caja que usa el sistema, y solo con "yolo" pueden abrirse sus gates).
A2_SOLO_YOLO = ("segmentacion", "protagonista")

# Módulos de los que dependen las cachés: si cambia su código (o sus constantes), la caché está
# obsoleta.
MODULOS_CACHE = ("tools/shared_perception_tool.py", "tools/espacio_aislamiento_tool.py", "tools/imagen.py")
MODULOS_GATES = ("tools/luz_tono_tool.py", "tools/lineas_direccion_tool.py",
                 "tools/composicion_espacial_tool.py", "tools/espacio_aislamiento_tool.py",
                 "tools/imagen.py", "tools/capa_verificacion.py", "tools/prioridad_tool.py",
                 "schemas/especialistas.py", "parametros/modelo_nulo_fuga.json")

# Dimensiones del Nivel 2, con la clave de W (la misma que usa PrioridadTool).
DIMENSIONES = ["composicion_espacial", "lineas_direccion", "espacio_aislamiento_sujeto", "luz_tono"]


# =============================================================================
# Rutas y configuración
# =============================================================================

def raiz() -> Path:
    """Raíz del repositorio: el primer directorio con `pyproject.toml` hacia arriba."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("No se encuentra pyproject.toml: abre el notebook dentro del repositorio")


def paquete() -> Path:
    return raiz() / "src" / "tfg_multiagente_fotografia"


def preparar_sys_path() -> Path:
    """Añade `src/` y `experimentos/` a `sys.path` y devuelve la raíz."""
    r = raiz()
    for d in (r / "src", r / "experimentos"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    return r


def cargar_config() -> dict:
    with open(raiz() / "experimentos" / "config_experimentos.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta_config(clave: str) -> Path:
    """Resuelve una ruta de `rutas:` relativa a la raíz del repositorio."""
    p = Path(cargar_config()["rutas"][clave])
    return (p if p.is_absolute() else raiz() / p).resolve()


def dir_resultados(experimento: str) -> Path:
    d = ruta_config("resultados") / experimento
    d.mkdir(parents=True, exist_ok=True)
    return d


def relativa(p: Path) -> str:
    """Ruta relativa a la raíz en formato posix (clave estable entre sistemas)."""
    p = Path(p).resolve()
    return p.relative_to(raiz()).as_posix() if p.is_relative_to(raiz()) else p.as_posix()


def guardar_json(obj, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(_a_json(obj), indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    return ruta


def _a_json(o):
    """Convierte tipos de NumPy y rutas a tipos serializables, y NaN/inf a null."""
    if isinstance(o, dict):
        return {str(k): _a_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_a_json(v) for v in o]
    if hasattr(o, "tolist"):          # arrays y escalares de NumPy
        return _a_json(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return o.as_posix()
    return o


def md5(ruta: Path) -> str:
    """MD5 del fichero, memorizado por (ruta, tamaño, fecha de modificación)."""
    st = Path(ruta).stat()
    return _md5_memo(str(Path(ruta).resolve()), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=4096)
def _md5_memo(ruta: str, _tam: int, _mtime: int) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        while trozo := f.read(1 << 20):
            h.update(trozo)
    return h.hexdigest()


def imagenes_C2() -> list[Path]:
    """Las imágenes de C2 ordenadas por número (imagen1, imagen2, ..., imagen100)."""
    return sorted(ruta_config("corpus_C2").glob("imagen*.jpg"), key=lambda p: int(p.stem[6:]))


def leer_csv_robusto(ruta: Path) -> pd.DataFrame:
    """Lee un CSV detectando el separador (`,` o `;`) y la codificación (utf-8 o cp1252)."""
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(ruta, sep=None, engine="python", encoding=codificacion, dtype=str,
                               keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {ruta} ni en utf-8 ni en cp1252")


def coste_usd(registro: dict, precios: dict) -> float | None:
    """Coste estimado en USD de una ejecución (`registro.json` de ejecutar_lote.py).

    Usa `precios.modelos` (USD por millón de tokens). Los tokens de razonamiento ya van dentro
    de `completion_tokens`. No descuenta la caché implícita de Gemini: es una cota superior.
    None si falta algún precio.
    """
    tabla = (precios or {}).get("modelos") or {}
    usos = [(u["modelo"], u["prompt_tokens"], u["completion_tokens"]) for u in registro.get("tokens_llm", [])]
    lv = registro.get("tokens_lectura_visual")
    if lv:
        usos.append((lv["modelo"], lv["prompt_tokens"], lv["candidates_tokens"] + lv["thoughts_tokens"]))
    total = 0.0
    for modelo, entrada, salida in usos:
        p = tabla.get(modelo.split("/")[-1]) or {}
        if p.get("entrada") is None or p.get("salida") is None:
            return None
        total += (entrada * p["entrada"] + salida * p["salida"]) / 1e6
    return total


def intervalo_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson (95 % por defecto) para la proporción k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    radio = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centro - radio), min(1.0, centro + radio))


# =============================================================================
# Anotaciones
# =============================================================================

def normalizar_valor(v) -> str:
    """Minúsculas, sin espacios y `sí` -> `si`. Vacío o NaN -> ''."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip().lower()
    return {"sí": "si", "nan": ""}.get(s, s)


def leer_anotacion(nombre: str, esquema: dict) -> pd.DataFrame:
    """Lee `anotaciones/{nombre}.csv` indexado por `imagen`, con las columnas del esquema normalizadas."""
    df = leer_csv_robusto(ruta_config("anotaciones") / f"{nombre}.csv").set_index("imagen")
    for col in esquema:
        df[col] = df[col].map(normalizar_valor)
    return df


def validar_anotacion(df: pd.DataFrame, esquema: dict, requeridas: dict | None = None) -> pd.DataFrame:
    """Resumen por columna: cuántas filas deben anotarse, cuántas lo están, cuáles faltan y
    cuáles tienen un valor fuera del esquema.

    `requeridas` limita, por columna, las imágenes en las que la columna es obligatoria
    (por defecto, todas las filas).
    """
    filas = []
    for col, validos in esquema.items():
        deben = list(requeridas.get(col, df.index)) if requeridas else list(df.index)
        valores = df[col].reindex(deben).fillna("")
        faltan = [i for i, v in valores.items() if v == ""]
        invalidas = {i: v for i, v in df[col].items() if v and v not in validos}
        filas.append({"columna": col, "requeridas": len(deben), "anotadas": len(deben) - len(faltan),
                      "completa": not faltan and not invalidas, "faltan": faltan,
                      "invalidas": invalidas})
    return pd.DataFrame(filas).set_index("columna")


def revisiones_obsoletas(a2: pd.DataFrame, percepcion: dict) -> pd.DataFrame:
    """Filas de A2 con `segmentacion` o `protagonista` revisadas sobre otra fuente o clase de YOLO
    distinta de la percepción actual: la máscara revisada ya no es la que usa el sistema."""
    revisadas = a2.index[(a2.segmentacion != "") | (a2.protagonista != "")]
    t = pd.DataFrame({
        "fuente_revisada": a2.loc[revisadas, "fuente_revisada"],
        "clase_revisada": a2.loc[revisadas, "clase_revisada"],
        "fuente_actual": [percepcion[s]["fuente"] for s in revisadas],
        "clase_actual": [percepcion[s]["clase"] or "" for s in revisadas],
    }, index=revisadas)
    return t[(t.fuente_revisada != t.fuente_actual) | (t.clase_revisada != t.clase_actual)]


# =============================================================================
# Constantes del sistema (leídas del código fuente, sin importarlo)
# =============================================================================

def constantes_modulo(ruta: Path) -> dict:
    """Constantes de módulo (NOMBRES_EN_MAYÚSCULAS) con valor literal o aritmético.

    Se leen con `ast` para no importar el módulo (importar `crew.py` o `shared_perception_tool.py`
    cargaría YOLO y la ResNet). Las cadenas largas (prompts) se guardan como su MD5.
    """
    arbol = ast.parse(Path(ruta).read_text(encoding="utf-8"))
    constantes = {}
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign):
            objetivos, valor = nodo.targets, nodo.value
        elif isinstance(nodo, ast.AnnAssign) and nodo.value is not None:
            objetivos, valor = [nodo.target], nodo.value
        else:
            continue
        for t in objetivos:
            if not (isinstance(t, ast.Name) and t.id.isupper()):
                continue
            try:
                v = ast.literal_eval(valor)
            except (ValueError, SyntaxError):
                try:   # expresiones aritméticas como 1 / 3; lo que dependa de nombres se omite
                    v = eval(compile(ast.Expression(valor), "<constante>", "eval"), {"__builtins__": {}}, {})
                except Exception:
                    continue
            if isinstance(v, str) and len(v) > 200:
                v = "md5:" + hashlib.md5(v.encode("utf-8")).hexdigest()
            try:
                constantes[t.id] = json.loads(json.dumps(v))   # tuplas -> listas, como al releer el JSON
            except (TypeError, ValueError):
                continue
    return constantes


def constantes_sistema(modulos=None) -> dict:
    """Constantes de `tools/*.py` y `crew.py` (o de los módulos .py indicados), por módulo."""
    base = paquete()
    if modulos is None:
        modulos = [p.relative_to(base).as_posix() for p in sorted((base / "tools").glob("*.py"))
                   if p.name != "__init__.py"] + ["crew.py"]
    return {m: constantes_modulo(base / m) for m in modulos if m.endswith(".py")}


def md5_codigo(modulos) -> dict:
    """MD5 de los ficheros del paquete indicados (rutas relativas a `src/tfg_multiagente_fotografia`)."""
    return {m: md5(paquete() / m) for m in modulos}


def modelo_llm() -> dict:
    """`MODEL` de los especialistas: variable de entorno o, si no, la línea de `.env` (solo esa)."""
    modelo = os.environ.get("MODEL")
    env = raiz() / ".env"
    if modelo is None and env.exists():
        for linea in env.read_text(encoding="utf-8").splitlines():
            if linea.strip().startswith("MODEL="):
                modelo = linea.split("=", 1)[1].strip().strip('"').strip("'")
    return {"MODEL": modelo}


# =============================================================================
# Procedencia y congelación
# =============================================================================

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=raiz(), capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def procedencia(ficheros: list[Path] | None = None) -> dict:
    """Commit, estado del árbol, versiones, GPU y MD5 de los ficheros indicados."""
    versiones = {}
    for nombre in PAQUETES:
        try:
            versiones[nombre] = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            versiones[nombre] = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda = torch.version.cuda
    except ImportError:
        gpu = cuda = None
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "arbol_limpio": _git("status", "--porcelain") == "",
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "gpu": gpu,
        "cuda": cuda,
        "versiones": versiones,
        "md5": {relativa(p): md5(p) for p in (ficheros or [])},
    }


RUTAS_CODIGO = ("src", "models", "pyproject.toml", "uv.lock")


def ficheros_congelados() -> list[Path]:
    """Ficheros cuyo contenido fija la congelación (además del commit del código)."""
    r, base = raiz(), paquete()
    anot = ruta_config("anotaciones")
    candidatos = [
        *imagenes_C2(),
        *sorted((r / "models" / "clasificador_contexto").glob("*")),
        r / "models" / "yolo" / "yolo11m.pt",
        *sorted((base / "parametros").glob("*.json")),
        *sorted((base / "config").glob("*.yaml")),
        r / "experimentos" / "config_experimentos.yaml",
        r / "experimentos" / "manifiesto_C2.csv",
        *sorted(anot.glob("*.csv")),
    ]
    return [p for p in candidatos if p.is_file()]


def estado_sistema() -> dict:
    """Lo que la congelación fija: commit, MD5 de datos/modelos/prompts/anotaciones, constantes y LLM."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "arbol_limpio": _git("status", "--porcelain") == "",
        "md5": {relativa(p): md5(p) for p in ficheros_congelados()},
        "constantes": constantes_sistema(),
        "llm": modelo_llm(),
    }


def diferencias_estado(congelado: dict, actual: dict) -> list[str]:
    """Diferencias entre un estado congelado y el actual, en frases legibles."""
    dif = []
    if congelado.get("commit") != actual["commit"]:
        # Commitear congelacion.json cambia HEAD; solo cuenta si cambió el código.
        cambios = _git("diff", "--name-only", congelado.get("commit", ""), "HEAD", "--", *RUTAS_CODIGO)
        if cambios:
            dif.append("código cambiado desde la congelación: " + ", ".join(cambios.splitlines()))
    sin_commit = _git("status", "--porcelain", "--", *RUTAS_CODIGO)
    if sin_commit:
        dif.append("cambios sin commit en el código: " + ", ".join(sin_commit.splitlines()))
    md5_c, md5_a = congelado.get("md5", {}), actual["md5"]
    for f in sorted(set(md5_c) | set(md5_a)):
        if md5_c.get(f) != md5_a.get(f):
            estado = "falta" if f not in md5_a else "nuevo" if f not in md5_c else "modificado"
            dif.append(f"{f}: {estado}")
    for modulo in sorted(set(congelado.get("constantes", {})) | set(actual["constantes"])):
        c, a = congelado.get("constantes", {}).get(modulo, {}), actual["constantes"].get(modulo, {})
        for nombre in sorted(set(c) | set(a)):
            if c.get(nombre) != a.get(nombre):
                dif.append(f"{modulo}:{nombre}: {c.get(nombre)!r} -> {a.get(nombre)!r}")
    if congelado.get("llm") != actual["llm"]:
        dif.append(f"LLM: {congelado.get('llm')} -> {actual['llm']}")
    return dif


def comprobar_congelacion() -> bool:
    """Compara el sistema actual con `experimentos/congelacion.json` (lo genera 00_preparacion).

    Devuelve True si coincide. Si no hay congelación o algo difiere, avisa y devuelve False: los
    resultados valen como prueba, no como resultados finales del capítulo 9.
    """
    ruta = raiz() / "experimentos" / "congelacion.json"
    if not ruta.exists():
        warnings.warn("No existe experimentos/congelacion.json: el sistema aún no está congelado. "
                      "Los resultados de esta ejecución son provisionales.", stacklevel=2)
        return False
    dif = diferencias_estado(json.loads(ruta.read_text(encoding="utf-8")), estado_sistema())
    if dif:
        warnings.warn("El sistema difiere de la congelación:\n  - " + "\n  - ".join(dif), stacklevel=2)
        return False
    return True


# =============================================================================
# Cachés de C2 (percepción y gates), construidas en un proceso aparte
# =============================================================================
# Cada caché guarda en su `meta` una FIRMA de todo aquello de lo que depende: el MD5 de cada
# imagen, el MD5 del código y las constantes de los módulos implicados, y lo propio de cada una
# (los pesos de YOLO; la fecha de la caché de percepción de la que parte la de gates). Si la
# firma actual no coincide, la caché está obsoleta y se reconstruye.

CACHES = {
    "percepcion": {"script": "_cache_percepcion.py", "modulos": MODULOS_CACHE, "minutos": 8,
                   "ficheros": ("percepcion.json", "mascaras.npz")},
    "gates": {"script": "_cache_gates.py", "modulos": MODULOS_GATES, "minutos": 5,
              "ficheros": ("gates.json",)},
}


def dir_cache(nombre: str) -> Path:
    return raiz() / "outputs" / "experimentos" / "cache" / nombre


def _leer_cache(nombre: str) -> dict:
    return json.loads((dir_cache(nombre) / CACHES[nombre]["ficheros"][0]).read_text(encoding="utf-8"))


def firma_cache(nombre: str) -> dict:
    """Lo que la caché `nombre` necesita que no cambie (sin el MD5 de las imágenes)."""
    modulos = CACHES[nombre]["modulos"]
    firma = {"md5_codigo": md5_codigo(modulos), "constantes": constantes_sistema(modulos)}
    if nombre == "percepcion":
        firma["md5_yolo"] = md5(raiz() / "models" / "yolo" / "yolo11m.pt")
    else:
        firma["fecha_percepcion"] = _leer_cache("percepcion")["meta"]["fecha"]
    return firma


def vigencia_cache(nombre: str) -> str | None:
    """None si la caché vale; si no, el motivo por el que está obsoleta."""
    if not all((dir_cache(nombre) / f).exists() for f in CACHES[nombre]["ficheros"]):
        return "no existe"
    if nombre != "percepcion" and (motivo := vigencia_cache("percepcion")):
        return f"la caché de percepción no vale ({motivo})"
    meta = _leer_cache(nombre)["meta"]
    guardadas = meta.get("md5_imagenes", {})
    actuales = {p.stem: md5(p) for p in imagenes_C2()}
    if set(guardadas) != set(actuales):
        return "el conjunto de imágenes de C2 ha cambiado"
    cambiadas = [s for s in actuales if guardadas[s] != actuales[s]]
    if cambiadas:
        return f"han cambiado {len(cambiadas)} imágenes ({', '.join(cambiadas[:5])})"
    for clave, valor in firma_cache(nombre).items():
        previo = meta.get(clave)
        if previo == valor:
            continue
        if isinstance(valor, dict) and isinstance(previo, dict):
            distintos = sorted(k for k in set(valor) | set(previo) if valor.get(k) != previo.get(k))
            return f"ha cambiado {clave} en {', '.join(distintos)}"
        return f"ha cambiado {clave}"
    return None


def construir_cache(nombre: str, recalcular: bool = False) -> Path:
    """Construye la caché en un proceso aparte si falta, está obsoleta o se pide recalcular.

    El proceso aparte carga los modelos y los engines; así el kernel del notebook no ocupa la
    GPU ni la memoria de YOLO. La de gates construye antes la de percepción si hace falta.
    """
    if nombre != "percepcion":
        construir_cache("percepcion")
    motivo = "se ha pedido recalcular" if recalcular else vigencia_cache(nombre)
    if motivo is None:
        return dir_cache(nombre)
    print(f"Construyendo la caché de {nombre} ({motivo}). Tarda unos {CACHES[nombre]['minutos']} minutos...")
    script = raiz() / "experimentos" / CACHES[nombre]["script"]
    proc = subprocess.run([sys.executable, str(script), str(raiz()), str(dir_cache(nombre))], cwd=raiz(),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Falló la caché de {nombre}:\n" + proc.stdout[-2000:] + proc.stderr[-3000:])
    print(proc.stdout.strip().splitlines()[-1])
    return dir_cache(nombre)


def meta_cache(nombre: str) -> dict:
    """La `meta` que debe guardar el script de la caché: firma actual + MD5 de las imágenes."""
    return {"fecha": datetime.now().isoformat(timespec="seconds"), "commit": _git("rev-parse", "HEAD"),
            **firma_cache(nombre), "md5_imagenes": {p.stem: md5(p) for p in imagenes_C2()}}


def cargar_cache(nombre: str) -> dict:
    """Devuelve el JSON de la caché (`meta` + `imagenes`). Falla si está obsoleta."""
    motivo = vigencia_cache(nombre)
    if motivo is not None:
        raise RuntimeError(f"La caché de {nombre} no vale ({motivo}): reconstrúyela con "
                           f"comun.construir_cache('{nombre}', recalcular=True)")
    return _leer_cache(nombre)


# Atajos para la caché de percepción (la usan 00_preparacion y E2).
def dir_cache_percepcion() -> Path:
    return dir_cache("percepcion")


def construir_cache_percepcion(recalcular: bool = False) -> Path:
    return construir_cache("percepcion", recalcular)


def vigencia_cache_percepcion() -> str | None:
    return vigencia_cache("percepcion")


def cargar_cache_percepcion() -> tuple[dict, dict, dict]:
    """Devuelve (meta, percepción por imagen, máscara por imagen). Falla si la caché está obsoleta."""
    datos = cargar_cache("percepcion")
    with np.load(dir_cache("percepcion") / "mascaras.npz") as npz:
        mascaras = {k: npz[k] for k in npz.files}
    return datos["meta"], datos["imagenes"], mascaras
