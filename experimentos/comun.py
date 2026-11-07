"""Utilidades compartidas por los scripts y notebooks de la validación (experimentos.md §7).

Se importa con `import comun` desde `experimentos/`. No importa CrewAI ni los modelos pesados
(YOLO, SAM, ResNet): localiza rutas, lee la configuración, los manifiestos y las anotaciones,
replica las decisiones de aplicabilidad, calcula intervalos de Wilson y costes, y gestiona la
congelación (§9). El verificador de citas de E3 está en `verificador.py`.

Desde la línea de órdenes (raíz del proyecto, en la máquina Windows):

    uv run python experimentos/comun.py congelar     # escribe experimentos/congelacion.json
    uv run python experimentos/comun.py comprobar    # compara el sistema con la congelación
"""
from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

ETIQUETAS_SISTEMA = ["animal", "arquitectura", "paisaje", "producto-still_life", "retrato-humano"]
ETIQUETAS_CONTEXTO = ETIQUETAS_SISTEMA + ["otro"]
DIMENSIONES = ["composicion_espacial", "lineas_direccion", "espacio_aislamiento_sujeto", "luz_tono"]

SI_NO_DUDOSO = {"si", "no", "dudoso"}
# Columnas de anotaciones/anotacion_prueba.csv (plantilla_anotacion_prueba.py).
ESQUEMA_ANOTACION_PRUEBA = {
    "contexto": set(ETIQUETAS_CONTEXTO),
    **{c: SI_NO_DUDOSO for c in ("color", "horizonte_natural", "horizontal_estructural",
                                 "horizonte_amplio", "perspectiva", "protagonista", "espacio",
                                 "nitidez", "anclaje")},
}
ESQUEMA_SEGMENTACION = {"segmentacion": {"bien", "parcial", "mal"},
                        "caja_protagonista": {"si", "no"}}

PAQUETES = ["crewai", "ultralytics", "torch", "torchvision", "opencv-contrib-python",
            "google-genai", "litellm", "numpy", "pandas", "scikit-learn", "pillow", "psutil"]

# Lo que no puede cambiar después de congelar sin invalidar las tandas.
RUTAS_CODIGO = ("src", "models", "pyproject.toml", "uv.lock")


# =============================================================================
# Rutas y configuración
# =============================================================================

def raiz() -> Path:
    """Raíz del repositorio: el primer directorio con `pyproject.toml` hacia arriba."""
    for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("No se encuentra pyproject.toml: ejecuta dentro del repositorio")


def paquete() -> Path:
    return raiz() / "src" / "tfg_multiagente_fotografia"


def preparar_sys_path() -> Path:
    """Añade `src/` y `experimentos/` a `sys.path` y devuelve la raíz."""
    r = raiz()
    for d in (r / "src", r / "experimentos"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    return r


@functools.lru_cache(maxsize=1)
def cargar_config() -> dict:
    with open(raiz() / "experimentos" / "config_experimentos.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def absoluta(ruta: str | Path) -> Path:
    """Una ruta de la configuración, relativa a la raíz si no es absoluta."""
    p = Path(ruta)
    return (p if p.is_absolute() else raiz() / p).resolve()


def ruta_config(clave: str) -> Path:
    return absoluta(cargar_config()["rutas"][clave])


def dir_resultados(experimento: str) -> Path:
    d = ruta_config("resultados") / experimento
    d.mkdir(parents=True, exist_ok=True)
    return d


def relativa(p: Path) -> str:
    """Ruta relativa a la raíz en formato posix (clave estable entre sistemas)."""
    p = Path(p).resolve()
    return p.relative_to(raiz()).as_posix() if p.is_relative_to(raiz()) else p.as_posix()


def guardar_json(obj, ruta: Path) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(_a_json(obj), indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    return ruta


def _a_json(o):
    """Tipos de NumPy y rutas a tipos serializables; NaN e infinito a null."""
    if isinstance(o, dict):
        return {str(k): _a_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_a_json(v) for v in o]
    if hasattr(o, "tolist"):
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


def leer_csv(ruta: Path) -> pd.DataFrame:
    """CSV con separador (`,` o `;`) y codificación (utf-8 o cp1252) detectados; todo como texto."""
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(ruta, sep=None, engine="python", encoding=codificacion, dtype=str,
                               keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {ruta} ni en utf-8 ni en cp1252")


# =============================================================================
# Conjuntos de fotos
# =============================================================================

def fotos_conjunto(nombre: str, comprobar: bool = True) -> dict[str, Path]:
    """{foto: ruta} de un conjunto, en el orden de su manifiesto.

    Con `comprobar=True` (por defecto) para si falta alguna foto o su MD5 no es el del
    manifiesto: así nunca se mide o se ejecuta un fichero distinto del anotado (§3).
    """
    cfg = cargar_config()["conjuntos"][nombre]
    carpeta, manifiesto = absoluta(cfg["carpeta"]), leer_csv(absoluta(cfg["manifiesto"]))
    fotos = {Path(n).stem: carpeta / n for n in manifiesto["nombre"]}
    if comprobar:
        problemas = []
        for (foto, ruta), esperado in zip(fotos.items(), manifiesto["md5"]):
            if not ruta.exists():
                problemas.append(f"{foto}: falta {ruta}")
            elif md5(ruta) != esperado:
                problemas.append(f"{foto}: MD5 {md5(ruta)} distinto del manifiesto ({esperado})")
        if problemas:
            raise RuntimeError(f"El conjunto «{nombre}» no coincide con su manifiesto:\n  - "
                               + "\n  - ".join(problemas))
    return fotos


# =============================================================================
# Anotaciones
# =============================================================================

def normalizar_valor(v) -> str:
    """Minúsculas, sin espacios y `sí` -> `si`. Vacío o NaN -> ''."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    s = str(v).strip().lower()
    return {"sí": "si", "nan": ""}.get(s, s)


def leer_anotacion(ruta: Path, esquema: dict) -> pd.DataFrame:
    """Anotación indexada por `imagen` (sin extensión), con las columnas del esquema
    normalizadas. Para si falta una columna o hay un valor fuera del esquema."""
    df = leer_csv(ruta)
    df["imagen"] = df["imagen"].map(lambda s: Path(s).stem)
    df = df.set_index("imagen")
    errores = []
    for col, validos in esquema.items():
        if col not in df.columns:
            errores.append(f"falta la columna {col}")
            continue
        df[col] = df[col].map(normalizar_valor)
        malos = df.index[~df[col].isin(validos)].tolist()
        if malos:
            errores.append(f"{col}: valores no válidos en {malos}")
    if errores:
        raise ValueError(f"{relativa(ruta)}:\n  - " + "\n  - ".join(errores))
    return df


def anotacion_prueba() -> pd.DataFrame:
    return leer_anotacion(ruta_config("anotaciones") / "anotacion_prueba.csv",
                          ESQUEMA_ANOTACION_PRUEBA)


def segmentacion(conjunto: str) -> pd.DataFrame:
    return leer_anotacion(ruta_config("anotaciones") / f"segmentacion_{conjunto}.csv",
                          ESQUEMA_SEGMENTACION)


# =============================================================================
# Estadística y coste
# =============================================================================

def intervalo_wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson (95 % por defecto) para la proporción k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    radio = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (max(0.0, centro - radio), min(1.0, centro + radio))


def coste_usd(registro: dict, precios: dict) -> float | None:
    """Coste estimado en USD de una ejecución (`registro.json` de ejecutar_lote.py).

    Usa `precios.modelos` (USD por millón de tokens). En CrewAI, `completion_tokens` ya incluye
    los de razonamiento; en la lectura visual se suman a mano. No descuenta la caché de Gemini:
    es una cota superior. None si falta algún precio.
    """
    tabla = (precios or {}).get("modelos") or {}
    usos = [(u["modelo"], u["prompt_tokens"], u["completion_tokens"])
            for u in registro.get("tokens_llm", [])]
    lv = registro.get("tokens_lectura_visual")
    if lv:
        usos.append((lv["modelo"], lv["prompt_tokens"],
                     lv["candidates_tokens"] + lv["thoughts_tokens"]))
    total = 0.0
    for modelo, entrada, salida in usos:
        p = tabla.get(modelo.split("/")[-1]) or {}
        if p.get("entrada") is None or p.get("salida") is None:
            return None
        total += (entrada * p["entrada"] + salida * p["salida"]) / 1e6
    return total


# =============================================================================
# Constantes del sistema (leídas del código fuente, sin importarlo)
# =============================================================================

def constantes_modulo(ruta: Path) -> dict:
    """Constantes de módulo (NOMBRES_EN_MAYÚSCULAS) con valor literal o aritmético.

    Se leen con `ast` para no importar el módulo (importar `crew.py` cargaría CrewAI, YOLO y la
    ResNet). Las cadenas largas (prompts) se guardan como su MD5.
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
                    v = eval(compile(ast.Expression(valor), "<constante>", "eval"),
                             {"__builtins__": {}}, {})
                except Exception:
                    continue
            if isinstance(v, str) and len(v) > 200:
                v = "md5:" + hashlib.md5(v.encode("utf-8")).hexdigest()
            try:
                constantes[t.id] = json.loads(json.dumps(v))   # tuplas -> listas
            except (TypeError, ValueError):
                continue
    return constantes


def constantes_sistema() -> dict:
    """Constantes de `tools/*.py` y `crew.py`, por módulo."""
    base = paquete()
    modulos = [p.relative_to(base).as_posix() for p in sorted((base / "tools").glob("*.py"))
               if p.name != "__init__.py"] + ["crew.py"]
    return {m: constantes_modulo(base / m) for m in modulos}


def constante(nombre: str) -> float | str | bool:
    """El valor de una constante del sistema, buscada en todos los módulos. Si está en varios,
    tiene que valer lo mismo en todos."""
    valores = {m: c[nombre] for m, c in constantes_sistema().items() if nombre in c}
    if not valores:
        raise KeyError(f"{nombre}: no existe")
    if len({json.dumps(v) for v in valores.values()}) > 1:
        raise KeyError(f"{nombre}: valores distintos según el módulo {valores}")
    return next(iter(valores.values()))


def modelo_llm() -> dict:
    """`MODEL` de los especialistas: variable de entorno o, si no, su línea de `.env`."""
    modelo = os.environ.get("MODEL")
    env = raiz() / ".env"
    if modelo is None and env.exists():
        for linea in env.read_text(encoding="utf-8").splitlines():
            if linea.strip().startswith("MODEL="):
                modelo = linea.split("=", 1)[1].strip().strip('"').strip("'")
    return {"MODEL": modelo}


# =============================================================================
# Decisiones de aplicabilidad (E2, paso 2: replicarlas a partir de las variables)
# =============================================================================

def replicar_decisiones(medidas: pd.DataFrame) -> pd.DataFrame:
    """Las 6 decisiones (`gate_*`) recalculadas desde las variables de `medidas_*.csv`
    (calibracion.py) con las constantes actuales del código. E2 comprueba con `assert` que
    coinciden con las que tomó el código (experimentos.md §6.2)."""
    num = lambda c: pd.to_numeric(medidas[c], errors="coerce")   # noqa: E731
    yolo = medidas["fuente"] == constante("FUENTE_CON_IDENTIDAD")
    fiable = num("segmentacion_fiable") == 1
    out = pd.DataFrame(index=medidas.index)
    out["gate_color"] = num("ratio_cromatico") >= constante("RATIO_CROMATICO_MIN")
    out["gate_horizonte"] = ((num("n_lineas_horizonte") > 0)
                             & (num("longitud_horizonte") >= constante("LONGITUD_MIN_HORIZONTE"))
                             & (num("angulo_horizonte").abs() <= constante("ANGULO_MAX_HORIZONTE")))
    out["gate_convergencia"] = ((num("n_lineas_fuga") >= constante("MIN_LINEAS_FUGA"))
                                & (num("score_convergencia") > num("score_nulo"))
                                & (num("apertura_haz") >= constante("APERTURA_MIN_GRADOS")))
    out["gate_espacio"] = yolo & fiable
    out["gate_nitidez"] = yolo & fiable & (num("var_max") >= constante("VAR_LAPLACIANO_MIN"))
    out["gate_composicion"] = yolo
    return out.astype(int)


# =============================================================================
# Procedencia y congelación (§9)
# =============================================================================

def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=raiz(), capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def procedencia() -> dict:
    """Commit, estado del árbol, máquina, GPU y versiones de los paquetes."""
    versiones = {}
    for nombre in PAQUETES:
        try:
            versiones[nombre] = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            versiones[nombre] = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda, version_torch = torch.version.cuda, torch.__version__
    except ImportError:
        gpu = cuda = version_torch = None
    return {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "arbol_limpio": _git("status", "--porcelain") == "",
        "maquina": platform.node(),
        "sistema": platform.platform(),
        "python": platform.python_version(),
        "torch": version_torch, "cuda": cuda, "gpu": gpu,
        "versiones": versiones,
    }


def ficheros_congelados() -> list[Path]:
    """Ficheros cuyo MD5 fija la congelación, además del commit (§9).

    Las fotos quedan fijadas por los manifiestos (y `fotos_conjunto` las comprueba al usarlas).
    De las anotaciones solo entra la de contexto y aplicabilidad del conjunto de prueba: la de
    segmentación del conjunto de prueba se hace DESPUÉS de congelar (§5.3).
    """
    r, base = raiz(), paquete()
    candidatos = [
        *sorted((r / "models" / "clasificador_contexto").glob("*")),
        r / "models" / "yolo" / "yolo11m.pt",
        r / "models" / "sam" / "sam2.1_b.pt",
        *sorted((base / "parametros").glob("*.json")),
        *sorted((base / "config").glob("*.yaml")),
        r / "experimentos" / "config_experimentos.yaml",
        r / "experimentos" / "manifiesto_calibracion.csv",
        r / "experimentos" / "manifiesto_prueba.csv",
        ruta_config("anotaciones") / "anotacion_prueba.csv",
    ]
    return [p for p in candidatos if p.is_file()]


def estado_sistema() -> dict:
    """Lo que la congelación fija: commit, MD5, constantes (incluye modelos de Gemini, flag del
    crítico y versión del prompt de la lectura visual) y el MODEL de los especialistas."""
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
        # Hacer commit de congelacion.json o de resultados cambia HEAD: solo cuenta el código.
        cambios = _git("diff", "--name-only", congelado.get("commit", ""), "HEAD", "--",
                       *RUTAS_CODIGO)
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
        c = congelado.get("constantes", {}).get(modulo, {})
        a = actual["constantes"].get(modulo, {})
        for nombre in sorted(set(c) | set(a)):
            if c.get(nombre) != a.get(nombre):
                dif.append(f"{modulo}:{nombre}: {c.get(nombre)!r} -> {a.get(nombre)!r}")
    if congelado.get("llm") != actual["llm"]:
        dif.append(f"LLM: {congelado.get('llm')} -> {actual['llm']}")
    return dif


def ruta_congelacion() -> Path:
    return raiz() / "experimentos" / "congelacion.json"


def comprobar_congelacion(avisar: bool = True) -> bool:
    """True si el sistema actual coincide con `congelacion.json`. Si no hay congelación o algo
    difiere, avisa y devuelve False: los resultados valen como prueba, no como finales."""
    ruta = ruta_congelacion()
    if not ruta.exists():
        if avisar:
            warnings.warn("No existe experimentos/congelacion.json: el sistema no está congelado.",
                          stacklevel=2)
        return False
    dif = diferencias_estado(json.loads(ruta.read_text(encoding="utf-8")), estado_sistema())
    if dif and avisar:
        warnings.warn("El sistema difiere de la congelación:\n  - " + "\n  - ".join(dif),
                      stacklevel=2)
    return not dif


def congelar() -> Path:
    """Escribe `congelacion.json` con la lista del §9. Exige el árbol limpio, los precios
    rellenos, los manifiestos verificados y la anotación del conjunto de prueba completa."""
    problemas = []
    if _git("status", "--porcelain"):
        problemas.append("el árbol de git no está limpio (haz commit antes de congelar)")
    precios = cargar_config().get("precios") or {}
    if precios.get("fecha_consulta") is None or any(
            v.get("entrada") is None or v.get("salida") is None
            for v in (precios.get("modelos") or {}).values()):
        problemas.append("faltan las tarifas de Gemini en config_experimentos.yaml (precios:)")
    for conjunto in cargar_config()["conjuntos"]:
        try:
            fotos_conjunto(conjunto)
        except RuntimeError as e:
            problemas.append(str(e))
    try:
        anot = anotacion_prueba()
        if len(anot) != len(fotos_conjunto("prueba", comprobar=False)):
            problemas.append(f"anotacion_prueba.csv tiene {len(anot)} filas")
    except (ValueError, FileNotFoundError) as e:
        problemas.append(str(e))
    if problemas:
        raise SystemExit("No se puede congelar:\n  - " + "\n  - ".join(problemas))

    congelacion = {
        **estado_sistema(),
        "entorno": procedencia(),
        "criterios": {e: cargar_config()[e]["criterios"] for e in ("E1", "E2", "E3", "E4")},
        "precios": precios,
    }
    return guardar_json(congelacion, ruta_congelacion())


def main() -> None:
    parser = argparse.ArgumentParser(description="Congelación del sistema (experimentos.md §9)")
    parser.add_argument("orden", choices=["congelar", "comprobar"])
    args = parser.parse_args()
    if args.orden == "congelar":
        ruta = congelar()
        print(f"Congelación escrita en {relativa(ruta)}. Haz commit de ese fichero.")
    else:
        ok = comprobar_congelacion()
        print("El sistema coincide con la congelación." if ok else
              "El sistema NO coincide con la congelación (ver avisos).")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
