"""Medida de las variables de decisión y revisión de la segmentación (experimentos.md, §4 y §5.3).

Sirve para los dos conjuntos con `--conjunto` (por defecto, `calibracion`):

    calibracion   data/        → resultados/calibracion/, anotaciones/segmentacion_calibracion.*
    prueba        data_prueba/ → resultados/prueba/,      anotaciones/segmentacion_prueba.*

Solo cambian las rutas; el procedimiento es el mismo. Tres subcomandos, en este orden:

    medir       Ejecuta las herramientas deterministas (sin LLM ni API) sobre cada foto y
                guarda, por foto, TODAS las variables de decisión de las condiciones de
                aplicabilidad: ratio cromático, longitud y ángulo del horizonte, score, score
                nulo y apertura de la convergencia, confianza de YOLO, score y relleno de SAM,
                varianzas del Laplaciano, y las métricas de composición. También genera la
                imagen de revisión de la segmentación de cada foto con fuente = yolo.
                Antes de medir comprueba que cada foto tiene el MD5 de su manifiesto: así no
                se mide un fichero distinto del anotado.
                Es reanudable: una foto ya medida con el mismo fichero y el mismo código de
                `src/` no se repite.
    plantilla   Genera el Excel de revisión de la segmentación (bien / parcial / mal) y del
                protagonista, solo para las fotos con fuente = yolo.
    exportar    Valida el Excel rellenado y escribe `anotaciones/segmentacion_{conjunto}.csv`.

CONJUNTO DE PRUEBA: la revisión de la segmentación se hace ANTES de mirar ninguna decisión del
sistema (§5.3 y §6.2). No abras `resultados/prueba/medidas_prueba.csv` hasta exportar.

Después, el análisis de huecos (propuesta de umbrales) cruza `medidas_calibracion.csv`
con esta revisión y con la anotación de aplicabilidad que ya existe para `data/`
(`resultados/aplicabilidad_vlm/anotacion_humana_A2.csv`).

REVISIÓN CIEGA A LAS CIFRAS. La imagen de verificación del Agente 3 imprime el score de
SAM, el relleno y si la segmentación es «fiable». Revisar viéndolos sesgaría el juicio,
y ese juicio es justo el que fija SAM_SCORE_MIN, RELLENO_MIN y RELLENO_MAX. Por eso aquí se
dibuja una imagen propia: la foto con la caja de YOLO (amarillo), la máscara de SAM
teñida y su contorno (cian), sin ningún número ni decisión del sistema.

Se ejecuta en la MISMA máquina que la validación (el equipo Windows, experimentos.md §2.3):
`entorno_calibracion.json` deja constancia de la máquina, las versiones y el commit.

Uso (desde la raíz del proyecto):
    uv run python experimentos/calibracion.py medir            # ~5-10 min la primera vez
    uv run python experimentos/calibracion.py medir --rehacer  # volver a medir todo
    uv run python experimentos/calibracion.py plantilla
    uv run python experimentos/calibracion.py exportar
    uv run python experimentos/calibracion.py medir --conjunto prueba   # y así con los otros
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tfg_multiagente_fotografia.rutas import OUTPUTS_PATH  # noqa: E402

DIR_EXPERIMENTOS = RAIZ / "experimentos"
DIR_ANOTACIONES = DIR_EXPERIMENTOS / "anotaciones"


@dataclass(frozen=True)
class Conjunto:
    """Rutas de un conjunto de fotos. Es lo único que cambia entre calibración y prueba."""
    nombre: str
    titulo: str                   # para los textos de la plantilla
    carpeta: Path                 # fotos
    manifiesto: Path              # MD5 de cada foto (`nombre`, `md5`)
    anotacion_aplicabilidad: Path  # de aquí sale el nombre del protagonista anotado
    dir_resultados: Path          # versionado
    dir_salidas: Path             # regenerable, no versionado

    @property
    def dir_revision(self) -> Path:
        return self.dir_salidas / "segmentacion"

    @property
    def cache(self) -> Path:
        return self.dir_salidas / "medidas.json"

    @property
    def medidas_csv(self) -> Path:
        return self.dir_resultados / f"medidas_{self.nombre}.csv"

    @property
    def entorno_json(self) -> Path:
        return self.dir_resultados / f"entorno_{self.nombre}.json"

    @property
    def plantilla(self) -> Path:
        return DIR_ANOTACIONES / f"segmentacion_{self.nombre}.xlsx"

    @property
    def segmentacion_csv(self) -> Path:
        return DIR_ANOTACIONES / f"segmentacion_{self.nombre}.csv"


CONJUNTOS = {
    "calibracion": Conjunto(
        nombre="calibracion", titulo="conjunto de calibración",
        carpeta=RAIZ / "data",
        manifiesto=DIR_EXPERIMENTOS / "manifiesto_calibracion.csv",
        anotacion_aplicabilidad=(DIR_EXPERIMENTOS / "resultados" / "aplicabilidad_vlm"
                                 / "anotacion_humana_A2.csv"),
        dir_resultados=DIR_EXPERIMENTOS / "resultados" / "calibracion",
        dir_salidas=OUTPUTS_PATH / "calibracion"),
    "prueba": Conjunto(
        nombre="prueba", titulo="conjunto de prueba",
        carpeta=RAIZ / "data_prueba",
        manifiesto=DIR_EXPERIMENTOS / "manifiesto_prueba.csv",
        anotacion_aplicabilidad=DIR_ANOTACIONES / "anotacion_prueba.csv",
        dir_resultados=DIR_EXPERIMENTOS / "resultados" / "prueba",
        dir_salidas=OUTPUTS_PATH / "prueba"),
}

# Nombres del conjunto de calibración que importa analisis_calibracion.py.
_CALIBRACION = CONJUNTOS["calibracion"]
DIR_CALIBRACION = _CALIBRACION.carpeta
DIR_RESULTADOS = _CALIBRACION.dir_resultados
MEDIDAS_CSV = _CALIBRACION.medidas_csv
SEGMENTACION_CSV = _CALIBRACION.segmentacion_csv
ANOTACION_APLICABILIDAD = _CALIBRACION.anotacion_aplicabilidad

# Imagen de revisión: lado mayor del fichero (para abrirlo) y de la miniatura del Excel.
LADO_REVISION = 1024
LADO_MINIATURA = 220
ALTO_FILA_PT = 170

AMARILLO = (0, 215, 255)
CIAN = (255, 255, 0)

CAMPOS = [
    "imagen", "md5", "ancho", "alto",
    # Percepción
    "fuente", "sujeto_discreto", "clase_yolo", "conf_yolo",
    "clase_yolo_max", "conf_yolo_max", "blob_area_ratio",
    # Luz y tono
    "ratio_cromatico", "gate_color",
    # Líneas: horizonte
    "longitud_horizonte", "angulo_horizonte", "n_lineas_horizonte", "gate_horizonte",
    "motivo_horizonte",
    # Líneas: convergencia
    "score_convergencia", "score_nulo", "apertura_haz", "n_lineas_fuga", "n_lineas_inliers",
    "gate_convergencia", "motivo_convergencia",
    # Espacio y aislamiento
    "score_sam", "relleno_bbox", "segmentacion_fiable", "var_figura", "var_fondo", "var_max",
    "ratio_espacio_negativo", "ratio_nitidez", "gate_espacio", "gate_nitidez",
    # Composición
    "d_tercios", "d_centro", "patron_dominante", "margen_patron", "d_equilibrio",
    "gate_composicion",
]


# =============================================================================
# Utilidades
# =============================================================================

def numero_imagen(ruta: Path) -> int:
    digitos = "".join(c for c in ruta.stem if c.isdigit())
    return int(digitos) if digitos else 10**9


def fotos(carpeta: Path) -> list[Path]:
    return sorted(
        (p for p in carpeta.iterdir()
         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: (numero_imagen(p), p.name),
    )


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def num(x, decimales=6):
    """Redondea para el CSV sin convertir None en número."""
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    return round(float(x), decimales)


def abre(metrica) -> int:
    return int(metrica.confianza >= 1.0)


def version_codigo() -> str:
    """Identifica el código de `src/` con el que se mide: el árbol de git de `src/` en HEAD
    y, si hay cambios sin commit, un resumen de esos cambios. Cambia solo si cambia `src/`,
    no con cualquier commit (de documentación, de resultados…)."""
    try:
        arbol = subprocess.run(["git", "rev-parse", "HEAD:src"], cwd=RAIZ, capture_output=True,
                               text=True, check=True).stdout.strip()
        cambios = subprocess.run(["git", "diff", "HEAD", "--", "src"], cwd=RAIZ,
                                 capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "desconocida"
    return arbol + (f"+{hashlib.md5(cambios).hexdigest()[:12]}" if cambios else "")


def comprobar_manifiesto(conjunto: Conjunto, rutas: list[Path]) -> None:
    """Para si alguna foto no es la del manifiesto (otro fichero con el mismo nombre)."""
    if not conjunto.manifiesto.exists():
        raise SystemExit(f"Falta {conjunto.manifiesto}: sin él no se puede comprobar que se "
                         "miden las fotos anotadas.")
    esperado = {Path(f["nombre"]).stem: f["md5"] for f in leer_csv(conjunto.manifiesto)}
    presentes = {r.stem for r in rutas}
    problemas = [f"{n}: falta en {conjunto.carpeta}" for n in sorted(esperado.keys() - presentes)]
    problemas += [f"{n}: no está en el manifiesto" for n in sorted(presentes - esperado.keys())]
    problemas += [f"{r.name}: MD5 {md5(r)} distinto del manifiesto ({esperado[r.stem]})"
                  for r in rutas if r.stem in esperado and md5(r) != esperado[r.stem]]
    if problemas:
        print(f"Las fotos de {conjunto.carpeta} no coinciden con {conjunto.manifiesto.name}:")
        for p in problemas:
            print(f"  - {p}")
        raise SystemExit("No se mide: se estaría midiendo un fichero distinto del anotado.")


def commit_actual() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                             text=True, check=True).stdout.strip()
        sucio = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=RAIZ,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+cambios_en_src" if sucio else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def entorno() -> dict:
    import torch
    import ultralytics

    from tfg_multiagente_fotografia.tools import espacio_aislamiento_tool as ea
    from tfg_multiagente_fotografia.tools import lineas_direccion_tool as ld
    from tfg_multiagente_fotografia.tools import luz_tono_tool as lt
    from tfg_multiagente_fotografia.tools import shared_perception_tool as sp

    def constantes(modulo, nombres):
        return {n: getattr(modulo, n) for n in nombres if hasattr(modulo, n)}

    return {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit_actual(),
        "maquina": platform.node(),
        "sistema": platform.platform(),
        "procesador": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "constantes_al_medir": {
            "percepcion": constantes(sp, ["CONF_YOLO_THRESHOLD", "BLOB_AREA_MIN_RATIO",
                                          "BLOB_AREA_MAX_RATIO"]),
            "luz_tono": constantes(lt, ["RATIO_CROMATICO_MIN", "PESO_MIN_MATIZ",
                                        "UMBRAL_FUSION_MATIZ", "SAT_MIN", "VAL_MIN"]),
            "lineas": constantes(ld, ["LONGITUD_MIN_HORIZONTE", "ANGULO_MAX_HORIZONTE",
                                      "APERTURA_MIN_GRADOS", "MIN_LINEAS_FUGA"]),
            "espacio": constantes(ea, ["SAM_SCORE_MIN", "RELLENO_MIN", "RELLENO_MAX",
                                       "VAR_LAPLACIANO_MIN", "MARGEN_FRONTERA_RATIO",
                                       "DEVICE_SAM"]),
        },
    }


# =============================================================================
# medir
# =============================================================================

def imagen_revision(ruta: Path, bbox) -> np.ndarray:
    """Foto + caja de YOLO + máscara de SAM, SIN cifras. Usa las mismas funciones que el
    Agente 3 (mismo espacio de trabajo, mismo prompt, mismo dispositivo), así que la
    máscara es la misma que se ha medido."""
    from tfg_multiagente_fotografia.tools.espacio_aislamiento_tool import (
        mascara_sam, rect_desde_bbox, reescalar,
    )
    from tfg_multiagente_fotografia.tools.imagen import cargar_imagen

    original = cargar_imagen(ruta)
    trabajo = reescalar(original)
    rect = rect_desde_bbox(bbox, original.shape, trabajo.shape)
    mascara, _ = mascara_sam(trabajo, rect)

    vista = trabajo.copy()
    tinte = vista.copy()
    tinte[mascara] = CIAN
    vista = cv2.addWeighted(tinte, 0.35, vista, 0.65, 0)
    grosor = max(2, round(max(vista.shape[:2]) / 400))
    contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vista, contornos, -1, CIAN, grosor)
    x, y, w, h = rect
    cv2.rectangle(vista, (x, y), (x + w, y + h), AMARILLO, grosor)

    # Al lado, la foto limpia: sin ella cuesta juzgar si el contorno sigue el borde real.
    lienzo = np.hstack([trabajo, np.full((trabajo.shape[0], 12, 3), 255, np.uint8), vista])
    escala = LADO_REVISION / max(lienzo.shape[:2])
    if escala < 1:
        lienzo = cv2.resize(lienzo, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    return lienzo


def medir_una(ruta: Path, percepcion, luz, lineas) -> dict:
    from tfg_multiagente_fotografia.tools.composicion_espacial_tool import (
        composicion_espacial_engine,
    )
    from tfg_multiagente_fotografia.tools.espacio_aislamiento_tool import (
        espacio_aislamiento_engine,
    )

    p = percepcion.detect(ruta)
    lt = luz.analizar(ruta)
    ld = lineas.analizar(ruta)
    ea = espacio_aislamiento_engine.analizar(
        ruta, p["bbox"], p["sujeto_discreto"], p["fuente"], p["mapa_saliencia_path"])
    ce = composicion_espacial_engine.analizar(
        ruta, p["bbox"], p["centroide_saliencia"], p["sujeto_discreto"], p["fuente"],
        p["mapa_saliencia_path"])

    # La percepción solo publica la confianza de YOLO si supera CONF_YOLO_THRESHOLD. Para el
    # barrido del umbral hace falta también la de la mejor caja cuando NO lo supera.
    img = cv2.imread(str(ruta))
    cajas = percepcion.model(img, verbose=False)[0].boxes
    if len(cajas):
        i = int(cajas.conf.argmax())
        clase_max, conf_max = percepcion.model.names[int(cajas.cls[i])], float(cajas.conf[i])
    else:
        clase_max, conf_max = None, None

    alto, ancho = img.shape[:2]
    return {
        "imagen": ruta.stem, "md5": md5(ruta), "ancho": ancho, "alto": alto,
        "fuente": p["fuente"], "sujeto_discreto": int(p["sujeto_discreto"]),
        "clase_yolo": p["clase"], "conf_yolo": num(p["confianza_yolo"]),
        "clase_yolo_max": clase_max, "conf_yolo_max": num(conf_max),
        "blob_area_ratio": num(p.get("blob_area_ratio")),
        "ratio_cromatico": num(lt.ratio_pixeles_cromaticos),
        "gate_color": abre(lt.esquema_cromatico),
        "longitud_horizonte": num(ld.longitud_horizonte),
        "angulo_horizonte": num(ld.angulo_horizonte.valor),
        "n_lineas_horizonte": ld.n_lineas_horizonte,
        "gate_horizonte": abre(ld.angulo_horizonte),
        "motivo_horizonte": ld.angulo_horizonte.fuente_confianza,
        "score_convergencia": num(ld.score_convergencia.valor),
        "score_nulo": num(ld.score_convergencia_nulo),
        "apertura_haz": num(ld.apertura_haz),
        "n_lineas_fuga": ld.n_lineas_fuga, "n_lineas_inliers": ld.n_lineas_inliers,
        "gate_convergencia": abre(ld.score_convergencia),
        "motivo_convergencia": ld.score_convergencia.fuente_confianza,
        "score_sam": num(ea.score_segmentacion), "relleno_bbox": num(ea.relleno_bbox),
        "segmentacion_fiable": num(ea.segmentacion_fiable),
        "var_figura": num(ea.var_laplaciano_figura), "var_fondo": num(ea.var_laplaciano_fondo),
        "var_max": num(max(ea.var_laplaciano_figura, ea.var_laplaciano_fondo)),
        "ratio_espacio_negativo": num(ea.ratio_espacio_negativo.valor),
        "ratio_nitidez": num(ea.ratio_nitidez.valor),
        "gate_espacio": abre(ea.ratio_espacio_negativo),
        "gate_nitidez": abre(ea.ratio_nitidez),
        "d_tercios": num(ce.d_tercios.valor), "d_centro": num(ce.d_centro.valor),
        "patron_dominante": ce.patron_dominante.valor,
        "margen_patron": num(ce.margen_patron), "d_equilibrio": num(ce.d_equilibrio.valor),
        "gate_composicion": abre(ce.d_tercios),
        # Solo para la imagen de revisión; no va al CSV.
        "_bbox": p["bbox"],
    }


def medir(args) -> None:
    conjunto = CONJUNTOS[args.conjunto]
    rutas = fotos(conjunto.carpeta)
    comprobar_manifiesto(conjunto, rutas)
    version = version_codigo()
    guardadas = {}
    if conjunto.cache.exists() and not args.rehacer:
        guardadas = json.loads(conjunto.cache.read_text(encoding="utf-8"))
    # Se repite una foto si cambió el fichero o el código de src/ con el que se midió.
    pendientes = [r for r in rutas
                  if r.stem not in guardadas or guardadas[r.stem]["md5"] != md5(r)
                  or guardadas[r.stem].get("_version_codigo") != version]
    print(f"{len(rutas)} fotos en {conjunto.carpeta} · {len(pendientes)} por medir "
          f"(código de src/: {version})")

    if pendientes:
        # Importación diferida: YOLO, SAM y torch solo se cargan si hay algo que medir.
        from tfg_multiagente_fotografia.tools.lineas_direccion_tool import (
            LineasDireccionEngine,
        )
        from tfg_multiagente_fotografia.tools.luz_tono_tool import LuzTonoEngine
        from tfg_multiagente_fotografia.tools.shared_perception_tool import (
            SharedPerceptionEngine,
        )

        percepcion, luz, lineas = SharedPerceptionEngine(), LuzTonoEngine(), LineasDireccionEngine()
        conjunto.dir_revision.mkdir(parents=True, exist_ok=True)
        inicio = time.perf_counter()
        for i, ruta in enumerate(pendientes, 1):
            fila = medir_una(ruta, percepcion, luz, lineas)
            fila["_version_codigo"] = version
            revision = conjunto.dir_revision / f"{ruta.stem}.jpg"
            if fila["fuente"] == "yolo":
                cv2.imwrite(str(revision), imagen_revision(ruta, fila["_bbox"]),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                # Si antes era yolo y ya no, su imagen de revisión no debe quedar.
                revision.unlink(missing_ok=True)
            guardadas[ruta.stem] = fila
            # Se guarda tras cada foto: una interrupción no obliga a repetir lo hecho.
            conjunto.cache.write_text(json.dumps(guardadas, ensure_ascii=False, indent=1),
                                      "utf-8")
            transcurrido = time.perf_counter() - inicio
            print(f"  {i}/{len(pendientes)} {ruta.name:<14} fuente={fila['fuente']:<9} "
                  f"({transcurrido / i:.1f} s/foto)", flush=True)

    conjunto.dir_resultados.mkdir(parents=True, exist_ok=True)
    with conjunto.medidas_csv.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore",
                                  lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(guardadas[r.stem] for r in rutas)
    if pendientes or not conjunto.entorno_json.exists():
        conjunto.entorno_json.write_text(json.dumps(entorno(), ensure_ascii=False, indent=1,
                                                    default=str), "utf-8")

    n_yolo = sum(guardadas[r.stem]["fuente"] == "yolo" for r in rutas)
    print(f"Medidas: {conjunto.medidas_csv}")
    print(f"Entorno: {conjunto.entorno_json}")
    print(f"{n_yolo} fotos con fuente = yolo · imágenes de revisión en {conjunto.dir_revision}")
    if conjunto.nombre == "prueba":
        print("Recuerda: no abras el CSV de medidas hasta exportar la revisión de la segmentación.")


# =============================================================================
# plantilla / exportar (revisión de la segmentación)
# =============================================================================

def leer_csv(ruta: Path) -> list[dict]:
    # Los CSV de anotación del proyecto mezclan codificación y separador.
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            texto = ruta.read_text(encoding=codificacion)
            break
        except UnicodeDecodeError:
            continue
    dialecto = csv.Sniffer().sniff(texto.splitlines()[0], delimiters=",;")
    return list(csv.DictReader(io.StringIO(texto), dialect=dialecto))


def plantilla(args) -> None:
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.drawing.image import Image as ImagenExcel
    from openpyxl.formatting.rule import CellIsRule, FormulaRule
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    from plantilla_revision import AMBAR, AZUL, BLANCA, BORDE, ROJO, VACIO, VERDE

    conjunto = CONJUNTOS[args.conjunto]
    destino = (args.salida or conjunto.plantilla).resolve()
    if destino.exists() and not args.sobrescribir:
        raise SystemExit(f"{destino} ya existe y podría tener anotaciones. Usa --sobrescribir "
                         "(perderías lo anotado) o --salida para otra ruta.")
    if not conjunto.medidas_csv.exists():
        raise SystemExit(f"Falta {conjunto.medidas_csv.name}: ejecuta antes `medir`.")

    # Del CSV de medidas solo se leen la fuente y la clase de YOLO, que la plantilla
    # necesita; ninguna decisión del sistema pasa a la plantilla.
    medidas = leer_csv(conjunto.medidas_csv)
    yolo = [m for m in medidas if m["fuente"] == "yolo"]
    protagonista = {}
    if conjunto.anotacion_aplicabilidad.exists():
        protagonista = {Path(f["imagen"]).stem: f.get("protagonista_nombre", "")
                        for f in leer_csv(conjunto.anotacion_aplicabilidad)}

    # --conservar: se rellenan los juicios ya exportados, salvo los de --revisar (fotos cuyo
    # fichero cambió) y los de fotos que antes no eran yolo, que quedan en blanco.
    previos = {}
    if args.conservar:
        if not conjunto.segmentacion_csv.exists():
            raise SystemExit(f"--conservar necesita {conjunto.segmentacion_csv}")
        previos = {f["imagen"]: f for f in leer_csv(conjunto.segmentacion_csv)
                   if f["imagen"] not in set(args.revisar)}

    columnas = [
        ("imagen", "Imagen", 10), ("revision", "Foto | caja y máscara", 62),
        ("abrir", "Abrir", 8), ("objeto_yolo", "Objeto en la caja (YOLO)", 16),
        ("protagonista_anotado", "Protagonista que anotaste", 18),
        ("segmentacion", "¿Segmentación?", 14), ("caja_protagonista", "¿Caja = protagonista?", 14),
        ("comentario", "Comentario (opcional)", 36), ("estado", "Estado", 12),
    ]
    letra = {c: get_column_letter(j) for j, (c, _, _) in enumerate(columnas, 1)}
    ayuda = {
        "segmentacion": (
            "Mira el contorno cian sobre el sujeto de la CAJA AMARILLA:\n\n"
            "bien: el contorno sigue el borde del sujeto; como mucho se escapa algún "
            "detalle fino (pelo suelto, una pata, un asa).\n\n"
            "parcial: recoge el sujeto, pero le falta o le sobra una parte apreciable "
            "(media figura, un brazo entero, un trozo de fondo pegado).\n\n"
            "mal: no es el sujeto: máscara vacía o diminuta, rellena casi toda la caja, "
            "o se ha ido al fondo o a otro objeto."),
        "caja_protagonista": (
            "si: la caja amarilla rodea al protagonista de la foto (el que anotaste, si "
            "anotaste uno).\n\nno: la caja rodea otro objeto (una persona de paso, un "
            "objeto secundario) o la foto no tiene protagonista."),
    }

    wb = Workbook()
    ws = wb.active
    ws.title = "Revisión"
    for j, (clave, titulo, ancho) in enumerate(columnas, 1):
        ws.cell(row=1, column=j, value=clave).font = Font(color="808080", size=9)
        cab = ws.cell(row=2, column=j, value=titulo)
        cab.fill, cab.font, cab.border = AZUL, BLANCA, BORDE
        cab.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = ancho
        if clave in ayuda:
            cab.comment = Comment(ayuda[clave], "Guía", width=360, height=260)
    ws.row_dimensions[2].height = 36

    def lista(opciones, titulo):
        dv = DataValidation(type="list", formula1='"' + ",".join(opciones) + '"',
                            allow_blank=True, promptTitle=titulo,
                            prompt="Pasa el ratón por la cabecera para ver la guía")
        ws.add_data_validation(dv)
        return dv

    dv_seg = lista(["bien", "parcial", "mal"], "Segmentación")
    dv_caja = lista(["si", "no"], "Caja = protagonista")

    faltan_imagenes = []
    for i, m in enumerate(yolo, 3):
        ruta = conjunto.dir_revision / f"{m['imagen']}.jpg"
        if not ruta.exists():
            faltan_imagenes.append(m["imagen"])
            continue
        ws.row_dimensions[i].height = ALTO_FILA_PT
        ws.cell(row=i, column=1, value=m["imagen"])
        img = cv2.imread(str(ruta))
        # Cabe en la celda: ancho de la columna (~440 px) y alto de la fila (~220 px).
        escala = min(LADO_MINIATURA * 2 / img.shape[1], LADO_MINIATURA / img.shape[0])
        mini = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        ws.add_image(ImagenExcel(io.BytesIO(cv2.imencode(".png", mini)[1].tobytes())),
                     f"{letra['revision']}{i}")
        enlace = ws.cell(row=i, column=3, value="abrir")
        enlace.hyperlink = os.path.relpath(ruta, destino.parent)
        enlace.font = Font(color="0563C1", underline="single")
        ws.cell(row=i, column=4, value=m["clase_yolo"])
        ws.cell(row=i, column=5, value=protagonista.get(m["imagen"], "") or "(sin protagonista)")
        seg, caja = f"{letra['segmentacion']}{i}", f"{letra['caja_protagonista']}{i}"
        ws.cell(row=i, column=len(columnas), value=(
            f'=IF(COUNTA({seg},{caja})=0,"pendiente",'
            f'IF(COUNTA({seg},{caja})<2,"incompleta","completa"))'))
        for clave, _, _ in columnas:
            celda = ws[f"{letra[clave]}{i}"]
            celda.border = BORDE
            celda.alignment = Alignment(vertical="center", wrap_text=True)
        dv_seg.add(ws[seg])
        dv_caja.add(ws[caja])
        if m["imagen"] in previos:
            previo = previos[m["imagen"]]
            ws[seg] = previo["segmentacion"]
            ws[caja] = previo["caja_protagonista"]
            ws[f"{letra['comentario']}{i}"] = previo.get("comentario") or None

    ultima = len(yolo) + 2
    zona_seg = f"{letra['segmentacion']}3:{letra['segmentacion']}{ultima}"
    ws.conditional_formatting.add(zona_seg, CellIsRule(operator="equal", formula=['"bien"'], fill=VERDE))
    ws.conditional_formatting.add(zona_seg, CellIsRule(operator="equal", formula=['"parcial"'], fill=AMBAR))
    ws.conditional_formatting.add(zona_seg, CellIsRule(operator="equal", formula=['"mal"'], fill=ROJO))
    zona = f"{letra['segmentacion']}3:{letra['caja_protagonista']}{ultima}"
    ws.conditional_formatting.add(zona, FormulaRule(formula=[f"LEN({letra['segmentacion']}3)=0"], fill=VACIO))
    zona_caja = f"{letra['caja_protagonista']}3:{letra['caja_protagonista']}{ultima}"
    ws.conditional_formatting.add(zona_caja, CellIsRule(operator="equal", formula=['"si"'], fill=VERDE))
    ws.conditional_formatting.add(zona_caja, CellIsRule(operator="equal", formula=['"no"'], fill=ROJO))
    estado = f"{letra['estado']}3:{letra['estado']}{ultima}"
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"completa"'], fill=VERDE))
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"incompleta"'], fill=AMBAR))
    ws.freeze_panes = f"{letra['objeto_yolo']}3"

    guia = wb.create_sheet("Guía")
    guia["A1"] = f"Revisión de la segmentación ({conjunto.titulo})"
    guia["A1"].font = Font(bold=True, size=14)
    textos = [
        "Cada imagen tiene dos paneles: a la izquierda la foto limpia; a la derecha, la caja "
        "de YOLO (amarillo) y la máscara de SAM (tinte y contorno cian). Abre la imagen "
        "grande con «abrir» si la miniatura no basta.",
        "Juzga SOLO si la máscara recoge el sujeto de la caja, no si la foto es buena ni si "
        "la caja está bien elegida (eso es la otra columna).",
        "La imagen no muestra a propósito ninguna cifra (score, relleno) ni si el sistema "
        "consideró la segmentación fiable: tu juicio es el que fija esos umbrales.",
        *[f"{c}: {t}" for c, t in ayuda.items()],
        "Al terminar: uv run python experimentos/calibracion.py exportar"
        + ("" if conjunto.nombre == "calibracion" else f" --conjunto {conjunto.nombre}"),
    ]
    for k, texto in enumerate(textos, 3):
        celda = guia.cell(row=k, column=1, value=texto)
        celda.alignment = Alignment(wrap_text=True, vertical="top")
        guia.row_dimensions[k].height = 30 if "\n" not in texto else 150
    guia.column_dimensions["A"].width = 110

    destino.parent.mkdir(parents=True, exist_ok=True)
    wb.save(destino)
    print(f"{len(yolo) - len(faltan_imagenes)} fotos con fuente = yolo · plantilla: {destino}")
    if args.conservar:
        en_blanco = [m["imagen"] for m in yolo if m["imagen"] not in previos]
        print(f"Rellenadas con la revisión anterior: {len(yolo) - len(en_blanco)} · "
              f"POR REVISAR: {', '.join(en_blanco) or 'ninguna'}")
    if faltan_imagenes:
        print(f"AVISO: sin imagen de revisión (vuelve a ejecutar `medir`): {faltan_imagenes}")


def exportar(args) -> None:
    from openpyxl import load_workbook

    conjunto = CONJUNTOS[args.conjunto]
    origen = (args.plantilla or conjunto.plantilla).resolve()
    if not origen.exists():
        raise SystemExit(f"No existe {origen}. Genera antes la plantilla.")
    ws = load_workbook(origen)["Revisión"]
    claves = [("" if c.value is None else str(c.value).strip()) for c in ws[1]]
    col = {c: j for j, c in enumerate(claves) if c}

    def celda(fila, clave):
        valor = fila[col[clave]]
        return "" if valor is None else str(valor).strip()

    filas, errores = [], []
    for fila in ws.iter_rows(min_row=3, values_only=True):
        imagen = celda(fila, "imagen")
        if not imagen:
            continue
        seg = celda(fila, "segmentacion").lower()
        caja = celda(fila, "caja_protagonista").lower().replace("sí", "si")
        malas = [n for n, v, ok in (("segmentacion", seg, {"bien", "parcial", "mal"}),
                                    ("caja_protagonista", caja, {"si", "no"})) if v not in ok]
        if malas:
            errores.append(f"{imagen}: falta o no es válido: {', '.join(malas)}")
            continue
        filas.append({"imagen": imagen, "clase_yolo": celda(fila, "objeto_yolo"),
                      "segmentacion": seg, "caja_protagonista": caja,
                      "comentario": celda(fila, "comentario")})

    # Tienen que estar revisadas exactamente las fotos yolo de la última medida: si se volvió
    # a medir después de generar la plantilla, puede haber fotos de más o de menos.
    if conjunto.medidas_csv.exists():
        yolo = {m["imagen"] for m in leer_csv(conjunto.medidas_csv) if m["fuente"] == "yolo"}
        revisadas = {f["imagen"] for f in filas} | {e.split(":")[0] for e in errores}
        errores += [f"{n}: es yolo en {conjunto.medidas_csv.name} y no está en la plantilla"
                    for n in sorted(yolo - revisadas)]
        errores += [f"{n}: está en la plantilla pero ya no es yolo (regenera la plantilla)"
                    for n in sorted(revisadas - yolo)]
    if not filas and not errores:
        errores.append("la hoja «Revisión» no tiene ninguna foto")
    if errores:
        print("La revisión no está lista para exportar:")
        for e in errores:
            print(f"  - {e}")
        raise SystemExit(1)

    destino = (args.salida or conjunto.segmentacion_csv).resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    resumen = {v: sum(f["segmentacion"] == v for f in filas) for v in ("bien", "parcial", "mal")}
    print(f"{len(filas)} fotos exportadas · {resumen} · {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="orden", required=True)
    # --conjunto se acepta en cada subcomando (`medir --conjunto prueba`).
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--conjunto", choices=sorted(CONJUNTOS), default="calibracion",
                       help="conjunto de fotos (por defecto, calibracion)")

    m = sub.add_parser("medir", parents=[comun],
                       help="ejecuta las herramientas y guarda las variables de decisión")
    m.add_argument("--rehacer", action="store_true",
                   help="volver a medir todas las fotos aunque estén en la caché")
    m.set_defaults(funcion=medir)

    p = sub.add_parser("plantilla", parents=[comun],
                       help="genera el Excel de revisión de la segmentación")
    p.add_argument("--salida", type=Path, default=None,
                   help="por defecto, anotaciones/segmentacion_{conjunto}.xlsx")
    p.add_argument("--sobrescribir", action="store_true")
    p.add_argument("--conservar", action="store_true",
                   help="rellenar con la revisión ya exportada (segmentacion_{conjunto}.csv)")
    p.add_argument("--revisar", nargs="+", default=[], metavar="IMAGEN",
                   help="con --conservar, fotos que se dejan en blanco para revisarlas de nuevo")
    p.set_defaults(funcion=plantilla)

    e = sub.add_parser("exportar", parents=[comun], help="valida la revisión y escribe el CSV")
    e.add_argument("--plantilla", type=Path, default=None,
                   help="por defecto, anotaciones/segmentacion_{conjunto}.xlsx")
    e.add_argument("--salida", type=Path, default=None,
                   help="por defecto, anotaciones/segmentacion_{conjunto}.csv")
    e.set_defaults(funcion=exportar)

    args = parser.parse_args()
    args.funcion(args)


if __name__ == "__main__":
    main()
