"""Implementa las métricas de espacio negativo y aislamiento del Agente 3."""
from functools import lru_cache
from pathlib import Path
from typing import Type

import cv2
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ultralytics import SAM

from tfg_multiagente_fotografia.rutas import PROYECT_PATH
from tfg_multiagente_fotografia.schemas.especialistas import (
    InformeEspacioAislamiento,
    MetricaConfianza,
)
from tfg_multiagente_fotografia.tools.capa_verificacion import (
    BLANCO,
    CIAN,
    GRIS,
    CapaVerificacion,
    color_confianza,
)
from tfg_multiagente_fotografia.tools.imagen import LADO_MAYOR_TRABAJO, cargar_imagen

SUBCARPETA = "espacio_aislamiento"

# --- Espacio de trabajo ---
# LADO_MAYOR_TRABAJO se importa de tools/imagen.py. El motivo que más pesa AQUÍ, y que no
# se aplica a los otros dos agentes: la varianza del Laplaciano DEPENDE DE LA RESOLUCIÓN
# (a más píxeles, más bordes finos que el operador recoge), así que sin un espacio fijo
# `VAR_LAPLACIANO_MIN` valdría cosas distintas en una foto de 800 px y en una de 4096.

# --- Segmentación ---
# SAM 2.1 (base) de Ultralytics con el rect del bbox como prompt. Sustituye a GrabCut, que en
# C2 devolvía máscaras vacías o desbordadas SIN AVISAR (imagen51: 0 px y las dos métricas
# abiertas con confianza 1). Además de una máscara más fiel, SAM devuelve una PUNTUACIÓN: la
# calidad que el propio modelo predice para su máscara. Eso permite un gate de calidad que
# GrabCut no podía tener. Los pesos (154 MB) se descargan solos a esta ruta la primera vez.
MODELO_SAM = PROYECT_PATH / "models" / "sam" / "sam2.1_b.pt"

# Lo que se publica en `fuente_mascara` cuando la máscara sale de SAM.
FUENTE_MASCARA_SAM = MODELO_SAM.stem

# Dispositivo FIJO, y CPU a propósito. La inferencia no tiene azar (no hay RNG que sembrar,
# a diferencia de GrabCut), pero el resultado bit a bit sí depende del backend. Medido con
# las versiones de uv.lock (ultralytics 8.4.102, torch 2.14): CPU y MPS discrepan en 1 píxel
# en imagen51. Con el dispositivo fijo, dos máquinas con las mismas versiones obtienen la
# misma máscara. En CPU cuesta ~0.9 s por imagen, despreciable frente al pipeline completo.
# OJO: cambiar la versión de ultralytics cambia la máscara (0.1-0.3 % de los píxeles entre
# 8.4.102 y 8.4.202), así que no se actualiza sin recalibrar los umbrales de este módulo.
DEVICE_SAM = "cpu"

# --- Gate de calidad de la segmentación (solo vía SAM) ---
# PROVISIONALES: todavía no hay anotación de la segmentación con la que calibrarlos con el
# criterio de huecos de VAR_LAPLACIANO_MIN. Orientación de una prueba previa sobre este mismo
# corpus: las segmentaciones buenas puntúan >= 0.97 y las fallidas revisadas a ojo <= 0.81
# (imagen5, imagen86, imagen87).
SAM_SCORE_MIN = 0.90

# Relleno = área de la máscara / área del rect. Muy bajo: SAM se quedó con un fragmento (o
# con nada, que era el fallo de GrabCut en imagen51). Casi 1: rellenó la caja entera en vez
# de recortar una figura. Es el "gate de relleno" que proponía el notebook 02. Igual de
# PROVISIONALES que el score; un objeto rectangular que llena su caja (un libro, un cuadro)
# puede rozar el techo legítimamente.
RELLENO_MIN = 0.20
RELLENO_MAX = 0.95

# Margen de frontera que se descuenta antes de medir la nitidez, como fracción del lado menor
# del espacio de trabajo. El Laplaciano se dispara justo en el borde entre figura y fondo —es
# un salto de intensidad, que es literalmente lo que el operador detecta—, así que ese borde
# inflaría las DOS varianzas a la vez y comprimiría el índice hacia 0.
MARGEN_FRONTERA_RATIO = 0.01

# --- Gate de `ratio_nitidez`: hay textura que comparar ---
# Calibrado sobre las 33 imágenes de data/. En escala LINEAL la magnitud es un continuo sin
# bandas vacías, pero en logarítmica hay dos huecos limpios:
#   5.1 (imagen18) --x5.4--> 27.6 (imagen14) --x4.2--> 116.3 (imagen24) --> resto
# Se elige el SEGUNDO (corte en 50) y no el primero (15) porque además cierra `imagen14`, que
# la revisión visual confirmó como segmentación fallida: un producto blanco sobre fondo blanco
# es una escena casi sin detalle, que es literalmente lo que esta condición describe. Ante dos
# huecos igual de limpios se toma el más estricto, mismo criterio que el percentil 99 del
# Agente 2: un falso positivo rompe la verificabilidad, un falso negativo solo hace callar.
# PENDIENTE DE RECALIBRAR sobre C2 con las máscaras de SAM (experimentos.md, §4); las cifras de
# este comentario están desactualizadas (imagen14 es ahora un tren). Valor provisional: 50.
VAR_LAPLACIANO_MIN = 50.0

# Valor de `fuente` (percepción compartida) que abre el gate. Ver el bloque de por qué.
FUENTE_CON_IDENTIDAD = "yolo"


# =============================================================================
# Cómputo (funciones de módulo: la sonda y el test las importan)
# =============================================================================

def reescalar(img):
    """Reduce al lado mayor de trabajo sin ampliar imágenes pequeñas.

    Usa `INTER_AREA` y no devuelve una escala teórica; el mapeo del bbox se hace con las
    dimensiones reales mediante `rect_desde_bbox`.
    """
    alto, ancho = img.shape[:2]
    escala = LADO_MAYOR_TRABAJO / max(alto, ancho)
    if escala >= 1.0:
        return img
    nuevo = (int(round(ancho * escala)), int(round(alto * escala)))
    return cv2.resize(img, nuevo, interpolation=cv2.INTER_AREA)


def rect_desde_bbox(bbox, shape_original, shape_trabajo):
    """Convierte `[cx, cy, w, h]` a un rectángulo `(x, y, w, h)` en el espacio de trabajo.

    Es el prompt de SAM (que lo recibe como xyxy) y el rect de la verificación visual.

    Usa las dimensiones efectivas de ambas imágenes para evitar errores de redondeo del
    reescalado y recorta el resultado a los límites de la imagen.
    """
    alto_orig, ancho_orig = shape_original[:2]
    alto, ancho = shape_trabajo[:2]
    sx, sy = ancho / ancho_orig, alto / alto_orig

    cx, cy, w, h = bbox
    x1 = max(0, int(round((cx - w / 2) * sx)))
    y1 = max(0, int(round((cy - h / 2) * sy)))
    x2 = min(ancho, int(round((cx + w / 2) * sx)))
    y2 = min(alto, int(round((cy + h / 2) * sy)))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


@lru_cache(maxsize=1)
def _modelo_sam():
    """Carga SAM una sola vez y solo cuando hace falta.

    La vía de saliencia, los casos analíticos del test y el arranque de la interfaz no pagan
    así los 154 MB ni el tiempo de carga.
    """
    MODELO_SAM.parent.mkdir(parents=True, exist_ok=True)
    return SAM(str(MODELO_SAM))


def mascara_sam(img_trabajo, rect):
    """Máscara booleana de figura por SAM 2 con el rect como prompt, y su puntuación.

    Devuelve `(mascara, score)`, con `score` en [0, 1]: la calidad que SAM predice para su
    propia máscara. Se segmenta el espacio de TRABAJO y no la original para que el contorno
    coincida píxel a píxel con las regiones donde se mide la nitidez.

    Si SAM no devuelve máscara, se devuelve una vacía con score 0: el gate de calidad la
    rechaza por relleno y por score, en vez de dejar que abra las métricas.
    """
    x, y, w, h = rect
    resultado = _modelo_sam()(
        img_trabajo,
        bboxes=[[x, y, x + w, y + h]],
        device=DEVICE_SAM,
        # conf=0.0 porque, por defecto, Ultralytics DESCARTA en silencio las máscaras con
        # score < 0.25, y entonces no habría ni máscara ni puntuación que publicar. El corte
        # lo decide el gate de este módulo, no la librería.
        conf=0.0,
        verbose=False,
    )[0]
    vacia = np.zeros(img_trabajo.shape[:2], bool)
    if resultado.masks is None or len(resultado.masks) == 0:
        return vacia, 0.0

    mascara = resultado.masks.data[0].cpu().numpy() > 0.5
    if mascara.shape != vacia.shape:  # salvaguarda: Ultralytics ya la devuelve a este tamaño
        alto, ancho = vacia.shape
        mascara = cv2.resize(
            mascara.astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_NEAREST
        ) > 0
    return mascara, float(resultado.boxes.conf[0])


def mascara_saliencia(mapa_path, shape_trabajo):
    """Máscara booleana de figura por umbral de Otsu sobre el mapa de saliencia.

    Vía para `sujeto_discreto=False`. El mapa ya está calculado y persistido por la percepción
    compartida, así que esta vía no cuesta nada. Se reescala al espacio de trabajo en vez de
    trabajar en el original para que las varianzas sean comparables con las de la otra vía.
    """
    mapa = cv2.imread(str(mapa_path), cv2.IMREAD_GRAYSCALE)
    if mapa is None:
        raise ValueError(f"No se pudo leer el mapa de saliencia: {mapa_path}")
    alto, ancho = shape_trabajo[:2]
    mapa = cv2.resize(mapa, (ancho, alto), interpolation=cv2.INTER_AREA)
    _, binaria = cv2.threshold(mapa, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binaria > 0


def _erosionar(mascara, margen):
    """Encoge una máscara booleana para quedarse con su interior."""
    if margen < 1:
        return mascara
    kernel = np.ones((margen * 2 + 1, margen * 2 + 1), np.uint8)
    return cv2.erode(mascara.astype(np.uint8), kernel, iterations=1) > 0


def medir_nitidez(img_trabajo, mascara):
    """Varianzas del Laplaciano dentro y fuera de la máscara, y su índice normalizado.

    Devuelve (var_figura, var_fondo, indice, hay_medicion).
    `indice` = (var_fig - var_fondo) / (var_fig + var_fondo) en [-1, 1].
    `hay_medicion` es False solo en el caso degenerado de varianza total nula (imagen
    absolutamente uniforme): ahí el 0.0 del índice es una convención y no una medida, así que
    quien llama debe emitir `valor_norm = None` para no publicar una adherencia inventada.
    """
    gris = cv2.cvtColor(img_trabajo, cv2.COLOR_BGR2GRAY)
    laplaciano = cv2.Laplacian(gris, cv2.CV_64F)

    alto, ancho = img_trabajo.shape[:2]
    margen = int(min(alto, ancho) * MARGEN_FRONTERA_RATIO)

    figura, fondo = _erosionar(mascara, margen), _erosionar(~mascara, margen)
    # Si erosionar vacía alguna región (sujeto minúsculo, o sujeto que ocupa casi todo), se
    # mide sin descontar la frontera: el número queda algo inflado por el borde, pero sigue
    # siendo interpretable, y es preferible a no medir.
    if not figura.any() or not fondo.any():
        figura, fondo = mascara, ~mascara

    var_fig = float(laplaciano[figura].var()) if figura.any() else 0.0
    var_fondo = float(laplaciano[fondo].var()) if fondo.any() else 0.0

    total = var_fig + var_fondo
    if total <= 0:
        return var_fig, var_fondo, 0.0, False
    return var_fig, var_fondo, (var_fig - var_fondo) / total, True


# =============================================================================
# Engine
# =============================================================================

class EspacioAislamientoEngine:
    """Calcula el informe del Agente 3 y genera su verificación visual."""

    def analizar(self, ruta_imagen, bbox, sujeto_discreto, fuente, mapa_saliencia_path):
        """Devuelve un `InformeEspacioAislamiento` ya validado por Pydantic.

        Recibe de la percepción compartida `bbox` ([cx, cy, w, h] en la imagen original),
        `sujeto_discreto` (elige la vía) y `fuente` (decide los gates). Ver el docstring del
        módulo sobre por qué esas dos últimas cosas son distintas.
        """
        original = cargar_imagen(ruta_imagen)
        alto_orig, ancho_orig = original.shape[:2]
        trabajo = reescalar(original)

        # --- Máscara de figura: la vía la elige `sujeto_discreto`, no `fuente` ---
        # La vía SAM también recibe los bbox de `fuente="saliencia"` con blob plausible (son
        # `sujeto_discreto=True`); su gate de identidad los cierra igualmente.
        score_sam = relleno = segmentacion_fiable = None
        if sujeto_discreto:
            rect = rect_desde_bbox(bbox, original.shape, trabajo.shape)
            mascara, score_sam = mascara_sam(trabajo, rect)
            # Cuenta toda la máscara, también lo que desborde el rect: por eso puede pasar de 1.
            relleno = float(np.count_nonzero(mascara)) / (rect[2] * rect[3])
            segmentacion_fiable = bool(
                score_sam >= SAM_SCORE_MIN and RELLENO_MIN <= relleno <= RELLENO_MAX
            )
            fuente_mascara = FUENTE_MASCARA_SAM
        else:
            rect = None
            mascara = mascara_saliencia(mapa_saliencia_path, trabajo.shape)
            fuente_mascara = "prior_saliencia"

        cobertura = float(np.count_nonzero(mascara)) / mascara.size
        espacio_negativo = 1.0 - cobertura
        var_fig, var_fondo, indice, hay_medicion = medir_nitidez(trabajo, mascara)

        # --- Gates ---
        # Condición común 1: el bbox tiene identidad semántica. `sujeto_discreto` NO se comprueba
        # aparte porque `fuente == "yolo"` ya lo implica por construcción (en
        # shared_perception_tool.py, source="yolo" solo se asigna en la rama que pone el flag a
        # True). Dos campos que codifican lo mismo invitan a que se contradigan.
        hay_identidad = fuente == FUENTE_CON_IDENTIDAD
        # Condición común 2: la máscara es fiable. Solo la vía SAM tiene con qué medirlo; en la
        # vía de saliencia (`segmentacion_fiable` None) no cierra nada, porque en datos reales
        # esa vía ya la cierra la condición 1 (fuente="yolo" implica sujeto_discreto=True).
        segmentacion_ok = segmentacion_fiable is not False
        hay_textura = max(var_fig, var_fondo) >= VAR_LAPLACIANO_MIN

        conf_espacio = 1.0 if (hay_identidad and segmentacion_ok) else 0.0
        conf_nitidez = 1.0 if (hay_identidad and segmentacion_ok and hay_textura) else 0.0

        # Motivo común a las dos métricas. Si fallan las dos condiciones se da el de identidad,
        # que es el más de fondo: sin identidad, una segmentación perfecta tampoco sería citable.
        if not hay_identidad:
            motivo_comun = (
                f"el sujeto no lo detecto YOLO sino el mapa de saliencia (fuente={fuente}), "
                "asi que no hay garantia de que la region segmentada sea un objeto y no una "
                "textura o una masa de fondo"
            )
        elif not segmentacion_ok:
            # El texto empieza siempre igual: el agente lo usa para elegir su salvedad.
            motivo_comun = (
                f"la segmentacion del sujeto no es fiable (score_segmentacion={score_sam:.3f}, "
                f"minimo {SAM_SCORE_MIN}; relleno_bbox={relleno:.3f}, rango admitido "
                f"[{RELLENO_MIN}, {RELLENO_MAX}]), asi que el contorno medido puede no ser el "
                "del sujeto"
            )
        else:
            motivo_comun = None

        fuente_conf_espacio = motivo_comun
        if motivo_comun is not None:
            fuente_conf_nitidez = motivo_comun
        elif not hay_textura:
            fuente_conf_nitidez = (
                f"la escena no tiene detalle suficiente en ninguna region "
                f"(max(var_figura, var_fondo)={max(var_fig, var_fondo):.1f} por debajo de "
                f"{VAR_LAPLACIANO_MIN}), asi que comparar la densidad de detalle de dos "
                "regiones no significa nada"
            )
        else:
            fuente_conf_nitidez = None

        # `valor_norm` de la nitidez: se emite SIEMPRE que haya medición real, incluso con el
        # gate cerrado, porque es consistente con el `valor` publicado y quien lo recompute
        # obtiene lo mismo. Solo es None en el caso degenerado de varianza total nula, donde el
        # 0.0 del índice es una convención y no una medida — es la lección del `valor_norm` del
        # Agente 2: el valor neutro de algo no medido no puede ser un número que su propia
        # fórmula contradiga.
        norm_nitidez = (indice + 1.0) / 2.0 if hay_medicion else None

        informe = InformeEspacioAislamiento(
            dimensiones_imagen=(ancho_orig, alto_orig),
            ratio_espacio_negativo=MetricaConfianza[float](
                valor=espacio_negativo,
                # None SIEMPRE y por definición: la métrica no nombra ningún patrón al que
                # adherirse. Más aire no es más adherente a nada.
                valor_norm=None,
                confianza=conf_espacio,
                fuente_confianza=fuente_conf_espacio,
            ),
            ratio_nitidez=MetricaConfianza[float](
                valor=indice,
                valor_norm=norm_nitidez,
                confianza=conf_nitidez,
                fuente_confianza=fuente_conf_nitidez,
            ),
            var_laplaciano_figura=var_fig,
            var_laplaciano_fondo=var_fondo,
            fuente_mascara=fuente_mascara,
            score_segmentacion=score_sam,
            relleno_bbox=relleno,
            segmentacion_fiable=segmentacion_fiable,
            verificacion_path=self._dibujar_debug(
                ruta_imagen, trabajo, mascara, rect, fuente_mascara,
                espacio_negativo, var_fig, var_fondo, indice,
                conf_espacio, conf_nitidez, score_sam, relleno, segmentacion_fiable,
            ),
        )
        return informe

    def _dibujar_debug(self, ruta_imagen, trabajo, mascara, rect, fuente_mascara,
                       espacio_negativo, var_fig, var_fondo, indice,
                       conf_espacio, conf_nitidez, score_sam, relleno, segmentacion_fiable):
        """Verificación visual. Lo que hay que poder juzgar a ojo es UNA cosa: si el contorno
        rodea al sujeto o si la segmentación se fue por otro lado. Ninguna cifra del informe
        dice eso, y es lo que ha decidido la calibración de este agente.

        Se dibuja sobre el espacio de TRABAJO y no sobre la original a propósito: es la imagen
        que el engine realmente procesa, así que el contorno coincide píxel a píxel con lo que
        se ha medido.
        """
        capa = CapaVerificacion(ruta_imagen, SUBCARPETA, img=trabajo)

        if rect is not None:
            x, y, w, h = rect
            capa.rectangulo((x, y), (x + w, y + h), GRIS)

        contornos, _ = cv2.findContours(
            mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(capa.img, contornos, -1, CIAN, capa.grosor)

        # Dos líneas cortas y no una: en las fotos verticales una sola se sale del encuadre.
        if segmentacion_fiable is None:
            lineas_segmentacion = [("segmentacion sin medida (via saliencia)", BLANCO)]
        else:
            color_seg = color_confianza(1.0 if segmentacion_fiable else 0.0)
            lineas_segmentacion = [
                (f"score_segmentacion = {score_sam:.3f}", color_seg),
                (f"relleno_bbox = {relleno:.3f} -> "
                 f"{'fiable' if segmentacion_fiable else 'NO fiable'}", color_seg),
            ]

        capa.texto([
            (f"fuente_mascara = {fuente_mascara}", BLANCO),
            *lineas_segmentacion,
            (f"ratio_espacio_negativo = {espacio_negativo:.3f}"
             f"  (sujeto = {1 - espacio_negativo:.3f})", color_confianza(conf_espacio)),
            (f"var_figura = {var_fig:.1f}   var_fondo = {var_fondo:.1f}", BLANCO),
            (f"ratio_nitidez = {indice:+.3f}", color_confianza(conf_nitidez)),
            (f"gate espacio = {'ABIERTO' if conf_espacio else 'cerrado'}",
             color_confianza(conf_espacio)),
            (f"gate nitidez = {'ABIERTO' if conf_nitidez else 'cerrado'}",
             color_confianza(conf_nitidez)),
        ])
        return capa.guardar("espacio")


# =============================================================================
# Herramienta CrewAI
# =============================================================================

class EspacioAislamientoInput(BaseModel):
    # Es el ÚNICO especialista que consume la percepción compartida de forma esencial: sus dos
    # métricas se derivan de una máscara, y sin bbox ni mapa de saliencia no hay máscara.
    ruta_imagen: str = Field(
        ...,
        description="Ruta local de la imagen original analizada (.jpg, .png)",
    )
    bbox: list[float] = Field(
        ...,
        description=(
            "Bbox del sujeto tal y como lo emite la percepción compartida: [cx, cy, w, h] en "
            "píxeles de la imagen ORIGINAL, con (cx, cy) el CENTRO y no la esquina."
        ),
    )
    sujeto_discreto: bool = Field(
        ...,
        description=(
            "Flag de la percepción compartida. Elige la VÍA de segmentación: True usa SAM 2 "
            "con el bbox como prompt, False usa el mapa de saliencia. No confundir con el gate."
        ),
    )
    fuente: str = Field(
        ...,
        description=(
            "Cómo obtuvo la percepción compartida el bbox: 'yolo', 'saliencia' o "
            "'sin_sujeto_claro'. Es lo que DECIDE la confianza de las dos métricas."
        ),
    )
    mapa_saliencia_path: str = Field(
        ...,
        description="Ruta al PNG del mapa de saliencia que guardó la percepción compartida.",
    )


espacio_aislamiento_engine = EspacioAislamientoEngine()


class EspacioAislamientoTool(BaseTool):
    name: str = "Espacio y aislamiento del sujeto"
    description: str = (
        "Segmenta el sujeto (con SAM 2 si hay bbox de sujeto, o con el mapa de saliencia "
        "si no) y mide dos cosas: el espacio negativo (qué fracción del "
        "encuadre no ocupa) y la separación figura-fondo por densidad de detalle "
        "(ratio_nitidez, un índice en [-1, 1]). Necesita el bbox, el flag "
        "sujeto_discreto, la fuente y el mapa de saliencia de la percepción compartida. "
        "IMPORTANTE sobre lo que ratio_nitidez significa: mide DENSIDAD DE DETALLE, NO "
        "ENFOQUE. Un sujeto nítido pero de superficie lisa da índice negativo sin que "
        "haya ningún desenfoque, así que no se puede hablar de enfoque ni de "
        "profundidad de campo a partir de este número. "
        "IMPORTANTE sobre la confianza: las dos métricas tienen gate y solo abren si el "
        "sujeto lo detectó YOLO y la segmentación es fiable (segmentacion_fiable, que "
        "resume la puntuación de SAM y el relleno del bbox); si el bbox salió del mapa de "
        "saliencia o la segmentación no es fiable, la herramienta devuelve confianza 0 y "
        "entonces los valores NO son citables como hechos, por razonables que parezcan. La "
        "nitidez exige además que la escena tenga detalle suficiente en alguna región. No "
        "recalcules esas confianzas ni las contradigas: el motivo exacto del cierre viaja "
        "en fuente_confianza. "
        "Guarda además una imagen de verificación (contorno de la máscara y métricas) "
        "cuya ruta devuelve en verificacion_path."
    )
    args_schema: Type[BaseModel] = EspacioAislamientoInput

    def _run(self, ruta_imagen, bbox, sujeto_discreto, fuente, mapa_saliencia_path):
        # El engine devuelve el modelo Pydantic (ya validado); la tool lo serializa a dict para
        # que el LLM reciba JSON limpio. Mismo reparto que en los Agentes 1, 2 y 4: los tests
        # trabajan contra el engine.
        informe = espacio_aislamiento_engine.analizar(
            ruta_imagen, bbox, sujeto_discreto, fuente, mapa_saliencia_path
        )
        return informe.model_dump()
