"""Implementa las métricas de espacio negativo y aislamiento del Agente 3."""
from pathlib import Path
from typing import Type

import cv2
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

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
# 5 iteraciones es el valor habitual de la literatura de GrabCut y el punto donde la máscara
# deja de moverse de forma apreciable.
ITER_GRABCUT = 5

# Semilla del RNG GLOBAL de OpenCV. Ver el bloque DETERMINISMO del docstring: sin esto las
# dos métricas del agente no son recomputables por un tercero.
SEMILLA_GRABCUT = 0

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
    """Convierte `[cx, cy, w, h]` al rectángulo de GrabCut en el espacio de trabajo.

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


def mascara_grabcut(img_trabajo, rect):
    """Máscara booleana de figura por GrabCut inicializado con rectángulo.

    True donde GrabCut cree que hay sujeto (primer plano seguro o probable). NUNCA avisa de un
    fallo: si el rect no contiene un sujeto devuelve igualmente una máscara, y por eso la
    aplicabilidad se corta aguas arriba con el gate en vez de confiar en el algoritmo.
    """
    # Sembrar ANTES de cada llamada, no una vez al importar: si se hiciera una sola vez, cada
    # análisis consumiría el estado que dejó el anterior y el resultado dependería del ORDEN en
    # que se procesan las imágenes. Analizar una foto suelta daría algo distinto que analizarla
    # dentro de un lote, que es un no-determinismo más sutil y peor porque no se ve.
    cv2.setRNGSeed(SEMILLA_GRABCUT)

    mascara = np.zeros(img_trabajo.shape[:2], np.uint8)
    modelo_fondo = np.zeros((1, 65), np.float64)
    modelo_figura = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        img_trabajo, mascara, rect, modelo_fondo, modelo_figura,
        ITER_GRABCUT, cv2.GC_INIT_WITH_RECT,
    )
    return np.isin(mascara, (cv2.GC_FGD, cv2.GC_PR_FGD))


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
        if sujeto_discreto:
            rect = rect_desde_bbox(bbox, original.shape, trabajo.shape)
            mascara = mascara_grabcut(trabajo, rect)
            fuente_mascara = "grabcut"
        else:
            rect = None
            mascara = mascara_saliencia(mapa_saliencia_path, trabajo.shape)
            fuente_mascara = "prior_saliencia"

        cobertura = float(np.count_nonzero(mascara)) / mascara.size
        espacio_negativo = 1.0 - cobertura
        var_fig, var_fondo, indice, hay_medicion = medir_nitidez(trabajo, mascara)

        # --- Gates ---
        # Condición común: el bbox tiene identidad semántica. `sujeto_discreto` NO se comprueba
        # aparte porque `fuente == "yolo"` ya lo implica por construcción (en
        # shared_perception_tool.py, source="yolo" solo se asigna en la rama que pone el flag a
        # True). Dos campos que codifican lo mismo invitan a que se contradigan.
        hay_identidad = fuente == FUENTE_CON_IDENTIDAD
        hay_textura = max(var_fig, var_fondo) >= VAR_LAPLACIANO_MIN

        conf_espacio = 1.0 if hay_identidad else 0.0
        conf_nitidez = 1.0 if (hay_identidad and hay_textura) else 0.0

        motivo_sin_identidad = (
            f"el sujeto no lo detecto YOLO sino el mapa de saliencia (fuente={fuente}), "
            "asi que no hay garantia de que la region segmentada sea un objeto y no una "
            "textura o una masa de fondo"
        )
        fuente_conf_espacio = None if hay_identidad else motivo_sin_identidad
        if hay_identidad and not hay_textura:
            fuente_conf_nitidez = (
                f"la escena no tiene detalle suficiente en ninguna region "
                f"(max(var_figura, var_fondo)={max(var_fig, var_fondo):.1f} por debajo de "
                f"{VAR_LAPLACIANO_MIN}), asi que comparar la densidad de detalle de dos "
                "regiones no significa nada"
            )
        elif not hay_identidad:
            fuente_conf_nitidez = motivo_sin_identidad
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
            verificacion_path=self._dibujar_debug(
                ruta_imagen, trabajo, mascara, rect, fuente_mascara,
                espacio_negativo, var_fig, var_fondo, indice,
                conf_espacio, conf_nitidez,
            ),
        )
        return informe

    def _dibujar_debug(self, ruta_imagen, trabajo, mascara, rect, fuente_mascara,
                       espacio_negativo, var_fig, var_fondo, indice,
                       conf_espacio, conf_nitidez):
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

        capa.texto([
            (f"fuente_mascara = {fuente_mascara}", BLANCO),
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
            "Flag de la percepción compartida. Elige la VÍA de segmentación: True usa GrabCut "
            "sobre el bbox, False usa el mapa de saliencia. No confundir con el gate."
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
        "Segmenta el sujeto y mide dos cosas: el espacio negativo (qué fracción del "
        "encuadre no ocupa) y la separación figura-fondo por densidad de detalle "
        "(ratio_nitidez, un índice en [-1, 1]). Necesita el bbox, el flag "
        "sujeto_discreto, la fuente y el mapa de saliencia de la percepción compartida. "
        "IMPORTANTE sobre lo que ratio_nitidez significa: mide DENSIDAD DE DETALLE, NO "
        "ENFOQUE. Un sujeto nítido pero de superficie lisa da índice negativo sin que "
        "haya ningún desenfoque, así que no se puede hablar de enfoque ni de "
        "profundidad de campo a partir de este número. "
        "IMPORTANTE sobre la confianza: las dos métricas tienen gate y solo abren si el "
        "sujeto lo detectó YOLO; si el bbox salió del mapa de saliencia la herramienta "
        "devuelve confianza 0 y entonces los valores NO son citables como hechos, por "
        "razonables que parezcan. La nitidez exige además que la escena tenga detalle "
        "suficiente en alguna región. No recalcules esas confianzas ni las contradigas: "
        "el motivo exacto del cierre viaja en fuente_confianza. "
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
