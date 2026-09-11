"""Implementa las métricas globales de horizonte y convergencia del Agente 2."""
import json
from pathlib import Path
from typing import Type

import cv2
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tfg_multiagente_fotografia.schemas.especialistas import (
    InformeLineasDireccion,
    MetricaConfianza,
)
from tfg_multiagente_fotografia.tools.capa_verificacion import (
    CapaVerificacion,
    color_confianza,
    BLANCO,
    CIAN,
    GRIS,
    MAGENTA,
    ROJO,
    VERDE,
)
from tfg_multiagente_fotografia.tools.imagen import LADO_MAYOR_TRABAJO, cargar_imagen

SUBCARPETA = "lineas_direccion"
MODELO_NULO_PATH = (
    Path(__file__).resolve().parents[1] / "parametros" / "modelo_nulo_fuga.json"
)


# =============================================================================
# Constantes calibradas para el análisis de líneas y convergencia.
# =============================================================================

# --- Espacio de trabajo ---
# LADO_MAYOR_TRABAJO se importa de tools/imagen.py, que es donde está el razonamiento
# completo. Lo que importa aquí: el reescalado es UNIFORME, y eso es innegociable
# porque deformar la relación de aspecto cambiaría los ángulos, y el ángulo ES una de
# las dos métricas de este agente.

# --- Detección ---
# El desenfoque es el mando decisivo, y no está para quitar ruido de sensor sino
# para BORRAR LA TEXTURA antes de que Canny la trace: con sigma 1.0 una masa de
# vegetación daba 1988 segmentos (más que cualquier fachada) porque Canny dibujaba
# el contorno de cada hoja; con 2.0 baja a 148 y la arquitectura conserva 189.
# Con 3.0 se pierde también la estructura. Calibrado en la Fase 1.
SIGMA_DESENFOQUE = 2.0
CANNY_T1 = 40           # umbral bajo de histéresis
CANNY_T2 = 70           # umbral alto
HOUGH_THRESHOLD = 55    # votos mínimos para aceptar una recta
# 50 px sobre 1024 = 4.9% del encuadre. Se probó 80 y se descartó: dejaba 13 de 23
# imágenes sin las dos rectas mínimas. Criterio del agente: PERMISIVO en la
# detección, ESTRICTO en el gate (el ruido de textura lo rechaza el modelo nulo).
HOUGH_MIN_LONGITUD = 50
HOUGH_MAX_HUECO = 10

# --- Reparto en franjas angulares (conos simétricos, EXCLUYENTES) ---
# Una línea alimenta a una métrica o a la otra, nunca a las dos: si no, la misma
# evidencia sostendría dos afirmaciones distintas ante el crítico.
#   |ang| <= 15  -> candidata a horizonte
#   15 < |ang| < 75 -> candidata a punto de fuga
#   |ang| >= 75  -> casi vertical, DESCARTADA (paralela al plano de imagen,
#                   intersecciones inestables; medido: incluirlas solo añadía
#                   3 falsos positivos y ningún verdadero positivo).
#
# ESTRECHAR ESTE CONO A 10° SE PROBÓ Y SE DESCARTÓ CON DATOS (Fase 3). La revisión
# visual había confirmado que los 4 candidatos más inclinados del corpus no eran
# horizontes sino estructura de la escena (patrón de muro, borde de colina, línea
# de fuga, ladera), y 10° parecía el corte natural. Medido, el cambio sale mal por
# las dos puntas:
#   - NO arregla el horizonte: el estimador se limita a elegir OTRA línea de la
#     misma escena sin horizonte, así que las 4 siguen afirmando uno con confianza
#     plena, solo que con otro ángulo (imagen20 pasó de -12.26° a +8.95°).
#   - Y rompe el gate de convergencia: las líneas de 10-15° pasan a la franja de
#     fuga, lo que ENSANCHA el haz de inliers, y `apertura_haz` es justo lo que
#     separaba convergencia de paralelismo. En imagen17 (vegetación) la apertura
#     saltó de 10.6° a 70.0° y el gate se abrió; de 4 imágenes abriendo se pasó a
#     10, incluidos un retrato y un producto — exactamente los falsos positivos que
#     motivaron el descarte de la regla en Aclaraciones §4.
# La lección, que vale para todo el sistema: este umbral NO es solo del horizonte,
# define la PARTICIÓN, y por tanto sostiene también la calibración del punto de
# fuga. La aplicabilidad del horizonte se acota en su gate, no aquí.
# (Tocarlo invalida además parametros/modelo_nulo_fuga.json y obliga a regenerarlo; la
# guarda de `cargar_modelo_nulo()` lo detecta.)
UMBRAL_HORIZONTE_GRADOS = 15.0
UMBRAL_VERTICAL_GRADOS = 75.0

# --- Núcleo de consenso (punto de fuga) ---
# Criterio de inlier en GRADOS y no en píxeles a propósito: un umbral de t píxeles
# equivale a exigir un error angular que decae como 1/distancia, así que el mismo
# número significa cosas distintas según dónde caiga el punto de fuga.
UMBRAL_INLIER_GRADOS = 2.0
# Dos líneas casi paralelas se cortan en un punto numéricamente basura. Criterio
# ANGULAR, invariante a la longitud del segmento y a la resolución (un `abs(det)`
# pequeño puede significar "casi paralelas" o simplemente "segmentos cortos").
UMBRAL_PAR_MIN_GRADOS = 3.0
# Un punto de fuga fuera del encuadre es legítimo, pero pasado cierto punto es
# indistinguible de líneas paralelas. Ninguna perspectiva real de data/ pasa de 0.83.
MAX_DIAGONALES_FUGA = 2.0
# Tercera condición del gate: separa CONVERGENCIA de PARALELISMO. Calibrado sobre
# data/: perspectivas reales 42-90°, falsos positivos por paralelismo 9-30°.
APERTURA_MIN_GRADOS = 35.0
# Techo de MEMORIA, no de diseño (la matriz de votación es pares x líneas). Sobre
# data/ nunca llega a actuar: el máximo observado son 201 candidatas.
MAX_LINEAS_HIPOTESIS = 250
# Mínimo aritmético: hacen falta 2 rectas para tener una intersección.
MIN_LINEAS_FUGA = 2

# --- Gate del horizonte (dos condiciones, ver `gate_horizonte`) ---
# Ojo: estos dos umbrales viven en el GATE, no en la partición. Cambiarlos NO
# invalida el modelo nulo, porque no alteran qué líneas alimentan al punto de fuga.
# Es la diferencia con UMBRAL_HORIZONTE_GRADOS, y es lo que hace que la
# aplicabilidad del horizonte se pueda calibrar sin tocar la Fase 2.
#
# Un horizonte real cruza buena parte del encuadre. Si la línea casi horizontal más
# larga de la escena no llega al 13% de la diagonal, lo que hay es un fragmento (el
# borde de un objeto, un trozo de sombra) y no una línea de horizonte. Calibrado
# sobre data/ con revisión visual: cierra el retrato (0.045), el producto (0.061),
# la fachada (0.074) y la masa de vegetación (0.117), donde no hay horizonte que
# medir. El corte se subió de 0.10 a 0.13 tras comprobar a ojo ese último caso, y
# de paso ensancha la banda de separación: 0.117 (cerrada) frente a 0.140 (la más
# corta de las que abren). Sigue siendo el umbral más frágil de los dos y el
# candidato natural a re-calibrar cuando crezca el dataset.
LONGITUD_MIN_HORIZONTE = 0.13
# Y aunque sea larga, una línea muy inclinada no es un horizonte mal nivelado: es
# estructura de la escena. Verificado a ojo sobre los 4 candidatos más inclinados
# del corpus, que resultaron ser un patrón de muro (-12.26°), el borde de una colina
# (-11.17°), una línea de fuga (-10.66°) y una ladera (+10.65°). Por debajo del corte
# el corpus no pasa de 8.25°, así que 10 cae en banda vacía — mismo criterio con el
# que se fijó APERTURA_MIN_GRADOS entre 30 y 42.
# NO se implementa estrechando el cono de la partición a 10°: se probó y rompe el
# gate de convergencia sin arreglar esto (ver el comentario de UMBRAL_HORIZONTE_GRADOS).
ANGULO_MAX_HORIZONTE = 10.0

# --- Nota sobre las variantes descartadas ---
# Este módulo NO lleva interruptores de ablación: hace una sola cosa y la hace
# siempre igual. Las dos variantes que se sopesaron quedaron descartadas y
# producción se queda con los valores ganadores ya fijos:
#   - hipótesis EXHAUSTIVAS: se enumeran TODOS los pares de líneas en vez de
#     sortearlos. Es tratable (C(n,2), caso peor 20.100 pares) y elimina la
#     semilla aleatoria, que es lo que importa: un `score` que depende de una
#     semilla no es recomputable por un tercero, y la premisa del sistema es que
#     el crítico cite números verificables. El propio artículo de RANSAC
#     (Fischler y Bolles, 1981) recomienda la selección determinista cuando hay
#     un criterio del problema para elegirla.
#   - gate de apertura SIEMPRE activo (ver `gate_convergencia`).


# =============================================================================
# Detección de segmentos
# =============================================================================

def reescalar(img):
    """Reduce la imagen al lado mayor de trabajo. Devuelve (imagen, escala).

    Las imágenes ya pequeñas no se amplían: inventar píxeles no añade información.
    INTER_AREA promedia al reducir, así que hace de antialiasing y no fabrica
    bordes falsos que Canny detectaría luego como estructura.
    """
    alto, ancho = img.shape[:2]
    escala = LADO_MAYOR_TRABAJO / max(alto, ancho)
    if escala >= 1.0:
        return img, 1.0
    nuevo = (int(round(ancho * escala)), int(round(alto * escala)))
    return cv2.resize(img, nuevo, interpolation=cv2.INTER_AREA), escala


def detectar_segmentos(img_bgr):
    """Gris -> desenfoque -> Canny -> Hough probabilístico.

    Devuelve (mapa_de_bordes, segmentos), con `segmentos` un array Nx4 de
    [x1, y1, x2, y2] en píxeles del ESPACIO DE TRABAJO (vacío si no hay ninguno).
    Se usa HoughLinesP y no HoughLines porque las rectas en (rho, theta) no tienen
    extremos y por tanto no tienen longitud, y `angulo_horizonte` la necesita dos
    veces: para elegir al candidato más largo y para publicar `longitud_horizonte`.
    """
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # ksize=(0,0): OpenCV deriva el tamaño del kernel de sigma.
    suavizada = cv2.GaussianBlur(gris, (0, 0), SIGMA_DESENFOQUE)
    bordes = cv2.Canny(suavizada, CANNY_T1, CANNY_T2, apertureSize=3, L2gradient=True)
    segmentos = cv2.HoughLinesP(
        bordes,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LONGITUD,
        maxLineGap=HOUGH_MAX_HUECO,
    )
    # HoughLinesP devuelve None (no un array vacío) cuando no encuentra nada.
    if segmentos is None:
        return bordes, np.empty((0, 4), dtype=np.int32)
    return bordes, segmentos.reshape(-1, 4)


def angulo_grados(segmento):
    """Ángulo del segmento respecto a la horizontal, en (-90, 90].

    Convención del informe: el eje y crece hacia ABAJO, así que un ángulo POSITIVO
    significa que el extremo derecho cae por debajo del izquierdo. 0 = horizontal
    perfecta, +-90 = vertical. Un segmento no tiene sentido de recorrido (A->B y
    B->A son la misma línea), así que se colapsan las dos mitades del círculo.
    """
    x1, y1, x2, y2 = segmento
    ang = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return ang


def longitud(segmento):
    """Longitud del segmento en píxeles del espacio de trabajo."""
    x1, y1, x2, y2 = segmento
    return float(np.hypot(x2 - x1, y2 - y1))


def clasificar(ang):
    """Reparte el segmento en el grupo que alimenta a cada métrica del Agente 2."""
    if abs(ang) <= UMBRAL_HORIZONTE_GRADOS:
        return "horizonte"
    if abs(ang) >= UMBRAL_VERTICAL_GRADOS:
        return "vertical"
    return "fuga"


def repartir(segmentos):
    """Agrupa los segmentos en las tres franjas excluyentes. Devuelve un dict."""
    grupos = {"horizonte": [], "fuga": [], "vertical": []}
    for seg in segmentos:
        grupos[clasificar(angulo_grados(seg))].append(seg)
    return grupos


# =============================================================================
# Núcleo de consenso
# =============================================================================

def preparar(segmentos):
    """Convierte [x1,y1,x2,y2] en lo que necesita la votación.

    Devuelve (rectas, medios, angulos, longitudes):
      - `rectas`: la recta infinita que contiene al segmento, en coordenadas
        homogéneas (a,b,c) con ax+by+c=0, obtenida como producto vectorial de sus
        extremos. Así la intersección de dos rectas es otro producto vectorial:
        sin divisiones y sin casos especiales para las verticales.
      - `medios`: punto medio, desde donde se mide si el segmento apunta al punto.
      - `angulos`: orientación en grados, plegada a [-90, 90).
    """
    segs = np.asarray(segmentos, dtype=np.float64).reshape(-1, 4)
    unos = np.ones(len(segs))
    p1 = np.column_stack([segs[:, 0], segs[:, 1], unos])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], unos])
    rectas = np.cross(p1, p2)

    medios = np.column_stack([(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2])
    angulos = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    angulos = (angulos + 90.0) % 180.0 - 90.0  # A->B y B->A son la misma recta
    longitudes = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return rectas, medios, angulos, longitudes


def dif_angular(a, b):
    """Diferencia entre dos ORIENTACIONES, en [0, 90].

    Doble plegado porque una recta no tiene sentido de recorrido: 179° y 1° no se
    diferencian en 178° sino en 2°. Olvidarlo es el fallo clásico de esta métrica:
    líneas perfectamente alineadas con el punto de fuga saldrían como el peor
    inlier posible.
    """
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def estimar_punto_fuga(segmentos, dimensiones):
    """Punto de fuga más votado, por consenso entre todos los pares de líneas.

    Funciona por votación: cada PAR de líneas propone su intersección como
    hipótesis, cada línea vota a favor de las hipótesis a las que apunta, y gana la
    más votada. El `score` es la fracción de líneas que la votaron.

    No lleva ninguna aleatoriedad: se enumeran todos los pares, así que dos
    ejecuciones sobre la misma foto dan exactamente el mismo `score`. El sistema no
    utiliza la variante con muestreo aleatorio.

    Devuelve SIEMPRE el mismo diccionario y nunca lanza: si no hay materia prima,
    score 0 y punto None. Decidir si eso "cuenta" es trabajo del gate, no de aquí.
    """
    ancho, alto = dimensiones
    diagonal = float(np.hypot(ancho, alto))
    n = len(segmentos)
    vacio = {
        "n_lineas": n,
        "n_inliers": 0,
        "score": 0.0,
        "punto_fuga": None,
        "inliers": np.zeros(n, dtype=bool),
        "apertura": 0.0,
    }
    if n < MIN_LINEAS_FUGA:
        return vacio

    rectas, medios, angulos, longitudes = preparar(segmentos)

    # 1. Hipótesis: TODOS los pares de las MAX_LINEAS_HIPOTESIS líneas más largas.
    # `np.triu_indices(k=1)` da los índices por encima de la diagonal, o sea cada
    # pareja una sola vez y sin emparejar una línea consigo misma.
    orden = np.argsort(-longitudes)[:MAX_LINEAS_HIPOTESIS]
    ia, ib = np.triu_indices(len(orden), k=1)
    ia, ib = orden[ia], orden[ib]
    if len(ia) == 0:
        return vacio

    valido = dif_angular(angulos[ia], angulos[ib]) >= UMBRAL_PAR_MIN_GRADOS
    ia, ib = ia[valido], ib[valido]
    if len(ia) == 0:
        return vacio

    cruces = np.cross(rectas[ia], rectas[ib])
    w = cruces[:, 2]
    finito = np.abs(w) > 1e-9  # guarda de división por cero, no criterio geométrico
    if not finito.any():
        return vacio
    puntos = cruces[finito, :2] / w[finito, None]

    cerca = np.hypot(puntos[:, 0] - ancho / 2, puntos[:, 1] - alto / 2) <= MAX_DIAGONALES_FUGA * diagonal
    puntos = puntos[cerca]
    if len(puntos) == 0:
        return vacio

    # 2. Votación: matriz (hipótesis x líneas) con el error angular de cada voto.
    dx = puntos[:, 0:1] - medios[None, :, 0]
    dy = puntos[:, 1:2] - medios[None, :, 1]
    error = dif_angular(np.degrees(np.arctan2(dy, dx)), angulos[None, :])
    # Si el punto cae encima del propio segmento, la dirección hacia él no tiene
    # sentido (cualquier ruido la gira 180°): esa línea no vota.
    lejos = np.hypot(dx, dy) > 0.01 * diagonal
    inliers = (error <= UMBRAL_INLIER_GRADOS) & lejos

    # 3. Gana la más votada. Desempates DETERMINISTAS (si no, la reproducibilidad
    # se rompe por aquí): primero menor error total, luego menor índice.
    votos = inliers.sum(axis=1)
    empatadas = np.flatnonzero(votos == votos.max())
    residuo = np.where(inliers[empatadas], error[empatadas], 0.0).sum(axis=1)
    ganadora = empatadas[int(np.argmin(residuo))]

    # Apertura angular del haz ganador: distingue CONVERGENCIA de PARALELISMO, que
    # el score no sabe separar. Un abanico de perspectiva real abarca decenas de
    # grados; un manojo de paralelas, casi 0.
    ang_in = angulos[inliers[ganadora]]
    apertura = float(dif_angular(ang_in[:, None], ang_in[None, :]).max()) if len(ang_in) > 1 else 0.0

    return {
        "n_lineas": n,
        "n_inliers": int(votos[ganadora]),
        "score": float(votos[ganadora]) / n,
        "punto_fuga": (float(puntos[ganadora, 0]), float(puntos[ganadora, 1])),
        "inliers": inliers[ganadora],
        "apertura": apertura,
    }


# =============================================================================
# Modelo nulo y gate de convergencia
# =============================================================================

def cargar_modelo_nulo():
    """Carga la tabla score_nulo(n) de parametros/modelo_nulo_fuga.json.

    Dos decisiones tomadas aquí, y las dos del mismo tipo: preferir un fallo ruidoso
    a un sistema que sigue funcionando dando una respuesta que ya no significa nada.

    (a) AUSENCIA DEL FICHERO -> se lanza, NO hay fallback. La sonda sí tenía uno
        (`UMBRAL_NULO_PROVISIONAL = 0.35`), y por eso mismo no puede sobrevivir al
        paso a producción: ese 0.35 es literalmente la constante inventada que el
        modelo nulo vino a sustituir. Si el fichero faltase y el gate se apoyara en
        un umbral fijo, el sistema seguiría emitiendo `score_convergencia` con
        confianza 1.0 en escenas sin perspectiva, el crítico las citaría como hecho
        y nada en la salida delataría que la segunda condición del gate se ha
        degradado. Es el mismo criterio que aplica `VectorPesosEngine.obtener()` al
        negarse a caer en la fila "otro" en silencio ante una etiqueta desconocida.

    (b) DESINCRONIZACIÓN -> también se lanza. La tabla mide qué score saca ESTE
        estimador sobre líneas aleatorias, así que tocar cualquier constante del
        núcleo de consenso la invalida: el gate seguiría abriendo y cerrando, pero
        comparando contra la línea base de otro algoritmo. Como el JSON guarda los
        parámetros con los que se generó y este módulo los tiene como constantes,
        comprobarlo es comparar dos diccionarios mediante una guarda automática.

    Devuelve el dict completo del JSON (contiene al menos "rejilla_n" y
    "score_nulo", que es lo que consume `score_nulo()`).
    """
    if not MODELO_NULO_PATH.exists():
        raise FileNotFoundError(
            f"No existe la tabla del modelo nulo ({MODELO_NULO_PATH}). El gate de "
            "convergencia no puede decidir sin ella, y no se sustituye por ningún umbral "
            "fijo porque eso es justo lo que la tabla vino a eliminar. Es necesario "
            "recalibrarla antes de ejecutar el sistema."
        )

    with open(MODELO_NULO_PATH, encoding="utf-8") as f:
        tabla = json.load(f)

    # Las dos listas se leen en paralelo (np.interp): si no midieran lo mismo, la
    # tabla devolvería la línea base de otro n sin dar ningún síntoma.
    if len(tabla["rejilla_n"]) != len(tabla["score_nulo"]):
        raise ValueError(
            f"{MODELO_NULO_PATH.name}: 'rejilla_n' y 'score_nulo' tienen distinta longitud "
            f"({len(tabla['rejilla_n'])} vs {len(tabla['score_nulo'])}). Regenéralo con: {regenerar}"
        )

    # Parámetros con los que se generó la tabla y que este módulo tiene que seguir
    # usando. La banda angular no es una constante suelta: es el par de cortes que
    # define la franja "fuga" en `clasificar()`, o sea las líneas que el nulo simuló.
    esperado = {
        "umbral_inlier_grados": UMBRAL_INLIER_GRADOS,
        "umbral_par_min_grados": UMBRAL_PAR_MIN_GRADOS,
        "max_diagonales_fuga": MAX_DIAGONALES_FUGA,
        "max_lineas_hipotesis": MAX_LINEAS_HIPOTESIS,
        "banda_angular_grados": [UMBRAL_HORIZONTE_GRADOS, UMBRAL_VERTICAL_GRADOS],
    }
    for clave in esperado:
        valor_tabla = tabla["parametros"][clave]
        valor_modulo = esperado[clave]
        if valor_tabla != valor_modulo:
            raise ValueError(
                f"{MODELO_NULO_PATH.name} se generó con {clave} = {valor_tabla}, pero este "
                f"módulo usa {valor_modulo}. La línea base ya no describe a este estimador, "
                f"así que el gate estaría contrastando contra otro algoritmo. "
                f"Regenera la tabla con: {regenerar}"
            )

    return tabla


def score_nulo(tabla, n):
    """Score esperable POR AZAR con `n` líneas: interpolación lineal en la rejilla.

    np.interp mantiene el valor de los extremos fuera del rango, que es justo la
    regla de lectura que se quiere (por encima del último nodo, el último valor).
    """
    return float(np.interp(n, tabla["rejilla_n"], tabla["score_nulo"]))


def gate_convergencia(resultado, nulo):
    """Gate de `score_convergencia`. Devuelve (confianza, fuente_confianza).

    Tres condiciones, cada una tapando un modo de fallo distinto:
      1. EXISTENCIA — materia prima suficiente para que la intersección exista;
      2. NO ES AZAR — converge más que el azar con esa misma cantidad de líneas.
         Comparación ESTRICTA a propósito: con 2 líneas score y nulo valen ambos
         1.0, y ese caso debe cerrarse (es aritmética, no convergencia);
      3. NO ES PARALELISMO — el haz se abre en abanico y no es un manojo de
         paralelas. Sin esta condición abrían el gate un muro de textura, una masa
         de vegetación y una escena de fauna: el modelo nulo no las caza porque un
         haz paralelo es MÁS ordenado que el azar, no menos.
    """
    if resultado["n_lineas"] < MIN_LINEAS_FUGA:
        return 0.0, (
            f"solo {resultado['n_lineas']} líneas candidatas (mínimo {MIN_LINEAS_FUGA}): "
            "no hay intersección que calcular"
        )
    if resultado["score"] <= nulo:
        return 0.0, (
            f"la convergencia observada ({resultado['score']:.3f}) no supera la esperable "
            f"por azar con {resultado['n_lineas']} líneas ({nulo:.3f}): sin perspectiva marcada"
        )
    if resultado["apertura"] < APERTURA_MIN_GRADOS:
        return 0.0, (
            f"haz de apertura {resultado['apertura']:.1f}° (mínimo {APERTURA_MIN_GRADOS:.0f}°): "
            "líneas casi paralelas, no convergencia en profundidad"
        )
    return 1.0, (
        f"{resultado['n_inliers']} de {resultado['n_lineas']} líneas concurren "
        f"(azar esperable {nulo:.3f}, apertura {resultado['apertura']:.1f}°)"
    )


# =============================================================================
# Nivelación del horizonte  [NUEVO — no existe en ninguna sonda]
# =============================================================================

def estimar_horizonte(segmentos_horizonte, diagonal_trabajo):
    """Elige el candidato a horizonte y mide su desviación respecto a la horizontal.

    Mucho más simple que el punto de fuga: no hay consenso, ni votación, ni modelo
    nulo. La materia prima ya viene filtrada por `clasificar()`, así que todos los
    segmentos cumplen |ángulo| <= UMBRAL_HORIZONTE_GRADOS y solo hay que quedarse
    con el MÁS LARGO, que es el criterio fijado en el contrato: la línea larga y
    casi horizontal de una foto es el horizonte, y un fragmento corto es ruido.

    El ángulo se devuelve CON SIGNO (ver el convenio en `angulo_grados`): es lo que
    permite al diagnóstico decir hacia qué lado se inclina la escena y no solo
    cuánto. La longitud se devuelve como FRACCIÓN DE LA DIAGONAL y no en píxeles,
    porque el píxel del espacio de trabajo no significa nada fuera de este módulo.

    SIN CANDIDATOS la métrica se emite igualmente con valores neutros, nunca con una
    excepción: ningún especialista puede silenciar una métrica, porque eso deja un
    agujero que el crítico rellenaría alucinando. Quien marca que no es citable es
    `gate_horizonte()`, igual que el dict `vacio` de `estimar_punto_fuga`.
    """
    if len(segmentos_horizonte) == 0:
        return {
            "n_candidatos": 0,
            "segmento": None,
            "angulo": 0.0,
            "longitud": 0.0,
            # `valor_norm=None`, no 0.0 ni 1.0. Se probaron las dos y las dos mienten:
            # 1.0 acreditaría "nivelación perfecta" sin ninguna línea que lo sostenga,
            # y 0.0 diría "inclinadísimo". Pero además —y esto lo cazó el test— un
            # número cualquiera aquí ROMPE LA AUDITABILIDAD: `valor_norm` se define
            # como 1-|valor|/cota, así que quien lo recompute desde el `valor`
            # publicado (0.0) obtendría 1.0 y encontraría una contradicción en el
            # informe. None es lo que dice la verdad: no hay medición, luego no hay
            # adherencia. Mismo criterio que `d_equilibrio` en el Agente 1, y el
            # contrato de `MetricaConfianza` ya lo admite.
            "valor_norm": None,
        }

    # `max` con `key=longitud` recorre la lista y se queda con el segmento cuya
    # longitud sea mayor. Ante un empate devuelve el primero, así que es determinista.
    ganador = max(segmentos_horizonte, key=longitud)
    angulo = angulo_grados(ganador)

    # Adherencia a la nivelación: 1 = horizonte perfecto, 0 = inclinado la cota
    # entera. Se normaliza el valor ABSOLUTO (el signo dice hacia dónde, la
    # adherencia dice cuánto). El clamp solo absorbe el error de coma flotante en
    # los extremos, igual que en `adherencia()` del Agente 1: la cota es exacta.
    valor_norm = 1.0 - abs(angulo) / UMBRAL_HORIZONTE_GRADOS
    valor_norm = min(1.0, max(0.0, valor_norm))

    return {
        "n_candidatos": len(segmentos_horizonte),
        "segmento": ganador,
        "angulo": angulo,
        "longitud": longitud(ganador) / diagonal_trabajo,
        "valor_norm": valor_norm,
    }


def gate_horizonte(horizonte):
    """Gate de `angulo_horizonte`. Devuelve (confianza, fuente_confianza).

    Es INDEPENDIENTE del gate de convergencia y no comparte nada con él: una escena
    puede tener horizonte legible y ninguna perspectiva, o al revés. Ésa es
    justamente la generalización que aporta este agente — un gate es local a cada
    métrica, no global al especialista.

    TRES condiciones, y no una: la primera versión solo comprobaba que existiera un
    candidato, y la revisión visual demostró que "existe materia prima" no es lo
    mismo que "hay evidencia" — 22 de 25 imágenes afirmaban un ángulo con confianza
    plena, incluidos retratos, bodegones y paisajes sin ningún horizonte. Es el mismo
    patrón que en la Fase 2 obligó a que el gate de convergencia tuviera tres
    condiciones en lugar de dos, y cada una tapa un modo de fallo distinto:
      1. EXISTENCIA — sin ninguna línea casi horizontal no hay nada que medir
         (niebla, cielo liso, macro, primer plano desenfocado);
      2. NO ES UN FRAGMENTO — un horizonte real cruza buena parte del encuadre; si
         el ganador no llega a LONGITUD_MIN_HORIZONTE, la afirmación se apoyaría en
         el borde de un objeto;
      3. NO ES ESTRUCTURA — una línea muy inclinada no es un horizonte torcido sino
         una ladera, un patrón o una línea de fuga. Verificado a ojo sobre las cuatro
         más inclinadas del corpus, que eran exactamente eso.
    Las dos últimas son independientes y ninguna sustituye a la otra: la longitud
    caza al retrato y al bodegón (fragmentos cortos), el ángulo caza a la ladera y al
    patrón (líneas largas pero inclinadas, hasta 0.205 de diagonal).
    """
    if horizonte["n_candidatos"] == 0:
        return 0.0, (
            f"la escena no ofrece ninguna línea a menos de {UMBRAL_HORIZONTE_GRADOS:.0f}° "
            "de la horizontal: no hay horizonte legible sobre el que medir la nivelación"
        )
    if horizonte["longitud"] < LONGITUD_MIN_HORIZONTE:
        return 0.0, (
            f"la línea casi horizontal más larga cubre solo {horizonte['longitud']:.3f} de la "
            f"diagonal (mínimo {LONGITUD_MIN_HORIZONTE:.2f}): es un fragmento suelto y no una "
            "línea de horizonte que cruce el encuadre"
        )
    if abs(horizonte["angulo"]) > ANGULO_MAX_HORIZONTE:
        return 0.0, (
            f"el candidato más largo está inclinado {horizonte['angulo']:+.2f}° (máximo "
            f"{ANGULO_MAX_HORIZONTE:.0f}°): a esa inclinación no es un horizonte desnivelado "
            "sino estructura de la escena (una ladera, un patrón o una línea de fuga)"
        )
    return 1.0, (
        f"medido sobre el más largo de {horizonte['n_candidatos']} segmentos casi "
        f"horizontales, que cubre {horizonte['longitud']:.3f} de la diagonal del encuadre"
    )


# =============================================================================
# Engine
# =============================================================================

# La tabla se carga en la primera llamada y se reutiliza, no al importar el módulo.
# Así se puede importar el estimador durante una recalibración aunque la tabla todavía
# no exista o esté desincronizada con sus constantes.
# El precio es que una tabla ausente o desincronizada salta en el primer análisis y
# no al arrancar; sigue siendo un fallo ruidoso e inmediato, que es lo que importa.
TABLA_NULO = None


def tabla_nulo():
    """Devuelve la tabla del modelo nulo, cargándola la primera vez que se pide."""
    global TABLA_NULO
    if TABLA_NULO is None:
        TABLA_NULO = cargar_modelo_nulo()
    return TABLA_NULO


class LineasDireccionEngine:
    """Mide las dos métricas geométricas globales del Agente 2.

    Devuelve el modelo Pydantic ya validado (no un dict): la validación es parte
    del contrato, y así los tests atacan al engine y no a la tool. Mismo patrón que
    `ComposicionEspacialEngine`.
    """

    def analizar(self, ruta_imagen):
        original = cargar_imagen(ruta_imagen)
        alto_orig, ancho_orig = original.shape[:2]
        trabajo, _escala = reescalar(original)
        alto_t, ancho_t = trabajo.shape[:2]
        diagonal_t = float(np.hypot(ancho_t, alto_t))

        _bordes, segmentos = detectar_segmentos(trabajo)
        grupos = repartir(segmentos)

        # --- Convergencia perspectiva ---
        fuga = grupos["fuga"]
        resultado = estimar_punto_fuga(fuga, (ancho_t, alto_t))
        nulo = score_nulo(tabla_nulo(), len(fuga))
        conf_fuga, fuente_fuga = gate_convergencia(resultado, nulo)

        # --- Nivelación del horizonte ---
        horizonte = estimar_horizonte(grupos["horizonte"], diagonal_t)
        conf_horizonte, fuente_horizonte = gate_horizonte(horizonte)

        # El punto solo se publica si el gate lo avala: una coordenada sin
        # significado invitaría al crítico a citarla como si lo tuviera. El SCORE,
        # en cambio, se publica siempre (ningún especialista silencia una métrica):
        # lo que dice que no es citable es su `confianza`, no su ausencia.
        punto = resultado["punto_fuga"]
        coord_punto_fuga = (
            (punto[0] / ancho_t, punto[1] / alto_t)
            if punto is not None and conf_fuga >= 1.0
            else None
        )

        verificacion_path = self._dibujar_debug(
            ruta_imagen=ruta_imagen,
            trabajo=trabajo,
            segmentos_fuga=fuga,
            resultado=resultado,
            nulo=nulo,
            conf_fuga=conf_fuga,
            horizonte=horizonte,
            conf_horizonte=conf_horizonte,
            n_total=len(segmentos),
        )

        return InformeLineasDireccion(
            # Dimensiones de la imagen ORIGINAL, no las del espacio de trabajo: es
            # lo que la GUI necesita para reconstruir píxeles reales, y todas las
            # magnitudes publicadas son relativas, luego válidas en ambos espacios.
            dimensiones_imagen=(ancho_orig, alto_orig),
            angulo_horizonte=MetricaConfianza[float](
                valor=horizonte["angulo"],
                valor_norm=horizonte["valor_norm"],
                confianza=conf_horizonte,
                fuente_confianza=fuente_horizonte,
            ),
            longitud_horizonte=horizonte["longitud"],
            n_lineas_horizonte=horizonte["n_candidatos"],
            score_convergencia=MetricaConfianza[float](
                valor=resultado["score"],
                # valor_norm = valor: la métrica ya nace normalizada y orientada
                # según la convención. Se emite igual por uniformidad del contrato.
                valor_norm=resultado["score"],
                confianza=conf_fuga,
                fuente_confianza=fuente_fuga,
            ),
            coord_punto_fuga=coord_punto_fuga,
            apertura_haz=resultado["apertura"],
            score_convergencia_nulo=nulo,
            n_lineas_total=len(segmentos),
            n_lineas_fuga=len(fuga),
            n_lineas_inliers=resultado["n_inliers"],
            verificacion_path=verificacion_path,
        )

    def _dibujar_debug(self, ruta_imagen, trabajo, segmentos_fuga, resultado, nulo,
                       conf_fuga, horizonte, conf_horizonte, n_total):
        """Verificación visual del Agente 2. Devuelve la ruta del PNG.

        DECISIÓN: se dibuja sobre la imagen de TRABAJO (`img=trabajo`), no sobre la
        original como hace el Agente 1. Motivo: las coordenadas de Hough ya están en
        ese espacio, así que dibujar sobre la original obligaría a deshacer `escala`
        en cada punto sin ganar nada (el PNG es material de depuración y el punto de
        fuga puede caer fuera del encuadre de todos modos). Ojo: por eso
        `capa.dimensiones` NO coincide con `dimensiones_imagen` del informe, y
        `capa.verificar_dimensiones()` no aplica aquí — la discrepancia es legítima.

        Igual que en el Agente 1, la SEMÁNTICA vive aquí (qué significa cada marca) y
        la legibilidad la resuelve CapaVerificacion (grosores y texto escalados).
        """
        capa = CapaVerificacion(ruta_imagen, SUBCARPETA, img=trabajo)
        punto = resultado["punto_fuga"]
        abre_fuga = conf_fuga >= 1.0

        # 1. Candidatas a fuga en gris; los inliers en cian y prolongados hasta el
        # punto, que es lo que deja ver de un vistazo si el consenso es real o son
        # líneas sueltas que se cruzan por casualidad.
        for seg, es_inlier in zip(segmentos_fuga, resultado["inliers"]):
            color = CIAN if es_inlier else GRIS
            capa.linea((seg[0], seg[1]), (seg[2], seg[3]), color)
            if es_inlier and punto is not None:
                medio = ((seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2)
                capa.linea(medio, punto, CIAN)

        # 2. El punto de fuga: magenta si el gate lo avala, rojo si se descartó.
        if punto is not None:
            capa.punto(punto, MAGENTA if abre_fuga else ROJO, factor=1.5)

        # 3. El candidato a horizonte, en primer plano (mismo criterio que el anclaje
        # del Agente 1: el protagonista se dibuja el último). El color codifica la
        # confianza —verde si es medible, rojo si no—, lo que permite revisar un lote
        # entero de imágenes sin leer un solo número. La horizontal gris que pasa por
        # su punto medio hace que la DESVIACIÓN se vea, en vez de haber que leerla.
        segmento_h = horizonte["segmento"]
        if segmento_h is not None:
            y_medio = (segmento_h[1] + segmento_h[3]) / 2
            capa.linea((0, y_medio), (capa.ancho, y_medio), GRIS)
            capa.linea(
                (segmento_h[0], segmento_h[1]),
                (segmento_h[2], segmento_h[3]),
                color_confianza(conf_horizonte),
            )

        # 4. Bloque de métricas y veredicto de los DOS gates (independientes).
        # `valor_norm` es None cuando no hubo ningún candidato: no hay medición, luego
        # no hay adherencia que escribir.
        if horizonte["valor_norm"] is None:
            adherencia_txt = "n/a"
        else:
            adherencia_txt = f"{horizonte['valor_norm']:.2f}"
        capa.texto([
            # n_total incluye las casi verticales descartadas, así que la suma de
            # fuga + horiz NO tiene por qué cuadrar con él: la diferencia son ellas.
            (f"n_total = {n_total}  fuga = {resultado['n_lineas']}"
             f"  horiz = {horizonte['n_candidatos']}", BLANCO),
            (f"score = {resultado['score']:.3f}   nulo = {nulo:.3f}"
             f"   apertura = {resultado['apertura']:.1f} deg", BLANCO),
            (f"gate convergencia = {'ABIERTO' if abre_fuga else 'cerrado'}",
             VERDE if abre_fuga else ROJO),
            (f"PF = {_texto_punto(punto, capa.dimensiones)}", MAGENTA if abre_fuga else ROJO),
            (f"angulo_horizonte = {horizonte['angulo']:+.2f} deg"
             f"  long = {horizonte['longitud']:.3f}  adh = {adherencia_txt}",
             color_confianza(conf_horizonte)),
            (f"gate horizonte = {'ABIERTO' if conf_horizonte >= 1.0 else 'cerrado'}",
             color_confianza(conf_horizonte)),
        ])
        return capa.guardar("lineas")


def _texto_punto(punto, dimensiones):
    """El punto de fuga en coordenadas normalizadas, como lo publica el informe."""
    if punto is None:
        return "None"
    ancho, alto = dimensiones
    return f"({punto[0] / ancho:.2f}, {punto[1] / alto:.2f}) norm"


# =============================================================================
# Herramienta CrewAI
# =============================================================================

class LineasDireccionInput(BaseModel):
    # UN SOLO CAMPO, y es una afirmación de diseño, no un olvido: el Agente 2 es el
    # único especialista totalmente desacoplado de la percepción compartida. No
    # recibe bbox, ni centroide_saliencia, ni sujeto_discreto, porque sus dos
    # métricas son geometría global sobre los segmentos de Hough.
    ruta_imagen: str = Field(
        ...,
        description="Ruta local de la imagen original analizada (.jpg, .png)",
    )


lineas_direccion_engine = LineasDireccionEngine()


class LineasDireccionTool(BaseTool):
    name: str = "Líneas y dirección"
    description: str = (
        "Analiza la geometría lineal global de la imagen: nivelación del horizonte "
        "(angulo_horizonte, en grados con signo) y convergencia perspectiva "
        "(score_convergencia y el punto de fuga donde concurren las líneas). "
        "No depende del sujeto: mide la estructura de la escena, no si las líneas apuntan "
        "a algo. Cada una de las dos métricas trae su propia `confianza` y son "
        "INDEPENDIENTES entre sí: una puede ser fiable y la otra no. "
        "IMPORTANTE: un score_convergencia alto NO significa que haya perspectiva; el "
        "estimador siempre devuelve un número y lo que autoriza a afirmarla es la "
        "confianza, que ya resuelve la herramienta con tres comprobaciones. No la "
        "recalcules ni la contradigas. "
        "Guarda además una imagen de verificación (candidatas, inliers prolongados hasta "
        "el punto de fuga y el horizonte) cuya ruta devuelve en verificacion_path."
    )
    args_schema: Type[BaseModel] = LineasDireccionInput

    def _run(self, ruta_imagen):
        # El engine devuelve el modelo Pydantic (ya validado); la tool lo serializa a
        # dict para que el LLM reciba JSON limpio y no el repr del modelo. Mismo
        # reparto que en ComposicionEspacialTool: los tests trabajan contra el engine.
        informe = lineas_direccion_engine.analizar(ruta_imagen)
        return informe.model_dump()
