"""Implementa las métricas globales de exposición y color del Agente 4."""
from typing import Type

import cv2
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from tfg_multiagente_fotografia.schemas.especialistas import (
    InformeLuzTono,
    MetricaConfianza,
)
from tfg_multiagente_fotografia.tools.capa_verificacion import (
    CapaVerificacion,
    color_confianza,
    BLANCO,
)
from tfg_multiagente_fotografia.tools.imagen import LADO_MAYOR_TRABAJO, cargar_imagen

SUBCARPETA = "luz_tono"


# =============================================================================
# Constantes calibradas para el análisis de luz y tono.
# =============================================================================

# --- Espacio de trabajo ---
# LADO_MAYOR_TRABAJO se importa de tools/imagen.py (razonamiento completo allí).

# --- Exposición (Aclaraciones §3.4.1) ---
# CIELAB: L va de 0 a 100 y es perceptualmente uniforme, que es el motivo de usarlo
# en vez del gris de BGR (donde la misma diferencia numérica significa cosas
# distintas según el tramo).
L_MIN, L_MAX = 0.0, 100.0
L_CENTRO = 50.0           # exposición centrada, referencia del `valor_norm`
CLIPPING_SOMBRAS = 5.0    # L < 5   -> sombra recortada
CLIPPING_LUCES = 95.0     # L > 95  -> luz quemada

# --- Armonía cromática (Aclaraciones §3.4.2) ---
# Filtro de píxeles acromáticos: sin él, el matiz de un gris es ruido puro (H no
# está definido cuando S -> 0) y contaminaría los grupos.
SAT_MIN = 0.15
VAL_MIN = 0.15
K_CLUSTERS = 5            # lo fija la espec; se mantiene
SEMILLA_KMEANS = 0
N_INIT_KMEANS = 10
# Masa mínima para que un grupo cuente como MATIZ DOMINANTE. Medido: sin este filtro,
# 17 de 25 imágenes salen "complementario" (68%), porque con 5 grupos sobre un
# círculo es casi imposible que NO haya un par cerca de 180°. Con el filtro baja a 8.
# NO es una condición de gate —`peso_matiz_dominante` no separa nada, va de 0.281 a
# 0.908 sin banda vacía— sino parte de la DEFINICIÓN de la métrica: un grupo sin masa
# no es un matiz dominante, es un artefacto de haber pedido 5.
PESO_MIN_MATIZ = 0.15
# Dos grupos más cercanos que esto son EL MISMO matiz y se fusionan ANTES de filtrar
# por masa. Sin este paso, k-means —obligado a devolver k grupos haya k colores o
# solo uno— parte el matiz dominante en rodajas y `matices_dominantes` publica dos
# matices donde solo hay uno; además entierra al matiz secundario real, cuyas propias
# rodajas se reparten la masa sin que ninguna llegue a PESO_MIN_MATIZ. El corte cae
# en banda vacía (rodajas del mismo matiz a 3-8°, pares legítimos a 23°+) y queda por
# DEBAJO del corte de "monocromático" (30°) para no fusionar matices que la propia
# taxonomía considera distintos. Medido también a 20°, que SÍ rompe la calibración.
UMBRAL_FUSION_MATIZ = 15.0
# El matiz se publica REDONDEADO, y no es cosmética. Medido: k-means no es bit-exacto
# entre ejecuciones aunque se fije `random_state`, porque el orden de reducción en
# coma flotante depende del reparto en hilos de BLAS/OpenMP; 13 de 25 imágenes daban
# resultados distintos entre dos ejecuciones del MISMO proceso. La discrepancia es de
# 2.8e-14 grados y nunca llegó a cambiar una etiqueta, pero `matices_dominantes` es un
# campo que el crítico CITA y un campo citable tiene que ser recomputable por un
# tercero. Redondear a 1 decimal lo resuelve (25/25) y no pierde nada real: OpenCV ya
# cuantiza H a 2° al guardarlo como H/2 en 8 bits.
DECIMALES_MATIZ = 1
# Techo de muestreo para k-means: con 1 MP de píxeles cromáticos el ajuste es lento y
# no más preciso. Se submuestrea de forma DETERMINISTA (paso fijo, no aleatoria).
MAX_PIXELES_KMEANS = 50_000

# --- Taxonomía de esquemas (cortes de Aclaraciones §3.4.2, sin tocar) ---
SPREAD_MONOCROMATICO = 30.0   # por debajo: un solo matiz
SPREAD_ANALOGO = 60.0         # por debajo: matices vecinos
COMPLEMENTARIO_CENTRO = 180.0  # par de matices opuestos...
COMPLEMENTARIO_TOLERANCIA = 30.0  # ...con esta holgura

# --- Gate de `esquema_cromatico` (única condición, ver `gate_cromatico`) ---
# Fracción mínima del encuadre con color analizable. Por debajo, lo que hay es un
# blanco y negro, una escena nocturna o una alta clave desaturada, y la etiqueta se
# estaría decidiendo sobre ruido.
# El umbral es SÓLIDO, a diferencia del LONGITUD_MIN_HORIZONTE del Agente 2 (que
# separa por 0.023): sobre data/ hay 4 imágenes en 0.000 y la siguiente en 0.287, así
# que cualquier corte intermedio da exactamente el mismo resultado.
RATIO_CROMATICO_MIN = 0.10

# Etiqueta reservada al gate cerrado. NO es una quinta categoría de la taxonomía: es
# la marca de que no había nada que clasificar, y solo aparece con confianza 0.
ESQUEMA_SIN_COLOR = "sin_color"


# =============================================================================
# Carga y espacio de trabajo
# =============================================================================

def reescalar(img):
    """Reduce la imagen al lado mayor de trabajo. Las pequeñas no se amplían.

    INTER_AREA promedia al reducir, que es justo lo que se quiere aquí: no fabrica
    colores que no estaban, así que los estadísticos de la distribución (media,
    desviación, matices dominantes) se conservan.
    """
    alto, ancho = img.shape[:2]
    escala = LADO_MAYOR_TRABAJO / max(alto, ancho)
    if escala >= 1.0:
        return img
    nuevo = (int(round(ancho * escala)), int(round(alto * escala)))
    return cv2.resize(img, nuevo, interpolation=cv2.INTER_AREA)


# =============================================================================
# Exposición y distribución tonal  [TRASLADADO — no tocar]
# =============================================================================

def medir_exposicion(img_bgr):
    """Media, desviación y clipping del canal L de CIELAB.

    OpenCV entrega L en [0,255] para imágenes de 8 bits (lo escala para que quepa en
    el tipo), así que hay que devolverlo a [0,100], que es el dominio real de CIELAB
    y el que fija el contrato. Olvidarlo daría una media_L de ~130 sobre una escala
    que el informe declara acotada en 100.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    canal_l = lab[:, :, 0].astype(np.float64) * (L_MAX / 255.0)
    total = canal_l.size
    return {
        "media_L": float(canal_l.mean()),
        "std_L": float(canal_l.std()),
        "pct_clipping_sombras": float((canal_l < CLIPPING_SOMBRAS).sum()) / total * 100.0,
        "pct_clipping_luces": float((canal_l > CLIPPING_LUCES).sum()) / total * 100.0,
    }


def adherencia_exposicion(media_l):
    """`valor_norm` de `media_L`: adherencia a la exposición centrada.

    1 = media exactamente en 50, 0 = negro o blanco totales. Misma idea que
    `adherencia()` del Agente 1 y que el `valor_norm` del horizonte en el Agente 2:
    la fórmula se documenta en el schema, de modo que un tercero puede recomputarla
    desde el `valor` publicado y encontrar el mismo número.

    El clamp solo absorbe el error de coma flotante: la cota es exacta, porque L
    está acotado en [0,100] por construcción.
    """
    valor = 1.0 - abs(media_l - L_CENTRO) / L_CENTRO
    return min(1.0, max(0.0, valor))


# =============================================================================
# Armonía cromática  [TRASLADADO — no tocar]
# =============================================================================

def dif_matiz(a, b):
    """Distancia entre dos MATICES en el círculo de color, en [0, 180].

    El matiz es circular: 350° y 10° no distan 340° sino 20°. Es el mismo plegado que
    `dif_angular` en el Agente 2, pero sobre 360 en lugar de 180, porque aquí sí hay
    sentido (el rojo y el cian son opuestos, no el mismo color).
    """
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def fusionar_matices(centros, masa, umbral=UMBRAL_FUSION_MATIZ):
    """Une aglomerativamente los grupos que son el mismo matiz (ver UMBRAL_FUSION_MATIZ).

    Une el par más cercano y repite hasta que el par más cercano supere el umbral. El
    centro resultante es la media CIRCULAR ponderada por masa (se suman los vectores
    unitarios y se vuelve con atan2), NO la media aritmética: promediar 350° y 10°
    daría 180°, que es el color opuesto al que tienen los dos.
    Determinista: el par más cercano se elige por índice y no hay ningún sorteo.
    """
    centros, masa = list(centros), list(masa)
    while len(centros) > 1:
        c = np.array(centros)
        d = dif_matiz(c[:, None], c[None, :])
        np.fill_diagonal(d, 999.0)  # un grupo nunca se fusiona consigo mismo
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] >= umbral:
            break
        m_i, m_j = masa[i], masa[j]
        r_i, r_j = np.radians(centros[i]), np.radians(centros[j])
        x = m_i * np.cos(r_i) + m_j * np.cos(r_j)
        y = m_i * np.sin(r_i) + m_j * np.sin(r_j)
        for k in sorted((i, j), reverse=True):  # descendente: no invalida el otro índice
            centros.pop(k)
            masa.pop(k)
        centros.append(float(np.degrees(np.arctan2(y, x)) % 360.0))
        masa.append(m_i + m_j)
    return np.array(centros), np.array(masa)


def clasificar_esquema(centros, spread):
    """Etiqueta el esquema comparando contra las plantillas de Aclaraciones §3.4.2.

    El orden importa: "complementario" se comprueba ANTES que "otro" pero DESPUÉS de
    los de spread bajo, porque una paleta monocromática no puede tener a la vez un par
    a 180°. Los cortes son los de la espec, sin tocar.
    """
    if spread < SPREAD_MONOCROMATICO:
        return "monocromático"
    if spread < SPREAD_ANALOGO:
        return "análogo"
    pares = dif_matiz(centros[:, None], centros[None, :])
    if np.any(np.abs(pares - COMPLEMENTARIO_CENTRO) <= COMPLEMENTARIO_TOLERANCIA):
        return "complementario"
    return "otro"


def medir_cromatica(img_bgr):
    """Matices dominantes por k-means sobre los píxeles cromáticos.

    Devuelve siempre el mismo diccionario y NUNCA lanza: si no hay píxeles
    cromáticos, devuelve la estructura con ratio 0 y sin matices. Decidir si eso
    "cuenta" es trabajo del gate, no de aquí — mismo reparto que `estimar_punto_fuga`
    con su dict `vacio` en el Agente 2.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # OpenCV: H en [0,179] (mitad de grados para caber en 8 bits), S y V en [0,255].
    matiz = hsv[:, :, 0].astype(np.float64) * 2.0
    sat = hsv[:, :, 1].astype(np.float64) / 255.0
    val = hsv[:, :, 2].astype(np.float64) / 255.0

    cromatico = (sat >= SAT_MIN) & (val >= VAL_MIN)
    n_total = matiz.size
    n_cromatico = int(cromatico.sum())
    ratio = n_cromatico / n_total

    vacio = {
        "ratio_cromatico": ratio,
        "matices": [],
        "pesos": [],
        "spread": 0.0,
        "spread_crudo": 0.0,
        "n_matices": 0,
        "peso_dominante": 0.0,
        "esquema": ESQUEMA_SIN_COLOR,
    }
    if n_cromatico < K_CLUSTERS:
        return vacio

    h = matiz[cromatico]
    # Ponderación por S*V, como fija la espec: un píxel pálido y oscuro aporta menos
    # a "cuál es el color de esta foto" que uno saturado y luminoso.
    peso = (sat[cromatico] * val[cromatico])

    # Submuestreo DETERMINISTA (paso fijo). Un `rng.choice` aquí reintroduciría por la
    # puerta de atrás la dependencia de semilla que el diseño quiere eliminar.
    if len(h) > MAX_PIXELES_KMEANS:
        paso = len(h) // MAX_PIXELES_KMEANS + 1
        h, peso = h[::paso], peso[::paso]

    # k-means sobre el matiz proyectado al CÍRCULO UNIDAD, no sobre el ángulo crudo.
    # Es imprescindible: en el ángulo crudo, 359° y 1° están a 358 de distancia, así
    # que k-means partiría en dos el rojo (el color más frecuente en fotografía) y el
    # spread saldría enorme para una imagen perfectamente monocromática.
    rad = np.radians(h)
    puntos = np.column_stack([np.cos(rad), np.sin(rad)])

    km = KMeans(
        n_clusters=K_CLUSTERS,
        random_state=SEMILLA_KMEANS,
        n_init=N_INIT_KMEANS,
    ).fit(puntos, sample_weight=peso)

    # El centroide vuelve a ángulo con atan2, que deshace la proyección sin casos
    # especiales en el cruce por 0°.
    centros = np.degrees(np.arctan2(km.cluster_centers_[:, 1], km.cluster_centers_[:, 0])) % 360.0
    etiquetas = km.labels_
    masa = np.array([peso[etiquetas == i].sum() for i in range(K_CLUSTERS)])
    masa = masa / masa.sum() if masa.sum() > 0 else masa

    # El spread CRUDO se calcula sobre los 5 grupos de k-means, antes de fusionar y
    # antes de filtrar. No se publica en el informe: es la columna de evidencia que
    # demuestra por qué hacen falta los dos pasos siguientes (sin ellos, 17 de 25
    # salían "complementario") y la consume la sonda para la memoria.
    spread_crudo = float(dif_matiz(centros[:, None], centros[None, :]).max()) if len(centros) > 1 else 0.0

    # 1) Fusionar las rodajas del mismo matiz; 2) quedarse con las que tienen masa.
    # El orden importa: fusionar DESPUÉS de filtrar dejaría fuera el matiz secundario
    # cuyas rodajas, por separado, no llegan al umbral — que es justo lo que se quiere
    # evitar.
    centros, masa = fusionar_matices(centros, masa)
    orden = np.argsort(-masa)
    # El `% 360` va DESPUÉS del redondeo y no antes, aunque `fusionar_matices` ya
    # devuelva ángulos en [0,360). Lo encontró el test de la Fase 5: un matiz de
    # 359.98° redondea a exactamente 360.0, que está FUERA del dominio [0,360) que
    # declara el schema. No es un problema aritmético —`dif_matiz` pliega igual el
    # 360 que el 0— sino de contrato: el crítico leería 360.0 y 0.0 como dos matices
    # distintos siendo el mismo rojo, y publicaría una cifra que su propio campo
    # declara imposible. Es el mismo tipo de fallo que el `valor_norm` del Agente 2:
    # un valor que contradice a la definición del campo que lo contiene.
    centros = np.round(centros[orden], DECIMALES_MATIZ) % 360.0
    masa = masa[orden]

    fuertes = centros[masa >= PESO_MIN_MATIZ]
    spread = float(dif_matiz(fuertes[:, None], fuertes[None, :]).max()) if len(fuertes) > 1 else 0.0

    return {
        "ratio_cromatico": ratio,
        "matices": [float(c) for c in fuertes],
        "pesos": [float(m) for m in masa[masa >= PESO_MIN_MATIZ]],
        "spread": spread,
        "spread_crudo": spread_crudo,
        "n_matices": int(len(fuertes)),
        "peso_dominante": float(masa[0]),
        "esquema": clasificar_esquema(fuertes, spread),
    }


def gate_cromatico(crom):
    """Gate de `esquema_cromatico`. Devuelve (confianza, fuente_confianza).

    UNA sola condición, y es una afirmación de diseño, no una versión provisional:
    la sonda midió las dos que el contrato provisional anticipaba y solo una separa.
    `peso_matiz_dominante` parecía la segunda ("si un grupo acapara la masa, la
    etiqueta la decide el ruido") y resultó NO ser un gate: va de 0.281 a 0.908 en un
    continuo sin ninguna banda vacía. Lo que sí hacía falta era usar ese peso para
    decidir qué grupos cuentan como matiz dominante, y eso cambia la MÉTRICA
    (PESO_MIN_MATIZ), no su aplicabilidad. La distinción vale para el Agente 3:
    cuando casi todo dispara, preguntarse primero si el defecto está en la medida
    antes de añadir condiciones.

    EXISTENCIA — la imagen tiene color que analizar (`ratio >= RATIO_CROMATICO_MIN`).
    Por debajo de ese piso, el filtro S/V ha tirado casi todo el encuadre y lo que
    queda es un puñado de píxeles de ruido de los que k-means saca igualmente una
    etiqueta perfectamente convincente.

    Ojo a la distinción que este gate existe para preservar, que es la sutileza de
    esta métrica: una foto en BLANCO Y NEGRO no es una foto MONOCROMÁTICA. La primera
    no tiene esquema medible (gate cerrado); la segunda tiene un matiz dominante
    fuerte y saturado (etiqueta legítima con confianza 1). Lo que las separa no es el
    número de grupos sino `ratio_pixeles_cromaticos`.
    """
    ratio = crom["ratio_cromatico"]
    if ratio < RATIO_CROMATICO_MIN:
        return 0.0, (
            f"solo el {ratio * 100:.1f}% del encuadre tiene color analizable (mínimo "
            f"{RATIO_CROMATICO_MIN * 100:.0f}%): la escena es acromática —un blanco y negro, "
            "una escena nocturna o una alta clave desaturada— y cualquier esquema se "
            "estaría decidiendo sobre ruido"
        )
    return 1.0, (
        f"medido sobre el {ratio * 100:.1f}% del encuadre con color analizable, "
        f"{crom['n_matices']} matiz(ces) dominante(s) tras descartar los grupos sin masa"
    )


# =============================================================================
# Engine
# =============================================================================

class LuzTonoEngine:
    """Mide las dos métricas globales del Agente 4.

    Devuelve el modelo Pydantic ya validado (no un dict): la validación es parte del
    contrato, y así los tests atacan al engine y no a la tool. Mismo patrón que
    `ComposicionEspacialEngine` y `LineasDireccionEngine`.
    """

    def analizar(self, ruta_imagen):
        original = cargar_imagen(ruta_imagen)
        alto_orig, ancho_orig = original.shape[:2]
        trabajo = reescalar(original)

        expo = medir_exposicion(trabajo)
        crom = medir_cromatica(trabajo)
        conf_crom, fuente_crom = gate_cromatico(crom)
        abre = conf_crom >= 1.0

        # Con el gate cerrado se publica la etiqueta reservada y se vacía la evidencia
        # cromática, por el mismo motivo por el que el Agente 2 devuelve
        # `coord_punto_fuga = None`: un matiz calculado sobre ruido invitaría al
        # crítico a citarlo como si significara algo. Lo que NO se toca es
        # `ratio_pixeles_cromaticos`, que se publica siempre y en crudo, porque es la
        # condición del gate (tiene que ser auditable) y porque su complementario es
        # un hecho compositivo citable por derecho propio.
        # Nótese que los valores neutros son COHERENTES con sus propias definiciones y
        # no se contradicen al recomputarlos —lista vacía, luego ningún par que medir,
        # luego spread 0.0 y n 0—: es la lección de `valor_norm` del Agente 2.
        esquema = crom["esquema"] if abre else ESQUEMA_SIN_COLOR
        matices = crom["matices"] if abre else []
        pesos = crom["pesos"] if abre else []
        spread = crom["spread"] if abre else 0.0
        n_matices = crom["n_matices"] if abre else 0

        verificacion_path = self._dibujar_debug(
            ruta_imagen=ruta_imagen,
            trabajo=trabajo,
            expo=expo,
            crom=crom,
            matices=matices,
            pesos=pesos,
            spread=spread,
            n_matices=n_matices,
            esquema=esquema,
            conf_crom=conf_crom,
        )

        return InformeLuzTono(
            # Dimensiones de la imagen ORIGINAL, no las del espacio de trabajo: es lo
            # que la GUI necesita, y todo lo publicado son estadísticos o magnitudes
            # relativas, válidas en ambos espacios.
            dimensiones_imagen=(ancho_orig, alto_orig),
            media_L=MetricaConfianza[float](
                valor=expo["media_L"],
                valor_norm=adherencia_exposicion(expo["media_L"]),
                # 1.0 SIEMPRE, y no es un descuido: no hay gate porque no hay modo de
                # fallo. Toda imagen tiene píxeles y un canal L.
                confianza=1.0,
                fuente_confianza=(
                    "la exposición es medible en cualquier imagen: no depende del sujeto "
                    "ni de que la escena ofrezca ninguna estructura, así que esta métrica "
                    "no tiene condición de aplicabilidad que pueda fallar"
                ),
            ),
            std_L=expo["std_L"],
            pct_clipping_sombras=expo["pct_clipping_sombras"],
            pct_clipping_luces=expo["pct_clipping_luces"],
            esquema_cromatico=MetricaConfianza[str](
                valor=esquema,
                # None SIEMPRE: es una categoría, no una adherencia. Se descarta la
                # propuesta de §3.4.2 de puntuarla (1 identificable / 0.5 "otro")
                # porque mezcla adherencia con aplicabilidad, que es justo lo que
                # MetricaConfianza separa.
                valor_norm=None,
                confianza=conf_crom,
                fuente_confianza=fuente_crom,
            ),
            spread_cromatico=spread,
            matices_dominantes=matices,
            n_matices_dominantes=n_matices,
            ratio_pixeles_cromaticos=crom["ratio_cromatico"],
            verificacion_path=verificacion_path,
        )

    def _dibujar_debug(self, ruta_imagen, trabajo, expo, crom, matices, pesos,
                       spread, n_matices, esquema, conf_crom):
        """Verificación visual del Agente 4. Devuelve la ruta del PNG.

        Se dibuja sobre la imagen de TRABAJO, igual que en el Agente 2 y por el mismo
        motivo: lo medido está en ese espacio y el PNG es material de depuración.

        La franja de paleta es lo que hace revisable a ojo la métrica: si la foto se
        ve dominada por un color y la franja muestra dos bloques, el agrupamiento ha
        partido un matiz en rodajas. De hecho así se detectó la necesidad de
        `fusionar_matices` — por revisión visual, no por diseño.
        """
        capa = CapaVerificacion(ruta_imagen, SUBCARPETA, img=trabajo)
        abre = conf_crom >= 1.0

        # Franja de matices dominantes, con el ancho proporcional a su masa: así se ve
        # de un vistazo si la paleta la sostiene un color o varios.
        x = 0
        alto_franja = max(8, capa.alto // 20)
        for matiz, masa in zip(matices, pesos):
            ancho = int(round(masa * capa.ancho))
            if ancho <= 0:
                continue
            # Se pinta el matiz puro (S=V=max) para que el color sea legible aunque en
            # la foto aparezca apagado.
            bgr = cv2.cvtColor(
                np.uint8([[[int(matiz / 2), 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0][0]
            capa.rectangulo((x, 0), (x + ancho, alto_franja),
                            (int(bgr[0]), int(bgr[1]), int(bgr[2])), relleno=True)
            x += ancho

        capa.texto([
            (f"media_L = {expo['media_L']:.1f}  (adh {adherencia_exposicion(expo['media_L']):.2f})"
             f"  std_L = {expo['std_L']:.1f}", BLANCO),
            (f"clipping: sombras {expo['pct_clipping_sombras']:.2f}%"
             f"  luces {expo['pct_clipping_luces']:.2f}%", BLANCO),
            # El complementario (1 - ratio) se escribe al lado a propósito: es la parte
            # ACROMÁTICA del encuadre, y en fotos como un sujeto blanquinegro contra un
            # muro naranja es el dato compositivo que más pesa, aunque la métrica de
            # armonía sea ciega a él por construcción.
            (f"ratio_cromatico = {crom['ratio_cromatico']:.3f}"
             f"  (acromatico = {1 - crom['ratio_cromatico']:.3f})", BLANCO),
            (f"n_matices = {n_matices}  spread = {spread:.1f} deg  ->  {esquema}",
             color_confianza(conf_crom)),
            (f"gate cromatico = {'ABIERTO' if abre else 'cerrado'}",
             color_confianza(conf_crom)),
        ])
        return capa.guardar("luztono")


# =============================================================================
# Herramienta CrewAI
# =============================================================================

class LuzTonoInput(BaseModel):
    # UN SOLO CAMPO, igual que el Agente 2 y por la misma razón: este especialista es
    # puramente GLOBAL. No recibe bbox, ni centroide_saliencia, ni sujeto_discreto,
    # porque sus métricas son estadísticos de la distribución de píxeles.
    ruta_imagen: str = Field(
        ...,
        description="Ruta local de la imagen original analizada (.jpg, .png)",
    )


luz_tono_engine = LuzTonoEngine()


class LuzTonoTool(BaseTool):
    name: str = "Luz y tono"
    description: str = (
        "Analiza la luz y el color globales de la imagen: exposición y distribución "
        "tonal (media_L sobre CIELAB, std_L y los porcentajes de clipping) y armonía "
        "cromática (esquema_cromatico y los matices dominantes en grados del círculo "
        "de color). No depende del sujeto: mide estadísticos de toda la imagen. "
        "IMPORTANTE: sus dos métricas NO tienen el mismo estatuto. La exposición se "
        "emite siempre con confianza 1.0 porque no existe el caso 'no medible'. El "
        "esquema cromático SÍ tiene gate: en un blanco y negro, una escena nocturna o "
        "una alta clave desaturada la herramienta devuelve confianza 0, y entonces la "
        "etiqueta NO es citable por convincente que suene. No recalcules esa confianza "
        "ni la contradigas. "
        "Ojo también a la semántica: 'monocromático' significa un solo matiz dominante "
        "ENTRE LOS PÍXELES CROMÁTICOS, no que la imagen tenga un solo color; la parte "
        "acromática del encuadre se lee como 1 - ratio_pixeles_cromaticos. "
        "Guarda además una imagen de verificación (franja con la paleta extraída y las "
        "métricas) cuya ruta devuelve en verificacion_path."
    )
    args_schema: Type[BaseModel] = LuzTonoInput

    def _run(self, ruta_imagen):
        # El engine devuelve el modelo Pydantic (ya validado); la tool lo serializa a
        # dict para que el LLM reciba JSON limpio y no el repr del modelo. Mismo
        # reparto que en los Agentes 1 y 2: los tests trabajan contra el engine.
        informe = luz_tono_engine.analizar(ruta_imagen)
        return informe.model_dump()
