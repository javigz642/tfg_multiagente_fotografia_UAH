from pydantic import BaseModel, Field
from typing import Type
from crewai.tools import BaseTool
from tfg_multiagente_fotografia.schemas.especialistas import MetricaConfianza, InformeComposicionEspacial
from tfg_multiagente_fotografia.tools.capa_verificacion import (
    CapaVerificacion,
    color_confianza,
    BLANCO,
    CIAN,
    GRIS,
    MAGENTA,
)
import cv2
from math import sqrt

SUBCARPETA = "composicion_espacial"

# Cotas superiores EXACTAS de cada distancia normalizada por la diagonal
COTA_D_TERCIOS = 1 / 3
COTA_D_CENTRO = 1 / 2

FUENTE_CON_IDENTIDAD = "yolo"


class ComposicionEspacialEngine():
    def analizar(self, ruta_imagen, bbox, centroide_saliencia, sujeto_discreto,
                 fuente, mapa_saliencia_path):
        #empezamos por d_equilibrio
        sal_map = cv2.imread(mapa_saliencia_path, cv2.IMREAD_GRAYSCALE)
        if sal_map is None:
            raise ValueError(f"No se pudo leer el mapa de saliencia: {mapa_saliencia_path}")
        # La capa lee la imagen original y expone sus dimensiones; el engine solo
        # se ocupa de la geometría.
        capa = CapaVerificacion(ruta_imagen, SUBCARPETA)
        #mapa de saliencia conserva las dimensiones de la original
        h,w = sal_map.shape[:2]
        # Guarda explícita del acoplamiento de espacios de coordenadas: bbox y
        # centroide_saliencia vienen del espacio de la imagen ORIGINAL, mientras
        # que (w, h) se derivan del MAPA. Si algún día el mapa dejara de conservar
        # el tamaño de entrada, mezclaríamos dos espacios en silencio.
        capa.verificar_dimensiones(sal_map.shape, "El mapa de saliencia")
        M = cv2.moments(sal_map)
        # Sin energía no hay centro de masa: preferimos fallar a inventar un equilibrio perfecto.
        if M["m00"] == 0:
            raise ValueError(f"Mapa de saliencia sin energía (todo negro): {mapa_saliencia_path}")
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        diagonal = sqrt(w**2 + h**2)
        d_equilibrio_valor = distancia_euclidea((cx,cy), (w/2, h/2)) / diagonal
        coord_centro_masa = (cx/w ,cy/h) #normalizado [0,1]
        confianza_equilibrio_valor = 1.0
        fuente_confianza_equilibrio = "métrica global sobre el mapa de saliencia"


        #------------------------regla de los tercios----------------

        # LA VÍA DE ANCLAJE Y EL GATE SON COSAS DISTINTAS, igual que en el Agente 3.
        # La VÍA la elige `sujeto_discreto`: con un bbox localizable se ancla en su
        # centro, y si no, en el centroide de saliencia como mejor referencia disponible.
        # El GATE lo decide `fuente`. Son ortogonales a propósito, y esa ortogonalidad es
        # el contrato del Nivel 2: se mide SIEMPRE (ningún especialista silencia una
        # métrica) y la `confianza` dice si lo medido es citable. De ahí que una imagen
        # pueda anclarse en un bbox perfectamente definido y salir con confianza 0.
        if sujeto_discreto:
            p_referencia = bbox[:2]
        else:
            p_referencia = centroide_saliencia[:2]

        confianza_tercios_valor = 1.0 if fuente == FUENTE_CON_IDENTIDAD else 0.0

        if fuente == FUENTE_CON_IDENTIDAD:
            fuente_confianza_tercios = (
                "anclado al bbox de una deteccion de YOLO, que aporta identidad "
                "semantica: hay un objeto reconocido sobre el que anclar"
            )
        elif sujeto_discreto:
            fuente_confianza_tercios = (
                f"el anclaje sale del blob de saliencia y no de una deteccion de YOLO "
                f"(fuente={fuente}): hay una region destacada y acotada, pero nada "
                "garantiza que sea un sujeto y no una textura, una masa de fondo o "
                "simplemente la zona mas brillante del encuadre"
            )
        else:
            fuente_confianza_tercios = (
                f"la escena no ofrece ningun objeto focal acotado (fuente={fuente}): el "
                "anclaje es el centroide de saliencia, una referencia de posicion y no "
                "un sujeto"
            )

        puntos_tercios = {(w/3, h/3):0,(2*w/3, h/3):0,(w/3, 2*h/3):0,(2*w/3, 2*h/3):0}

        for p in puntos_tercios.keys():
            puntos_tercios[p] = distancia_euclidea(p_referencia, p)
        punto_cercano_tercios = min(puntos_tercios, key = puntos_tercios.get)
        distancia_minima = puntos_tercios[punto_cercano_tercios]
        d_tercios_valor = distancia_minima/diagonal

        # Centrado: hipótesis compositiva RIVAL de los tercios, medida sobre el
        # MISMO anclaje y con el MISMO denominador para que ambas sean comparables
        # como distancias reales. Una foto deliberadamente centrada da
        # d_tercios = 1/6 sin que eso sea un fallo; d_centro lo desambigua.
        d_centro_valor = distancia_euclidea(p_referencia, (w/2, h/2)) / diagonal

        # Arbitraje entre las dos hipótesis rivales. Se resuelve AQUÍ, en el script,
        # y no en el prompt del crítico, porque es una comparación determinista y es
        # justo donde un LLM podría equivocarse usando los `valor_norm` (que se
        # escalan por cotas distintas y NO son comparables) en vez de los `valor`.
        # El margen dice cuánto gana el ganador: cerca de 0 es un desempate técnico.
        patron_dominante_valor = "centrada" if d_centro_valor < d_tercios_valor else "tercios"
        margen_patron = abs(d_tercios_valor - d_centro_valor)

        # Verificación visual: se dibuja en píxeles (antes de normalizar) porque
        # es el espacio en el que ya están todos los puntos calculados arriba.
        verificacion_path = self._dibujar_debug(
            capa=capa,
            p_anclaje=p_referencia,
            p_cercano=punto_cercano_tercios,
            p_centro_masa=(cx, cy),
            d_tercios_valor=d_tercios_valor,
            d_centro_valor=d_centro_valor,
            d_equilibrio_valor=d_equilibrio_valor,
            confianza_tercios=confianza_tercios_valor,
            patron_dominante_valor=patron_dominante_valor,
            margen_patron=margen_patron,
            fuente=fuente,
        )

        return InformeComposicionEspacial(
            dimensiones_imagen=(w,h),
            d_tercios=MetricaConfianza[float](valor=d_tercios_valor, valor_norm=adherencia(d_tercios_valor, COTA_D_TERCIOS), confianza=confianza_tercios_valor, fuente_confianza=fuente_confianza_tercios),
            coord_centroide=(p_referencia[0]/w, p_referencia[1]/h),
            coord_p_cercano=(punto_cercano_tercios[0]/w, punto_cercano_tercios[1]/h),
            d_centro=MetricaConfianza[float](valor=d_centro_valor, valor_norm=adherencia(d_centro_valor, COTA_D_CENTRO), confianza=confianza_tercios_valor, fuente_confianza=fuente_confianza_tercios),
            # valor_norm=None: es una categoría, no una adherencia. Comparte la
            # confianza del anclaje porque sin un sujeto con identidad el veredicto
            # arbitra entre dos distancias que tampoco son fiables.
            patron_dominante=MetricaConfianza[str](valor=patron_dominante_valor, confianza=confianza_tercios_valor, fuente_confianza=fuente_confianza_tercios),
            margen_patron=margen_patron,
            # valor_norm=None deliberado: la cota teórica (1/2) no refleja el rango
            # real (~[0, 0.16]) y normalizar por ella diría que todo está equilibrado.
            # Pendiente de calibración empírica por percentiles (validación §5).
            d_equilibrio=MetricaConfianza[float](valor=d_equilibrio_valor, confianza=confianza_equilibrio_valor, fuente_confianza=fuente_confianza_equilibrio),
            coord_centro_masa=coord_centro_masa,
            verificacion_path=verificacion_path,
        )

    def _dibujar_debug(self, capa, p_anclaje, p_cercano, p_centro_masa,
                       d_tercios_valor, d_centro_valor, d_equilibrio_valor, confianza_tercios,
                       patron_dominante_valor, margen_patron, fuente):
        """Dibuja la verificación visual del Agente 1 sobre la capa recibida.

        Separa la representación del cálculo: recibe los puntos ya resueltos en
        PÍXELES y solo dibuja. La capa se ocupa de que todo sea legible a cualquier
        resolución; aquí solo vive la SEMÁNTICA (qué significa cada marca). Se pinta
        de fondo a primer plano (rejilla → puntos de fuerza → líneas de distancia →
        puntos protagonistas → texto) para que nada tape lo importante. Devuelve la
        ruta del PNG guardado.
        """
        w, h = capa.dimensiones
        centro = (w / 2, h / 2)

        # 1. Rejilla de tercios (gris tenue, al fondo)
        for x in (w / 3, 2 * w / 3):
            capa.linea((x, 0), (x, h), GRIS)
        for y in (h / 3, 2 * h / 3):
            capa.linea((0, y), (w, y), GRIS)

        # 2. Los 4 puntos de fuerza
        for px in (w / 3, 2 * w / 3):
            for py in (h / 3, 2 * h / 3):
                capa.punto((px, py), GRIS, relleno=False)

        # El color del anclaje (y de su línea) codifica la confianza binaria.
        color_tercios = color_confianza(confianza_tercios)

        # 3. Línea de d_tercios: anclaje -> punto de fuerza ganador
        capa.linea(p_anclaje, p_cercano, color_tercios)

        # 3.b Línea de d_centro: anclaje -> centro geométrico (hipótesis rival:
        # se ve de un vistazo si la foto se acoge al centro o a la rejilla según
        # cuál de las dos líneas del anclaje es más corta).
        capa.linea(p_anclaje, centro, CIAN)

        # 5. Línea de d_equilibrio: centro de masa -> centro geométrico
        capa.linea(p_centro_masa, centro, MAGENTA)

        # 4. Puntos, en primer plano (círculos rellenos). El ANCLAJE va el último
        # a propósito: es el protagonista (lleva el color de la confianza) y en
        # composiciones centradas coincide con el centro geométrico y con el
        # centro de masa, que si no lo taparían.
        capa.punto(p_cercano, color_tercios, factor=1.5)
        capa.punto(centro, BLANCO, factor=1.5)
        capa.punto(p_centro_masa, MAGENTA, factor=2)
        capa.punto(p_anclaje, color_tercios, factor=2)

        # 6. Información textual (esquina superior izquierda)
        # Se muestran valor y adherencia juntos: el valor es la distancia auditable
        # y la adherencia la lectura normalizada que consumirá el crítico.
        capa.texto([
            # De dónde salió el anclaje: es lo que DECIDE la confianza de las tres
            # métricas ancladas, así que sin este dato la imagen no explica por qué el
            # anclaje está en rojo. Mismo papel que `fuente_mascara` en el Agente 3.
            (f"fuente_anclaje = {fuente}", BLANCO),
            (f"d_tercios = {d_tercios_valor:.4f}  adh={adherencia(d_tercios_valor, COTA_D_TERCIOS):.2f}"
             f"  (conf={confianza_tercios:.1f})", color_tercios),
            (f"d_centro = {d_centro_valor:.4f}  adh={adherencia(d_centro_valor, COTA_D_CENTRO):.2f}", CIAN),
            (f"d_equilibrio = {d_equilibrio_valor:.4f}", MAGENTA),
            # El veredicto se escribe también en la imagen para que la verificación
            # visual sea autoexplicativa: se lee el patrón ganador y de un vistazo se
            # comprueba contra cuál de las dos líneas del anclaje es más corta.
            (f"patron = {patron_dominante_valor}  (margen={margen_patron:.4f})",
             color_tercios if patron_dominante_valor == "tercios" else CIAN),
        ])

        return capa.guardar("composicion")


class ComposicionEspacialInput(BaseModel):
    ruta_imagen: str = Field(..., description="Ruta local de la imagen original analizada (.jpg, .png)")
    bbox: list[float] = Field(..., description="[cx, cy, w, h] del sujeto principal, o bbox de referencia si fuente='sin_sujeto_claro'")
    centroide_saliencia: list[int] = Field(..., description="[x, y] del centroide del blob de saliencia principal (o del píxel de máxima saliencia si no hay contornos separables)")
    sujeto_discreto: bool = Field(
        ...,
        description=(
            "Flag de la percepción compartida. Elige la VÍA de anclaje: True ancla en el "
            "centro del bbox, False en el centroide de saliencia. No confundir con el gate."
        ),
    )
    fuente: str = Field(
        ...,
        description=(
            "Cómo obtuvo la percepción compartida el bbox: 'yolo', 'saliencia' o "
            "'sin_sujeto_claro'. Es lo que DECIDE la confianza de las tres métricas "
            "ancladas al sujeto (d_tercios, d_centro y patron_dominante)."
        ),
    )
    mapa_saliencia_path: str = Field(..., description="Ruta al mapa de saliencia por residuo espectral")

composicion_espacial_engine = ComposicionEspacialEngine()
class ComposicionEspacialTool(BaseTool):
    name: str = "Composición espacial"
    description:str = (
        "Analiza la composición espacial de la imagen: regla de los tercios y equilibrio visual. " \
        "Ancla la regla de los tercios al sujeto si sujeto_discreto=True, o a la saliencia si no lo hay. " \
        "Resuelve además, de forma determinista, a qué patrón se acoge la foto (patron_dominante: " \
        "'tercios' o 'centrada') y con cuánta ventaja (margen_patron): esa comparación ya está hecha, " \
        "no la repitas ni la contradigas. " \
        "IMPORTANTE sobre la confianza: las tres métricas ancladas al sujeto (d_tercios, "
        "d_centro y patron_dominante) tienen gate y solo abren si el sujeto lo detectó "
        "YOLO; si el bbox salió del mapa de saliencia la herramienta devuelve confianza 0 "
        "y entonces sus valores NO son citables como hechos, por razonables que parezcan. "
        "d_equilibrio es global y no tiene gate: su confianza es 1.0 siempre, así que en "
        "esas escenas sigue habiendo algo que afirmar. No recalcules esas confianzas ni "
        "las contradigas: el motivo exacto del cierre viaja en fuente_confianza. "
        "Guarda además una imagen de verificación (rejilla de tercios, puntos de fuerza, anclaje "
        "coloreado por confianza y las dos líneas de distancia) cuya ruta devuelve en verificacion_path."
    )
    args_schema: Type[BaseModel] = ComposicionEspacialInput
    def _run(self, ruta_imagen, bbox, centroide_saliencia, sujeto_discreto, fuente, mapa_saliencia_path):
        # El engine devuelve el modelo Pydantic (ya validado); la tool lo serializa a
        # dict para que el LLM reciba JSON limpio y no el repr del modelo, igual que
        # hace SharedPerceptionTool. Los tests trabajan contra el engine.
        informe = composicion_espacial_engine.analizar(
            ruta_imagen, bbox, centroide_saliencia, sujeto_discreto, fuente, mapa_saliencia_path
        )
        return informe.model_dump()



def distancia_euclidea(p1, p2):
    return sqrt((p2[0] - p1[0])**2+(p2[1] - p1[1])**2)


def adherencia(valor, cota):
    """Convierte una distancia acotada en la lectura normalizada `valor_norm` de [0,1].

    Convención ÚNICA para todo el sistema (ver MetricaConfianza.valor_norm):
    1 = adherencia máxima al patrón que la métrica nombra (distancia 0),
    0 = adherencia nula (distancia en su cota).

    Es adherencia, no calidad: dice cuánto se ajusta la foto a una convención
    compositiva concreta, no si eso es un acierto aquí (eso lo decide el crítico
    con W). El `clamp` solo absorbe el error de coma flotante en los extremos:
    las cotas son exactas, no empíricas.
    """
    return min(1.0, max(0.0, 1.0 - valor / cota))
