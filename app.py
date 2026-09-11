"""Interfaz Streamlit"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from tfg_multiagente_fotografia.main import construir_inputs
from tfg_multiagente_fotografia.rutas import OUTPUTS_PATH, dir_ejecucion


SUBIDAS_PATH = OUTPUTS_PATH / "subidas"

# Paneles mostrados en la interfaz: título, explicación, JSON y claves anidadas.
PANELES_VERIFICACION = [
    (
        "Sujeto detectado",
        "El rectángulo marca el sujeto principal, y su color dice de dónde sale: verde si el "
        "detector de objetos lo ha identificado y sabe qué es, naranja si solo es la región "
        "que más destaca —sin identidad que la confirme— y rojo si la escena no ofrece "
        "ningún sujeto claro. De esa diferencia depende que varias mediciones se puedan "
        "afirmar o no.",
        "salida_orquestador.json",
        ("percepcion_compartida", "percepcion_bbox_path"),
    ),
    (
        "Mapa de saliencia",
        "Dónde se concentra el peso visual de la escena: cuanto más claro, más atrae la "
        "mirada esa zona. De este mapa sale el centro de masas con el que se mide el "
        "equilibrio de la composición.",
        "salida_orquestador.json",
        ("percepcion_compartida", "mapa_saliencia_path"),
    ),
    (
        "Composición espacial",
        "La rejilla de tercios y sus cuatro puntos de fuerza. Las líneas miden la distancia "
        "del sujeto al punto de fuerza más cercano y la del peso visual al centro del "
        "encuadre.",
        "salida_composicion_espacial.json",
        ("informe", "verificacion_path"),
    ),
    (
        "Líneas y dirección",
        "En cian, las líneas de la escena que concurren en un mismo punto de fuga. En verde "
        "o rojo, el segmento tomado como horizonte, sobre una horizontal gris de referencia "
        "que hace visible su desviación.",
        "salida_lineas_direccion.json",
        ("informe", "verificacion_path"),
    ),
    (
        "Espacio y aislamiento",
        "El contorno cian es la silueta del sujeto recortada del fondo. Con ella se mide "
        "cuánto encuadre queda libre a su alrededor y si el sujeto concentra más detalle "
        "que lo que le rodea.",
        "salida_espacio_aislamiento.json",
        ("informe", "verificacion_path"),
    ),
    (
        "Luz y tono",
        "La franja superior es la paleta de colores dominantes de la fotografía, y el ancho "
        "de cada bloque es proporcional a la parte del encuadre que ocupa ese color.",
        "salida_luz_tono.json",
        ("informe", "verificacion_path"),
    ),
]

# Leyenda estática de las escalas de lectura de las métricas.
LEYENDA_METRICAS = [
    ("Composición espacial", [
        ("d_tercios", "0 a 0.33",
         "Distancia del sujeto al punto de fuerza más cercano de la rejilla, en fracciones "
         "de la diagonal. 0 = justo encima de él , 0.17 = en el centro exacto del encuadre "
         ", 0.33 = en una esquina."),
        ("d_centro", "0 a 0.5",
         "La misma distancia, pero medida al centro del encuadre. Se lee enfrentada a "
         "`d_tercios`: la foto se acoge a la referencia cuya distancia sea menor."),
        ("d_equilibrio", "referencia orientativa",
         "Cuánto se desplaza el peso visual del conjunto respecto al centro. No tiene "
         "escala cerrada: como orientación, por debajo de 0.03 el peso está repartido y "
         "por encima de 0.12 está claramente cargado hacia un lado."),
    ]),
    ("Líneas y dirección", [
        ("angulo_horizonte", "-15° a +15°",
         "Desviación del horizonte respecto a la horizontal, con signo: positivo = el lado "
         "derecho cae por debajo del izquierdo. 0 = perfectamente nivelado."),
        ("score_convergencia", "0 a 1",
         "Parte de las líneas de la escena que fugan hacia un mismo punto. 1 = todas "
         "convergen. Es una medida de estructura en profundidad, no de si las líneas "
         "señalan al sujeto."),
    ]),
    ("Espacio y aislamiento", [
        ("ratio_espacio_negativo", "0 a 1",
         "Es directamente la parte del encuadre libre de sujeto: 0.96 significa que el "
         "sujeto ocupa el 4% restante."),
        ("ratio_nitidez", "-1 a +1",
         "Cuánto más detalle concentra el sujeto que el fondo. 0 = el mismo en ambos , "
         "+0.50 = el triple , +0.80 = nueve veces. En negativo se lee igual con los papeles "
         "cambiados: el fondo es el que concentra el detalle."),
    ]),
    ("Luz y tono", [
        ("media_L", "0 a 100",
         "Luminosidad media. 0 = negro absoluto, 100 = blanco absoluto y 50 equivale al "
         "gris medio que se usa como referencia de exposición. Por debajo de 35 la imagen "
         "es de clave baja; por encima de 65, de clave alta."),
        ("std_L", "0 a 50",
         "Contraste global. El techo de 50 solo lo alcanza una imagen partida en negro puro "
         "y blanco puro sin medios tonos, así que una escena real queda bastante por debajo."),
        ("spread_cromatico", "0° a 180°",
         "Cuánto se separan entre sí los colores dominantes en el círculo cromático. "
         "0 = un único matiz , por debajo de 60 = matices vecinos , en torno a 180 = colores "
         "opuestos."),
        ("pct_clipping_sombras y pct_clipping_luces", "0% a 100%",
         "Parte del encuadre que ha perdido todo el detalle por haberse ido a negro puro o "
         "a blanco puro. Unas décimas son normales en cualquier escena."),
    ]),
]


# Carga y persistencia de la imagen.

def guardar_imagen_subida(archivo_subido) -> Path:
    """Guarda la subida en `outputs/subidas/` y devuelve su ruta."""
    # Conserva solo el nombre del fichero subido.
    nombre = Path(archivo_subido.name).name

    SUBIDAS_PATH.mkdir(parents=True, exist_ok=True)

    ruta = SUBIDAS_PATH / nombre
    ruta.write_bytes(archivo_subido.getbuffer())
    return ruta


# Ejecución del crew con indicador de progreso.

def analizar(ruta_imagen: Path) -> bool:
    """Ejecuta el crew tras un indicador de carga genérico y devuelve si terminó bien.

    El progreso paso a paso (qué tarea corre en cada momento) no se muestra aquí: desde
    que las cuatro tareas del Nivel 2 corren en paralelo (`async_execution=True` en
    `crew.py`) ya no hay un orden fijo que anunciar, y reconstruirlo exigía tocar el
    contexto interno de hilos de Streamlit, una complejidad innecesaria para una
    interfaz deliberadamente sencilla. El
    detalle real de la ejecución (qué agente corre, qué herramienta invoca, con qué
    resultado) se sigue viendo en la terminal, con `verbose=True` en los agentes.
    """
    from tfg_multiagente_fotografia.crew import TfgMultiagenteFotografia

    with st.spinner("Analizando la foto… (puede tardar varios minutos)"):
        try:
            crew = TfgMultiagenteFotografia().crew()
            crew.kickoff(inputs=construir_inputs(str(ruta_imagen)))
        except Exception as exc:
            st.error(f"El análisis no ha podido completarse: {exc}")
            return False

    st.success("Análisis completado.")
    return True


# Lectura de resultados persistidos.

def leer_json(ruta_imagen: Path, nombre_fichero: str) -> dict:
    """Lee un JSON de la ejecución y devuelve `{}` si falta o está corrupto."""
    ruta = dir_ejecucion(ruta_imagen) / nombre_fichero
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def mostrar_critica(critica: dict) -> None:
    """Muestra la síntesis, las salvedades y la leyenda de métricas."""
    sintesis = critica.get("sintesis", "")
    if not sintesis:
        st.warning(
            "El análisis no ha dejado ninguna crítica escrita. Si las imágenes de "
            "verificación de abajo sí aparecen, el fallo ocurrió en el último paso."
        )
        return

    st.subheader("Crítica compositiva")
    st.markdown(sintesis)

    salvedades = critica.get("salvedades") or []
    if salvedades:
        with st.expander(f"Lo que el sistema no puede afirmar sobre esta imagen ({len(salvedades)})"):
            for salvedad in salvedades:
                st.markdown(f"- {salvedad}")

    mostrar_leyenda_metricas()


def mostrar_leyenda_metricas() -> None:
    """Muestra plegada la escala de lectura de las métricas citadas."""
    with st.expander("Cómo se leen las cifras de la crítica"):
        st.markdown(
            "Las cifras entre corchetes son **mediciones sobre esta fotografía**, no notas ni "
            "puntuaciones: cada una vive en su propia escala y ninguna se considera buena o mala." \
            "Cuando una medición no es fiable en una imagen concreta, el sistema no "
            "la cita sino que la traslada al apartado de lo que no puede afirmar."
        )
        for dimension, metricas in LEYENDA_METRICAS:
            st.markdown(f"**{dimension}**")
            filas = "\n".join(
                f"| `{campo}` | {escala} | {lectura} |" for campo, escala, lectura in metricas
            )
            st.markdown(f"| Medición | Escala | Cómo se lee |\n|---|---|---|\n{filas}")


# Imágenes de verificación.

def mostrar_verificaciones(ruta_imagen: Path) -> None:
    """Muestra las imágenes de verificación disponibles en una rejilla."""
    paneles = []
    for titulo, explicacion, fichero, claves in PANELES_VERIFICACION:
        valor = leer_json(ruta_imagen, fichero)

        for clave in claves:
            if not isinstance(valor, dict):
                valor = None
                break
            valor = valor.get(clave)

        if valor and Path(valor).exists():
            paneles.append((titulo, explicacion, valor))

    if not paneles:
        st.info("Este análisis no ha dejado imágenes de verificación.")
        return

    st.subheader("En qué se apoya esta lectura")
    st.caption("Cada panel es lo que midió un agente, dibujado sobre la fotografía.")

    for inicio in range(0, len(paneles), 2):
        columnas = st.columns(2)
        for columna, (titulo, explicacion, ruta_png) in zip(columnas, paneles[inicio:inicio + 2]):
            with columna:
                st.image(ruta_png, width="stretch")
                st.markdown(f"**{titulo}**")
                st.caption(explicacion)


# Estado y disposición de la interfaz.

def vista_esperando() -> None:
    """Muestra el cargador y permite iniciar un análisis."""
    archivo = st.file_uploader(
        "Elige una fotografía", type=["jpg", "jpeg", "png"]
    )
    if archivo is None:
        return

    st.image(archivo, caption=archivo.name, width=400)

    # El crew solo se lanza al pulsar el botón.
    if st.button("Analizar", type="primary"):
        ruta = guardar_imagen_subida(archivo)
        if analizar(ruta):
            st.session_state.ruta_analizada = str(ruta)
            st.rerun()


def vista_listo(ruta_imagen: Path) -> None:
    """Muestra los resultados guardados para la imagen."""
    if st.button("Analizar otra fotografía"):
        st.session_state.ruta_analizada = None
        st.rerun()

    columna_foto, columna_critica = st.columns([1, 2])
    with columna_foto:
        st.image(str(ruta_imagen), caption=ruta_imagen.name, width="stretch")
    with columna_critica:
        mostrar_critica(leer_json(ruta_imagen, "salida_critica.json"))

    mostrar_verificaciones(ruta_imagen)


def main() -> None:
    """Punto de entrada de la interfaz Streamlit."""
    # Debe ser la primera llamada a Streamlit.
    st.set_page_config(
        page_title="Análisis de composición fotográfica",
        page_icon="📷",
        layout="wide",
    )

    st.title("Análisis de composición fotográfica")
    st.caption(
        "Sistema multiagente que mide la composición de una fotografía y redacta una "
        "crítica en la que cada afirmación se apoya en una métrica verificable."
    )

    # None representa ESPERANDO; una ruta representa LISTO.
    st.session_state.setdefault("ruta_analizada", None)

    if st.session_state.ruta_analizada is None:
        vista_esperando()
    else:
        vista_listo(Path(st.session_state.ruta_analizada))


if __name__ == "__main__":
    main()
