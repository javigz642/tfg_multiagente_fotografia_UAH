"""Utilidades de carga de imagen compartidas por los especialistas."""
import cv2

# Lado mayor, en píxeles, del espacio de trabajo común a los Agentes 2, 3 y 4.
#

# El calculó de esos agentes corre sobre la imagen reducida a ese tamaño, con
#  reescalado UNIFORME (mismo factor en x y en y). Tres motivos:
#   - Los umbrales en píxeles significan lo mismo en todas las fotos. `minLineLength=50`
#     es el 6% del encuadre en una imagen de 800 px y el 1.2% en una de 4096, donde
#     entraría la juntura de un azulejo como "línea".
#   - La varianza del Laplaciano (Agente 3) DEPENDE de la resolución: a más píxeles, más
#     bordes finos que el operador recoge. Sin un espacio fijo, su umbral de textura
#     significaría cosas distintas en cada foto.
#   - Coste: 16 veces menos píxeles que procesar en las imágenes grandes.
#
# Que sea uniforme es innegociable: deformar la relación de aspecto cambiaría los
# ángulos, y el ángulo ES una de las dos métricas del Agente 2. Al escalar
# uniformemente los ángulos se conservan exactos y solo cambian las longitudes, por un
# factor conocido que se puede deshacer.
LADO_MAYOR_TRABAJO = 1024


def cargar_imagen(ruta):
    """Lee la imagen en BGR (el orden de canales de OpenCV y CapaVerificacion).

    Se usa `cv2.imread` y no PIL a propósito: PIL lleva una guarda anti
    decompression-bomb que lanza con las imágenes de mucha resolución, y aquí no hace
    falta porque el reescalado al espacio de trabajo ocurre inmediatamente después.
    """
    img = cv2.imread(str(ruta))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    return img
