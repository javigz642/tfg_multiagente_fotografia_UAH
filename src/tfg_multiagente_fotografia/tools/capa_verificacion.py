"""Primitivas de dibujo compartidas para las verificaciones visuales."""
from pathlib import Path

import cv2

# Reexportadas por compatibilidad; la definición vive en `rutas.py`.
from tfg_multiagente_fotografia.rutas import OUTPUTS_PATH, PROYECT_PATH

# --- Paleta (BGR) ---
GRIS = (150, 150, 150)
VERDE = (0, 255, 0)
ROJO = (0, 0, 255)
NARANJA = (0, 165, 255)
MAGENTA = (255, 0, 255)
CIAN = (255, 255, 0)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# `COLOR_POR_FUENTE` indica el origen del bbox; `color_confianza` indica fiabilidad.
COLOR_POR_FUENTE = {
    "yolo": VERDE,
    "saliencia": NARANJA,
    "sin_sujeto_claro": ROJO,
}


def color_confianza(confianza):
    """Devuelve verde para confianza total y rojo en otro caso."""
    return VERDE if confianza >= 1.0 else ROJO


# Límite de tamaño de las imágenes de verificación mostradas por la interfaz.
LADO_MAYOR_VERIFICACION = 1600

# Compresión PNG aplicada al guardar las verificaciones.
COMPRESION_PNG = 6


def escribir_verificacion(destino, img):
    """Escribe una verificación reduciendo solo su lado mayor."""
    alto, ancho = img.shape[:2]
    mayor = max(alto, ancho)
    if mayor > LADO_MAYOR_VERIFICACION:
        factor = LADO_MAYOR_VERIFICACION / mayor
        img = cv2.resize(
            img, (round(ancho * factor), round(alto * factor)), interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(str(destino), img, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESION_PNG])
    return str(destino)


class CapaVerificacion:
    """Copia de la imagen original sobre la que dibujar las marcas de verificación.

    Se construye una por análisis y se guarda al final. Nunca toca la imagen
    original en disco ni el array que reciba.
    """

    def __init__(self, ruta_imagen, subcarpeta, img=None):
        """`img` opcional para no releer del disco si el llamante ya la tiene cargada
        (una imagen de 3279x4096 son ~40 MB por lectura)."""
        self.ruta_imagen = Path(ruta_imagen)
        if img is None:
            img = cv2.imread(str(self.ruta_imagen))
            if img is None:
                raise ValueError(f"No se pudo leer la imagen original: {ruta_imagen}")
        self.img = img.copy()  # nunca modificamos lo que nos dan

        self.alto, self.ancho = self.img.shape[:2]
        menor = min(self.ancho, self.alto)
        self.grosor = max(2, int(menor * 0.004))
        self.radio = max(4, int(menor * 0.008))
        self.escala = max(0.6, menor * 0.0015)

        self.carpeta = OUTPUTS_PATH / subcarpeta
        self.carpeta.mkdir(parents=True, exist_ok=True)

    @property
    def dimensiones(self):
        """(ancho, alto) en píxeles, en el orden en que lo reporta el informe."""
        return (self.ancho, self.alto)

    def verificar_dimensiones(self, shape, nombre):
        """Falla si un derivado (mapa de saliencia, máscara...) no conserva el tamaño.

        Los agentes mezclan coordenadas que vienen de la imagen original (bbox,
        centroides) con dimensiones derivadas de esos mapas. Si dejaran de coincidir,
        el error sería silencioso y sistemático, así que se corta aquí.
        """
        alto, ancho = shape[:2]
        if (alto, ancho) != (self.alto, self.ancho):
            raise ValueError(
                f"{nombre} no conserva las dimensiones de la imagen original "
                f"({nombre}={ancho}x{alto}, original={self.ancho}x{self.alto}): las "
                "coordenadas de ambos espacios no son comparables."
            )

    # --- Primitivas ---
    def linea(self, p1, p2, color):
        cv2.line(self.img, _px(p1), _px(p2), color, self.grosor)

    def rectangulo(self, p1, p2, color, relleno=False):
        """`relleno=True` pinta el interior en vez del contorno.

        Lo necesita el Agente 4 para la franja de matices dominantes, donde el
        color ES el dato y no una marca sobre otra cosa; el resto de agentes usan
        el contorno, que no tapa la imagen que hay debajo.
        """
        grosor = cv2.FILLED if relleno else self.grosor
        cv2.rectangle(self.img, _px(p1), _px(p2), color, grosor)

    def punto(self, p, color, factor=1.0, relleno=True):
        """`factor` escala el radio base para jerarquizar (protagonista vs referencia)."""
        cv2.circle(
            self.img,
            _px(p),
            max(1, int(self.radio * factor)),
            color,
            -1 if relleno else self.grosor,
        )

    def etiqueta(self, p, texto, color):
        """Texto suelto anclado a un punto (p. ej. sobre un bbox)."""
        self._texto_con_contorno(texto, _px(p), color)

    def texto(self, lineas):
        """Bloque de líneas `(texto, color)` en la esquina superior izquierda."""
        margen = int(self.radio * 2)
        alto_linea = int(40 * self.escala)
        for i, (contenido, color) in enumerate(lineas):
            self._texto_con_contorno(contenido, (margen, margen + alto_linea * (i + 1)), color)

    def _texto_con_contorno(self, contenido, origen, color):
        """Contorno negro debajo del texto: legible también sobre fondos claros."""
        for tono, grosor in ((NEGRO, self.grosor * 3), (color, self.grosor)):
            cv2.putText(
                self.img, contenido, origen, cv2.FONT_HERSHEY_SIMPLEX, self.escala, tono, grosor
            )

    def guardar(self, sufijo):
        """Guarda como `{nombre_imagen}_{sufijo}.png` y devuelve la ruta como str.

        El lado mayor se acota al escribir (ver `escribir_verificacion`). No se toca
        `self.img`: `dimensiones` sigue reportando el tamaño real de la imagen
        analizada, que es lo que los informes publican en `dimensiones_imagen`.
        """
        salida = self.carpeta / f"{self.ruta_imagen.stem}_{sufijo}.png"
        return escribir_verificacion(salida, self.img)


def _px(p):
    """OpenCV exige enteros; los cálculos del sistema van en float."""
    return (int(p[0]), int(p[1]))
