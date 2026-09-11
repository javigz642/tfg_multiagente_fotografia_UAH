from ultralytics import YOLO
from pydantic import BaseModel, Field
from typing import Type
from crewai.tools import BaseTool
import cv2, numpy as np
from pathlib import Path

from tfg_multiagente_fotografia.rutas import OUTPUTS_PATH, PROYECT_PATH
from tfg_multiagente_fotografia.tools.capa_verificacion import (
    CapaVerificacion,
    COLOR_POR_FUENTE,
    escribir_verificacion,
)

SALIENCY_PATH = OUTPUTS_PATH / "saliency_map"
YOLO_PATH = OUTPUTS_PATH / "yolo"
YOLO_WEIGHTS_DIR = PROYECT_PATH / "models" / "yolo"

CONF_YOLO_THRESHOLD = 0.75

# Techo y suelo de plausibilidad para el contorno ganador de saliencia.
# Otsu SIEMPRE devuelve un ganador, incluso cuando no existe un sujeto discreto
# (paisaje, arquitectura, niebla, texturas repetitivas). Estos límites son la
# única "puerta de calidad" de la rama de saliencia, equivalente funcional al
# CONF_YOLO_THRESHOLD de la rama YOLO:
#   - Si el blob ocupa más del techo, es más probable que sea fondo/textura global.
#   - Si ocupa menos del suelo, es más probable que sea ruido puntual.
# En ambos casos se marca sujeto_discreto=False en vez de fingir una detección.
# Valores heurísticos: calibrar en la validación 5.1 sobre el subconjunto anotado.
BLOB_AREA_MAX_RATIO = 0.60
BLOB_AREA_MIN_RATIO = 0.005

# Tamaño del bbox sintético en el caso residual (Otsu no encuentra ningún
# contorno). Fracción del lado de la imagen, centrado en el píxel de máxima
# saliencia (heurístico, sin base empírica).
BBOX_RATIO = 0.2


class SharedPerceptionEngine:
    def __init__(self):
        self.model = YOLO(str(YOLO_WEIGHTS_DIR / "yolo11m.pt"))
        SALIENCY_PATH.mkdir(parents=True, exist_ok=True)
        YOLO_PATH.mkdir(parents=True, exist_ok=True)
        # La carpeta de shared_perception la crea CapaVerificacion.

    def detect(self, img_path):
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {img_path}")

        h_img, w_img = img.shape[:2]
        area_img = float(h_img * w_img)

        # Se pasa la matriz de píxeles ya cargada (no la ruta) para evitar que
        # Ultralytics vuelva a leer el archivo de disco.
        result = self.model(img, verbose=False)[0]

        # Verificación visual pura: nadie la vuelve a leer, así que se acota el lado
        # mayor igual que las cinco que pasan por CapaVerificacion.guardar(). Es la
        # otra que se dibujaba a resolución completa (esta carpeta sumaba 257 MB).
        deteccion_yolo_path = YOLO_PATH / f"{img_path.stem}_yolo.png"
        escribir_verificacion(deteccion_yolo_path, result.plot())

        bbox = label = conf = source = None
        sujeto_discreto = False

        # --- RAMA 1: YOLO ---
        if len(result.boxes) > 0:
            i = result.boxes.conf.argmax()
            conf = result.boxes.conf[i].item()
            if conf >= CONF_YOLO_THRESHOLD:
                cx, cy, w, h = result.boxes.xywh[i].tolist()
                bbox = [cx, cy, w, h]
                label = self.model.names[int(result.boxes.cls[i])]
                source = "yolo"
                sujeto_discreto = True  # YOLO aporta identidad semántica del sujeto
            else:
                conf = None

        # --- RAMA 2: SALIENCIA (siempre se calcula: el mapa lo usa el Agente 1
        # para el equilibrio visual, con o sin sujeto detectado por YOLO) ---
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, sal_map_f = sal.computeSaliency(img)          # float en [0, 1]
        sal_map = (sal_map_f * 255).astype("uint8")      # versión para guardar/umbralizar

        saliency_map_path = SALIENCY_PATH / f"{img_path.stem}_saliency.png"
        cv2.imwrite(str(saliency_map_path), sal_map)

        thresh = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_area_ratio = None
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w_c, h_c = cv2.boundingRect(c)
            blob_area_ratio = cv2.contourArea(c) / area_img
        else:
            # Caso residual: mapa sin blobs separables. Muy improbable en
            # fotografía real (ruido y compresión siempre generan algo).
            py0, px0 = np.unravel_index(np.argmax(sal_map), sal_map.shape)
            w_c, h_c = int(w_img * BBOX_RATIO), int(h_img * BBOX_RATIO)
            x, y = int(px0 - w_c / 2), int(py0 - h_c / 2)

        # Clip a los límites de la imagen ANTES de derivar el centroide, para que
        # el bbox devuelto (y no solo el dibujado) sea siempre válido. Necesario
        # porque este rect puede inicializar cv2.grabCut (GC_INIT_WITH_RECT) en
        # el Agente 3, que exige un rectángulo contenido en la imagen.
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w_c), min(h_img, y + h_c)
        w_c, h_c = x2 - x1, y2 - y1
        px, py = x1 + w_c // 2, y1 + h_c // 2

        # Centroide del blob saliente principal (NO un pico de intensidad puro:
        # el argmax solo se usa en el caso residual sin contornos).
        centroide_saliencia = [int(px), int(py)]

        # Puerta de calidad de la rama de saliencia
        blob_plausible = (
            blob_area_ratio is not None
            and BLOB_AREA_MIN_RATIO <= blob_area_ratio <= BLOB_AREA_MAX_RATIO
        )

        if bbox is None:
            bbox = [float(px), float(py), float(w_c), float(h_c)]
            if blob_plausible:
                source = "saliencia"
                sujeto_discreto = True
            else:
                # No se finge una detección: se emite el bbox igualmente (los
                # agentes lo necesitan como referencia), pero marcado como no
                # fiable para que apliquen su plantilla de "no aplica" en vez de
                # anclar silenciosamente en textura de fondo.
                source = "sin_sujeto_claro"
                sujeto_discreto = False

        # --- Verificación visual del resultado FINAL de la percepción compartida ---
        # Se reutiliza la matriz ya cargada (`img`) para no releer del disco.
        capa = CapaVerificacion(img_path, "shared_perception", img=img)
        color = COLOR_POR_FUENTE[source]

        bx1 = max(0, int(bbox[0] - bbox[2] / 2))
        by1 = max(0, int(bbox[1] - bbox[3] / 2))
        bx2 = min(w_img, int(bbox[0] + bbox[2] / 2))
        by2 = min(h_img, int(bbox[1] + bbox[3] / 2))

        capa.rectangulo((bx1, by1), (bx2, by2), color)
        etiqueta = f"fuente={source}" + (f" | {label} {conf:.2f}" if label else "")
        capa.etiqueta((bx1, max(capa.radio * 2, by1 - capa.radio)), etiqueta, color)

        perception_bbox_path = capa.guardar("bbox_final")

        return {
            "bbox": bbox,
            "clase": label,
            "confianza_yolo": conf,
            "fuente": source,                     # yolo | saliencia | sin_sujeto_claro
            "sujeto_discreto": sujeto_discreto,   # los agentes deben consultar ESTE flag
            "blob_area_ratio": blob_area_ratio,
            "mapa_saliencia_path": str(saliency_map_path),
            "centroide_saliencia": centroide_saliencia,
            "deteccion_yolo_path": str(deteccion_yolo_path),
            "percepcion_bbox_path": str(perception_bbox_path),
        }


class SharedPerceptionInput(BaseModel):
    """Esquema de entrada para el análisis de imagen."""
    image_path: str = Field(..., description="La ruta local del archivo de imagen (.jpg, .png).")


shared_perception_engine = SharedPerceptionEngine()


class SharedPerceptionTool(BaseTool):
    name: str = "Percepción compartida"
    description: str = (
        "Detecta el sujeto principal de la imagen: bbox y clase (YOLO). "
        "Calcula también el mapa de saliencia por residuo espectral y el centroide "
        "de su región más destacada. Si YOLO no detecta con confianza_yolo suficiente, "
        "sintetiza un bbox a partir de esa región. Si además la región no supera "
        "los límites de plausibilidad de área, lo marca como 'sin_sujeto_claro' "
        "(sujeto_discreto=False) para que los especialistas no anclen su análisis "
        "en textura de fondo. Guarda una imagen con las detecciones YOLO y otra "
        "con el bbox final, indicando su fuente, para verificación visual."
    )
    args_schema: Type[BaseModel] = SharedPerceptionInput


    def _run(self, image_path):
        return shared_perception_engine.detect(image_path)