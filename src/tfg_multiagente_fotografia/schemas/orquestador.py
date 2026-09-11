from typing import Optional

from pydantic import BaseModel, Field, model_validator


class PercepcionCompartida(BaseModel):
    """Esquema real devuelto por SharedPerceptionTool."""
    bbox: list[float] = Field(..., description="[cx, cy, w, h] del sujeto o bbox de referencia.")
    clase: Optional[str] = Field(None, description="Clase COCO, o None si fuente no es 'yolo'.")
    confianza_yolo: Optional[float] = Field(
        None, description="Confianza YOLO, o None si fuente no es 'yolo'."
    )
    fuente: str = Field(
        ..., description="Origen del bbox: 'yolo', 'saliencia' o 'sin_sujeto_claro'."
    )
    sujeto_discreto: bool = Field(
        ...,
        description=(
            "Indica si existe un sujeto localizable y elige la vía de anclaje o segmentación. "
            "La aplicabilidad semántica de las métricas dependientes del sujeto la determina fuente."
        ),
    )
    blob_area_ratio: Optional[float] = Field(
        None, description="Área relativa del blob de saliencia; None si no hay contornos."
    )
    mapa_saliencia_path: str = Field(..., description="Ruta al mapa de saliencia por residuo espectral")
    centroide_saliencia: list[int] = Field(..., description="[x, y] del centroide de saliencia.")
    deteccion_yolo_path: str = Field(..., description="Ruta al PNG con las detecciones YOLO.")
    percepcion_bbox_path: str = Field(..., description="Ruta al PNG de verificación del bbox final.")


class VectorPesosW(BaseModel):
    """Fila de pesos contextuales W, validada tras leer el fichero de parámetros."""
    composicion_espacial: float = Field(..., ge=0.0, le=1.0)
    lineas_direccion: float = Field(..., ge=0.0, le=1.0)
    espacio_aislamiento_sujeto: float = Field(..., ge=0.0, le=1.0)
    luz_tono: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _suma_uno(self):
        """Comprueba que la fila de pesos suma 1."""
        total = (
            self.composicion_espacial
            + self.lineas_direccion
            + self.espacio_aislamiento_sujeto
            + self.luz_tono
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Los 4 pesos de W deben sumar 1, suman {total}")
        return self


class SalidaOrquestador(BaseModel):
    """Salida estructurada del Nivel 1, sin incluir el vector W."""
    etiqueta_contexto: str = Field(..., description='"animal", "arquitectura", "paisaje", "producto-still_life", "retrato-humano" u "otro"')
    percepcion_compartida: PercepcionCompartida
