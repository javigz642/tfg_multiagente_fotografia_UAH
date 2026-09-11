from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tfg_multiagente_fotografia.rutas import OUTPUTS_PATH, PROYECT_PATH
from tfg_multiagente_fotografia.tools.cnn_context import ContextClassifier

# Los TRES artefactos del clasificador viven juntos: los pesos, el mapeo de etiquetas con
# el que se entrenó la cabeza y el umbral de rechazo. Son una unidad —unos pesos sin su
# class_to_idx no se pueden interpretar— y antes estaban partidos en dos niveles, con el
# .pt suelto en la raíz de models/ y sus dos JSON en la subcarpeta.
MODELS_DIR = PROYECT_PATH / "models" / "clasificador_contexto"

classifier = ContextClassifier(
    weights_path=MODELS_DIR / "pesos_fine_tuning_ligero.pt",
    class_to_idx_path=MODELS_DIR / "class_to_idx.json",
    threshold_path=MODELS_DIR / "umbral.json",
    gradcam_dir=OUTPUTS_PATH / "gradcam",
)


class ContextClassifierInput(BaseModel):
    image_path: str = Field(..., description="Ruta a la imagen a clasificar")



class ContextClassifierTool(BaseTool):
    name: str = "Clasificador de Contexto Fotografico"
    description: str = (
        "Clasifica el contexto de una foto (animal, arquitectura, paisaje, "
        "producto-still_life, retrato-humano, u 'otro' si la confianza es baja) "
        "y genera un mapa Grad-CAM de explicabilidad."
    )
    args_schema: type[BaseModel] = ContextClassifierInput

    def _run(self, image_path: str) -> dict:
        return classifier.predict(image_path)