"""Carga el vector contextual W desde un fichero versionado."""


import json
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tfg_multiagente_fotografia.schemas.orquestador import VectorPesosW

# En `parametros/` y no en `config/`: ahí viven los dos ficheros que se editan para el
# análisis de sensibilidad de §5, mientras que `config/` es el directorio que CrewAI
# descubre por convención y solo debe contener agents.yaml y tasks.yaml.
PESOS_PATH = Path(__file__).resolve().parents[1] / "parametros" / "pesos_contextuales.json"


class VectorPesosEngine:
    """Carga la matriz W una vez y devuelve la fila correspondiente a un contexto."""

    def __init__(self):
        with open(PESOS_PATH, encoding="utf-8") as f:
            datos = json.load(f)

        self.columnas = datos["columnas"]
        # Cada fila se valida al cargar (los 4 pesos deben sumar 1): si alguien edita
        # el JSON a mano para el análisis de sensibilidad, el fallo salta aquí y no
        # a mitad de una crítica ya generada.
        self.pesos = {
            etiqueta: VectorPesosW(**dict(zip(self.columnas, fila)))
            for etiqueta, fila in datos["pesos"].items()
        }

    def obtener(self, etiqueta_contexto: str) -> dict:
        if etiqueta_contexto not in self.pesos:
            # No se cae a la fila "otro" en silencio: una etiqueta desconocida
            # significa que el clasificador y este fichero se han desincronizado,
            # y eso hay que verlo, no enmascararlo con pesos uniformes.
            raise ValueError(
                f"Contexto '{etiqueta_contexto}' no está en {PESOS_PATH.name}. "
                f"Etiquetas válidas: {sorted(self.pesos)}"
            )
        return self.pesos[etiqueta_contexto].model_dump()


class VectorPesosInput(BaseModel):
    etiqueta_contexto: str = Field(
        ...,
        description=(
            'Etiqueta de contexto devuelta por el clasificador: "animal", "arquitectura", '
            '"paisaje", "producto-still_life", "retrato-humano" u "otro".'
        ),
    )


vector_pesos_engine = VectorPesosEngine()


class VectorPesosTool(BaseTool):
    name: str = "Vector de pesos contextuales"
    description: str = (
        "Devuelve el vector de pesos contextuales W correspondiente a una etiqueta de "
        "contexto fotográfico. W indica la IMPORTANCIA relativa de cada dimensión de "
        "análisis en ese tipo de escena, con las claves 'composicion_espacial', "
        "'lineas_direccion', 'espacio_aislamiento_sujeto' y 'luz_tono' (suman 1). "
        "Es una hipótesis de diseño cargada de un fichero de configuración, no un valor "
        "calculado ni opinable. OJO: W es importancia, NO fiabilidad; la fiabilidad de "
        "cada medición viaja en el campo 'confianza' de cada métrica de los especialistas."
    )
    args_schema: Type[BaseModel] = VectorPesosInput

    def _run(self, etiqueta_contexto):
        return vector_pesos_engine.obtener(etiqueta_contexto)
