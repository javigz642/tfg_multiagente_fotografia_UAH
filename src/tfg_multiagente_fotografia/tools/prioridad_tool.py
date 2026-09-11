"""Parte determinista del Nivel 3: arbitraje por W y comprobación de informes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from tfg_multiagente_fotografia.rutas import dir_ejecucion
from tfg_multiagente_fotografia.schemas.critico import ArbitrajeCritico
from tfg_multiagente_fotografia.schemas.orquestador import VectorPesosW
from tfg_multiagente_fotografia.tools.vector_pesos_tool import vector_pesos_engine

# Las CLAVES son las de `VectorPesosW` y no un nombre libre: son las mismas con las que se
# indexa W, así que una discrepancia de nombre revienta al leer el peso en vez de colarse.
# Su ORDEN es además el desempate determinista del ordenamiento (ver `_ordenar_dimensiones`).
# Ojo a la asimetría del tercero: la clave de W es `espacio_aislamiento_sujeto` y el fichero
# se llama `salida_espacio_aislamiento.json`. El mapeo es explícito justamente por eso.
INFORMES = {
    "composicion_espacial": "salida_composicion_espacial.json",
    "lineas_direccion": "salida_lineas_direccion.json",
    "espacio_aislamiento_sujeto": "salida_espacio_aislamiento.json",
    "luz_tono": "salida_luz_tono.json",
}


def es_metrica(campo) -> bool:
    """Devuelve si un campo tiene la forma de una `MetricaConfianza`."""
    return isinstance(campo, dict) and "valor" in campo and "confianza" in campo


class PrioridadEngine:
    """Cruza los cuatro informes con la fila de W y emite el arbitraje."""

    def __init__(self, raiz: Path | None = None):
        """Fija opcionalmente la raíz de los informes; por defecto se deriva de la imagen."""
        self.raiz = raiz

    def _raiz_de(self, ruta_imagen: str | Path) -> Path:
        return self.raiz if self.raiz is not None else dir_ejecucion(ruta_imagen)

    # -- carga ------------------------------------------------------------------------

    def _cargar_informe(self, dimension: str, ruta_imagen: str | Path) -> dict:
        """Lee un `salida_*.json` y comprueba que habla de la imagen que se está analizando."""
        destino = self._raiz_de(ruta_imagen) / INFORMES[dimension]
        if not destino.exists():
            raise ValueError(
                f"Falta el informe de '{dimension}' ({destino.name}). El crítico no puede "
                f"arbitrar sin las cuatro dimensiones: ejecuta antes las tareas del Nivel 2."
            )

        datos = json.loads(destino.read_text(encoding="utf-8"))
        # Los especialistas emiten {informe, diagnostico}; aquí solo interesa la parte
        # medida. Se acepta también el informe pelado para poder fabricar casos en los tests.
        informe = datos.get("informe", datos)

        esperado = Path(ruta_imagen).stem
        verificacion = informe.get("verificacion_path")
        if verificacion:
            # El PNG de verificación se llama `{stem}_{sufijo}.png`, así que el guion bajo
            # evita que `imagen3` case con `imagen30`.
            if not Path(verificacion).stem.startswith(f"{esperado}_"):
                raise ValueError(
                    f"El informe de '{dimension}' ({destino.name}) NO corresponde a "
                    f"'{esperado}': su verificacion_path apunta a "
                    f"'{Path(verificacion).name}'. Son informes de una ejecución anterior; "
                    f"arbitrar con ellos daría una crítica coherente y falsa."
                )
        return informe

    # -- arbitraje ---------------------------------------------------------------------

    @staticmethod
    def _ordenar_dimensiones(pesos: dict) -> list[str]:
        """Ordena por peso descendente, con desempate DETERMINISTA.

        El desempate no es un adorno: la fila 'otro' de W es uniforme (0.25 en las cuatro
        dimensiones), así que sin un segundo criterio el orden dependería de detalles de
        implementación y dos ejecuciones idénticas podrían producir críticas ordenadas de
        forma distinta. Se desempata por el orden canónico de `INFORMES`, que es el de los
        campos de `VectorPesosW`.
        """
        canonico = list(INFORMES)
        return sorted(canonico, key=lambda d: (-pesos[d], canonico.index(d)))

    def arbitrar(self, ruta_imagen: str | Path, etiqueta_contexto: str) -> ArbitrajeCritico:
        # `obtener` no cae en la fila "otro" si la etiqueta es desconocida: lanza. Ese fallo
        # ruidoso se propaga a propósito — significa que el clasificador y el JSON de pesos
        # se han desincronizado, y enmascararlo con pesos uniformes sería lo peor posible.
        pesos = vector_pesos_engine.obtener(etiqueta_contexto)

        citables: list[str] = []
        cerradas: list[str] = []
        cobertura: dict[str, float] = {}

        for dimension in INFORMES:
            informe = self._cargar_informe(dimension, ruta_imagen)

            metricas = {k: v for k, v in informe.items() if es_metrica(v)}
            if not metricas:
                raise ValueError(
                    f"El informe de '{dimension}' no contiene ninguna métrica con la forma "
                    f"{{valor, confianza}}. O el fichero está corrupto o el contrato de "
                    f"MetricaConfianza ha cambiado y este módulo se ha quedado atrás."
                )

            abiertas = 0
            for nombre, metrica in sorted(metricas.items()):
                ruta = f"{dimension}.{nombre}"
                # La comparación es `> 0` y no `== 1`: el contrato admite confianzas
                # intermedias (el refinamiento por concentración de saliencia que el Agente 1
                # dejó anotado), aunque hoy los cuatro especialistas la emitan binaria.
                if metrica["confianza"] > 0:
                    citables.append(ruta)
                    abiertas += 1
                else:
                    cerradas.append(ruta)

            cobertura[dimension] = round(abiertas / len(metricas), 4)

        return ArbitrajeCritico(
            etiqueta_contexto=etiqueta_contexto,
            vector_pesos=VectorPesosW(**pesos),
            orden_dimensiones=self._ordenar_dimensiones(pesos),
            metricas_citables=citables,
            metricas_cerradas=cerradas,
            cobertura_por_dimension=cobertura,
        )


prioridad_engine = PrioridadEngine()


class PrioridadInput(BaseModel):
    ruta_imagen: str = Field(
        ...,
        description=(
            "Ruta de la fotografía que se está analizando. Sirve para comprobar que los cuatro "
            "informes en disco corresponden a ESTA imagen y no a una ejecución anterior."
        ),
    )
    etiqueta_contexto: str = Field(
        ...,
        description=(
            'Etiqueta de contexto del orquestador: "animal", "arquitectura", "paisaje", '
            '"producto-still_life", "retrato-humano" u "otro". Selecciona la fila de W.'
        ),
    )


class PrioridadTool(BaseTool):
    name: str = "Arbitraje de prioridades"
    description: str = (
        "Aplica el vector de pesos contextuales W sobre los cuatro informes de los "
        "especialistas y devuelve el arbitraje: el ORDEN en que deben tratarse las cuatro "
        "dimensiones, qué métricas son CITABLES, cuáles están CERRADAS y la cobertura de "
        "cada dimensión. Lee los informes de disco: no hay que pasárselos. "
        "IMPORTANTE: este arbitraje no es una sugerencia, es la estructura de la crítica. El "
        "orden de las dimensiones se respeta tal cual, y una métrica que aparece en "
        "metricas_cerradas NO se cita con ningún valor por convincente que parezca — de eso "
        "se habla en las salvedades, diciendo qué no se puede afirmar y por qué. "
        "La cobertura es un recuento de qué se pudo MEDIR, no una nota de calidad: una "
        "cobertura baja en la dimensión de mayor peso es en sí misma la observación más "
        "relevante que puede hacerse sobre esa fotografía, no un motivo para pasarla por alto."
    )
    args_schema: Type[BaseModel] = PrioridadInput

    def _run(self, ruta_imagen, etiqueta_contexto):
        return prioridad_engine.arbitrar(ruta_imagen, etiqueta_contexto).model_dump()
