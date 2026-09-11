from collections.abc import Callable

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tasks.task_output import TaskOutput

from tfg_multiagente_fotografia.tools.context_classifier_tool import ContextClassifierTool
from tfg_multiagente_fotografia.tools.shared_perception_tool import SharedPerceptionTool
from tfg_multiagente_fotografia.tools.composicion_espacial_tool import ComposicionEspacialTool
from tfg_multiagente_fotografia.tools.lineas_direccion_tool import LineasDireccionTool
from tfg_multiagente_fotografia.tools.luz_tono_tool import LuzTonoTool
from tfg_multiagente_fotografia.tools.espacio_aislamiento_tool import EspacioAislamientoTool
from tfg_multiagente_fotografia.tools.prioridad_tool import PrioridadTool
from tfg_multiagente_fotografia.tools.lectura_visual_tool import LecturaVisualTool
from tfg_multiagente_fotografia.schemas.orquestador import SalidaOrquestador
from tfg_multiagente_fotografia.schemas.critico import CriticaCompositiva
from tfg_multiagente_fotografia.schemas.especialistas import (
    DiagnosticoComposicionEspacial,
    DiagnosticoLineasDireccion,
    DiagnosticoLuzTono,
    DiagnosticoEspacioAislamiento,
)

# Controla si el crítico dispone de la tercera fuente: la lectura visual de la fotografía.
FLAG_CRITICO_VE_IMAGEN = True

# El crítico usa un modelo distinto porque recibe el contexto más grande y debe invocar
# `PrioridadTool` y, opcionalmente, `LecturaVisualTool`.
MODELO_CRITICO = "gemini/gemini-2.5-pro"

# Los informes se escriben en el directorio independiente calculado por `main.py`.

@CrewBase
class TfgMultiagenteFotografia():
    """Crew de seis agentes para analizar la composición fotográfica."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def orchestrator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['orchestrator_agent'], # type: ignore[index]
            verbose=True,
            tools = [ContextClassifierTool(), SharedPerceptionTool()],
        )

    @agent
    def composicion_espacial_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['composicion_espacial_agent'], # type: ignore[index]
            verbose=True,
            tools=[ComposicionEspacialTool()],
            allow_delegation=False,
        )

    @agent
    def lineas_direccion_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['lineas_direccion_agent'], # type: ignore[index]
            verbose=True,
            tools=[LineasDireccionTool()],
            allow_delegation=False,
        )

    @agent
    def luz_tono_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['luz_tono_agent'], # type: ignore[index]
            verbose=True,
            tools=[LuzTonoTool()],
            allow_delegation=False,
        )

    @agent
    def espacio_aislamiento_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['espacio_aislamiento_agent'], # type: ignore[index]
            verbose=True,
            tools=[EspacioAislamientoTool()],
            allow_delegation=False,
        )

    @agent
    def critico_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['critico_agent'], # type: ignore[index]
            verbose=True,
            llm=LLM(model=MODELO_CRITICO),
            tools=[PrioridadTool()] + ([LecturaVisualTool()] if FLAG_CRITICO_VE_IMAGEN else []),
            allow_delegation=False,
        )

    @task
    def compose_task(self) -> Task:
        return Task(
            config=self.tasks_config['compose_task'], # type: ignore[index]
            output_pydantic=SalidaOrquestador,
            output_file="{dir_ejecucion}/salida_orquestador.json",
        )

    # PRUEBA: async_execution=True en las cuatro tareas del Nivel 2. Son
    # ortogonales entre sí (ninguna aparece en el `context` de otra), así que Process.sequential
    # las lanza en hilos propios nada más terminar compose_task, y espera a las cuatro antes de
    # ejecutar critica_task (que sí las tiene a las cinco en su `context`). Si no convence, basta
    # con quitar `async_execution=True` de las cuatro.
    @task
    def composicion_espacial_task(self) -> Task:
        return Task(
            config=self.tasks_config['composicion_espacial_task'], # type: ignore[index]
            context=[self.compose_task()],
            async_execution=True,
            output_pydantic=DiagnosticoComposicionEspacial,
            output_file="{dir_ejecucion}/salida_composicion_espacial.json",
        )

    # No recibe el contexto de tareas anteriores: sus métricas son globales.
    @task
    def lineas_direccion_task(self) -> Task:
        return Task(
            config=self.tasks_config['lineas_direccion_task'], # type: ignore[index]
            context=[],
            async_execution=True,
            output_pydantic=DiagnosticoLineasDireccion,
            output_file="{dir_ejecucion}/salida_lineas_direccion.json",
        )

    # No recibe el contexto de tareas anteriores: sus métricas son globales.
    @task
    def luz_tono_task(self) -> Task:
        return Task(
            config=self.tasks_config['luz_tono_task'], # type: ignore[index]
            context=[],
            async_execution=True,
            output_pydantic=DiagnosticoLuzTono,
            output_file="{dir_ejecucion}/salida_luz_tono.json",
        )

    # Recibe la percepción compartida, pero no los informes de otros especialistas.
    @task
    def espacio_aislamiento_task(self) -> Task:
        return Task(
            config=self.tasks_config['espacio_aislamiento_task'], # type: ignore[index]
            context=[self.compose_task()],
            async_execution=True,
            output_pydantic=DiagnosticoEspacioAislamiento,
            output_file="{dir_ejecucion}/salida_espacio_aislamiento.json",
        )

    # El crítico es el único que recibe las cinco tareas anteriores.
    @task
    def critica_task(self) -> Task:
        return Task(
            config=self.tasks_config['critica_task'], # type: ignore[index]
            context=[
                self.compose_task(),
                self.composicion_espacial_task(),
                self.lineas_direccion_task(),
                self.luz_tono_task(),
                self.espacio_aislamiento_task(),
            ],
            output_pydantic=CriticaCompositiva,
            output_file="{dir_ejecucion}/salida_critica.json",
        )

    @crew
    def crew(self, task_callback: Callable[[TaskOutput], None] | None = None) -> Crew:
        """Construye el crew y, opcionalmente, conecta el callback de progreso."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose = True,
            tracing=True,
            task_callback=task_callback,
        )
