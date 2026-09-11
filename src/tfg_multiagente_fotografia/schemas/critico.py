"""Modelos de salida del Nivel 3 y de la crítica compositiva."""

from pydantic import BaseModel, Field

from tfg_multiagente_fotografia.schemas.orquestador import VectorPesosW


class ArbitrajeCritico(BaseModel):
    """Arbitraje determinista emitido por `PrioridadTool`."""

    etiqueta_contexto: str = Field(
        ..., description="Etiqueta con la que se seleccionó la fila de W."
    )
    vector_pesos: VectorPesosW = Field(
        ..., description="Fila de W aplicada, transcrita sin modificar."
    )
    orden_dimensiones: list[str] = Field(
        ..., description="Las cuatro dimensiones ordenadas por W descendente; no reordenar."
    )
    metricas_citables: list[str] = Field(
        ..., description="Rutas `dimension.metrica` cuya confianza es mayor que 0."
    )
    metricas_cerradas: list[str] = Field(
        ...,
        description="Rutas `dimension.metrica` con confianza 0; sus valores no son citables.",
    )
    cobertura_por_dimension: dict[str, float] = Field(
        ...,
        description=(
            "Fracción de métricas citables de cada dimensión, en [0,1]. "
            "Mide aplicabilidad, no calidad, y no modifica W."
        ),
    )


class AfirmacionMetrica(BaseModel):
    """Afirmación compositiva respaldada por métricas citables."""

    texto: str = Field(
        ..., description="Afirmación en prosa con citas inline `[metrica.campo = valor]`."
    )
    dimension: str = Field(
        ..., description="Dimensión a la que pertenece, usando la clave del arbitraje."
    )
    campos_citados: list[str] = Field(
        ...,
        min_length=1,
        description="Rutas `dimension.metrica.campo` citadas en texto; todas deben ser citables.",
    )


class Tension(BaseModel):
    """Hallazgos de distintas dimensiones que apuntan en sentidos opuestos."""

    texto: str = Field(
        ..., description="Descripción de la tensión con evidencia citable de ambos lados."
    )
    dimensiones: list[str] = Field(
        ..., min_length=2, description="Dimensiones implicadas, al menos dos."
    )
    resolucion_por_w: str = Field(
        ...,
        description="Dimensión que prevalece según W, citando su peso y el contexto.",
    )
    campos_citados: list[str] = Field(
        ...,
        min_length=1,
        description="Rutas `dimension.metrica.campo` citadas en texto.",
    )


class ObservacionVisual(BaseModel):
    """Hecho visible que no cubren las métricas del sistema."""

    texto: str = Field(
        ...,
        description=(
            "Observación sin cifras sobre el contenido de la escena; no duplica ni contradice "
            "las cuatro dimensiones medidas."
        ),
    )
    ambito: str = Field(
        ..., description="Tema observado, expresado en pocas palabras y fuera de las métricas."
    )


class CriticaCompositiva(BaseModel):
    """Salida del Nivel 3: arbitraje y crítica redactada por el LLM."""

    arbitraje: ArbitrajeCritico = Field(
        ..., description="Resultado de PrioridadTool, transcrito sin alterar."
    )
    afirmaciones: list[AfirmacionMetrica] = Field(
        ...,
        description="Afirmaciones respaldadas, ordenadas según arbitraje.orden_dimensiones.",
    )
    tensiones: list[Tension] = Field(
        default_factory=list,
        description="Tensiones reales resueltas mediante W; lista vacía si no existen.",
    )
    observaciones_visuales: list[ObservacionVisual] = Field(
        default_factory=list,
        description="Observaciones de contenido; lista vacía si vio_imagen es false.",
    )
    salvedades: list[str] = Field(
        default_factory=list,
        description="Una salvedad por cada métrica cerrada, redactada como hecho de la escena.",
    )
    sintesis: str = Field(
        ...,
        description=(
            "Crítica final en español, de unas 6-10 frases y siguiendo el orden del arbitraje. "
            "No introduce evidencia nueva: toda cifra procede de afirmaciones o tensiones."
        ),
    )
    vio_imagen: bool = Field(
        ..., description="True solo si el crítico utilizó la herramienta de lectura visual."
    )
