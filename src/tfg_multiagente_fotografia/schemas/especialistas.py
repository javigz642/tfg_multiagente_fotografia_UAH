from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class MetricaConfianza(BaseModel, Generic[T]):
    """Valor de una métrica junto con su normalización y aplicabilidad."""

    valor: T = Field(..., description="Resultado de la métrica en el dominio definido por su campo.")
    valor_norm: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Adherencia al patrón de la métrica en [0,1], donde 1 es máxima; no es una "
            "nota de calidad ni se compara entre métricas distintas. None si no está definida."
        ),
    )
    confianza: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Aplicabilidad de la medición en [0,1]. Con 0, su valor no es citable. "
            "No representa importancia contextual."
        ),
    )
    fuente_confianza: Optional[str] = Field(
        None,
        description="Motivo de la confianza asignada; None cuando no requiere salvedad.",
    )


class InformeComposicionEspacial(BaseModel):
    """Informe del Agente 1: regla de los tercios y equilibrio visual."""

    dimensiones_imagen: tuple[int, int] = Field(
        ..., description="Dimensiones (ancho, alto) de la imagen original, en píxeles."
    )

    # --- Regla de los tercios ---
    d_tercios: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Distancia del anclaje al punto de tercios más cercano, dividida por la diagonal; "
            "valor en [0,1/3] y valor_norm = 1 - 3*valor. Confianza 1 solo si fuente='yolo'."
        ),
    )
    coord_centroide: tuple[float, float] = Field(
        ...,
        description=(
            "Anclaje (x,y) normalizado: centro del bbox si sujeto_discreto=True; en otro caso, "
            "centroide de saliencia. La vía de anclaje no determina su confianza."
        ),
    )
    coord_p_cercano: tuple[float, float] = Field(
        ...,
        description="Punto de tercios más cercano al anclaje, en coordenadas (x,y) normalizadas.",
    )

    # --- Centrado ---
    d_centro: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Distancia del mismo anclaje al centro (0.5,0.5), dividida por la diagonal; "
            "valor en [0,1/2] y valor_norm = 1 - 2*valor. Comparte la confianza de d_tercios."
        ),
    )

    # --- Arbitraje entre patrones ---
    patron_dominante: MetricaConfianza[str] = Field(
        ...,
        description=(
            "Patrón 'centrada' o 'tercios', elegido por la menor distancia bruta. "
            "valor_norm es None y la confianza coincide con d_tercios."
        ),
    )
    margen_patron: float = Field(
        ...,
        ge=0.0,
        description=(
            "Diferencia absoluta entre d_tercios.valor y d_centro.valor, como fracción de "
            "diagonal; valores próximos a 0 indican empate."
        ),
    )

    # --- Equilibrio visual ---
    d_equilibrio: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Distancia del centro de masa de saliencia al centro de la imagen, dividida por la "
            "diagonal. Es global, tiene confianza 1 y valor_norm None."
        ),
    )
    coord_centro_masa: tuple[float, float] = Field(
        ..., description="Centro de masa de saliencia en coordenadas (x,y) normalizadas."
    )

    # --- Verificación visual ---
    verificacion_path: str = Field(
        ..., description="Ruta al PNG de verificación de la composición espacial."
    )


class DiagnosticoComposicionEspacial(BaseModel):
    """Salida del Agente 1: informe determinista y diagnóstico del LLM."""

    informe: InformeComposicionEspacial = Field(
        ..., description="Informe devuelto por la herramienta, transcrito sin alterar."
    )
    diagnostico: str = Field(
        ...,
        description=(
            "Diagnóstico en español de esta dimensión, basado en campos escalares citables del "
            "informe. Omite valores con confianza 0 y no prioriza ni valora la fotografía."
        ),
    )


class InformeLineasDireccion(BaseModel):
    """Informe del Agente 2: horizonte y convergencia perspectiva."""

    dimensiones_imagen: tuple[int, int] = Field(
        ..., description="Dimensiones (ancho, alto) de la imagen original, en píxeles."
    )

    # --- Nivelación del horizonte ---
    angulo_horizonte: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Ángulo firmado del candidato a horizonte en grados: positivo si cae hacia la "
            "derecha; |valor| <= 15. valor_norm = 1 - |valor|/15, o None sin candidato. "
            "Confianza 1 si hay candidato, longitud_horizonte >= 0.13 y |valor| <= 10."
        ),
    )
    longitud_horizonte: float = Field(
        ...,
        ge=0.0,
        description=(
            "Longitud del candidato a horizonte como fracción de la diagonal; 0 si no existe. "
            "Forma parte del gate de angulo_horizonte."
        ),
    )
    n_lineas_horizonte: int = Field(
        ...,
        ge=0,
        description="Número de segmentos candidatos dentro de ±15° de la horizontal.",
    )

    # --- Convergencia perspectiva ---
    score_convergencia: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Fracción de líneas candidatas que votan el punto de fuga ganador, en [0,1]; "
            "valor_norm = valor. Confianza 1 si n_lineas_fuga >= 2, valor supera "
            "score_convergencia_nulo y apertura_haz >= 35°."
        ),
    )
    coord_punto_fuga: Optional[tuple[float, float]] = Field(
        None,
        description=(
            "Punto de fuga en coordenadas (x,y) normalizadas, que pueden quedar fuera de [0,1]. "
            "None si el gate de convergencia está cerrado."
        ),
    )
    apertura_haz: float = Field(
        ...,
        ge=0.0,
        le=90.0,
        description=(
            "Mayor diferencia angular entre los inliers del haz ganador, en grados [0,90]; "
            "distingue convergencia de paralelismo."
        ),
    )
    score_convergencia_nulo: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Score de convergencia esperable por azar para n_lineas_fuga según el modelo nulo; "
            "es el umbral auditable del gate."
        ),
    )

    # --- Recuentos ---
    n_lineas_total: int = Field(
        ..., ge=0, description="Número total de segmentos detectados por Hough."
    )
    n_lineas_fuga: int = Field(
        ...,
        ge=0,
        description="Número de segmentos candidatos a fuga, con orientación entre 15° y 75°.",
    )
    n_lineas_inliers: int = Field(
        ...,
        ge=0,
        description=(
            "Candidatos que votan el punto ganador; score_convergencia.valor = "
            "n_lineas_inliers / n_lineas_fuga."
        ),
    )

    # --- Verificación visual ---
    verificacion_path: str = Field(
        ..., description="Ruta al PNG de verificación de horizonte y convergencia."
    )


class DiagnosticoLineasDireccion(BaseModel):
    """Salida del Agente 2: informe determinista y diagnóstico del LLM."""

    informe: InformeLineasDireccion = Field(
        ..., description="Informe devuelto por la herramienta, transcrito sin alterar."
    )
    diagnostico: str = Field(
        ...,
        description=(
            "Diagnóstico en español de líneas y dirección, basado en campos escalares citables. "
            "Trata los dos gates por separado, omite valores con confianza 0 y no emite juicios."
        ),
    )


class InformeLuzTono(BaseModel):
    """Informe del Agente 4: exposición, distribución tonal y armonía cromática."""

    dimensiones_imagen: tuple[int, int] = Field(
        ..., description="Dimensiones (ancho, alto) de la imagen original, en píxeles."
    )

    # --- Exposición y distribución tonal ---
    media_L: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Media del canal L de CIELAB en [0,100]: por debajo de 50 indica clave baja y por "
            "encima, clave alta. valor_norm = 1 - |valor - 50|/50 y confianza 1 siempre."
        ),
    )
    std_L: float = Field(
        ...,
        ge=0.0,
        description=(
            "Desviación típica del canal L; mide contraste tonal global y se interpreta junto "
            "a media_L. Su máximo teórico en [0,100] es 50."
        ),
    )
    pct_clipping_sombras: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Porcentaje de píxeles con L < 5, sin detalle en sombras.",
    )
    pct_clipping_luces: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Porcentaje de píxeles con L > 95, sin detalle en altas luces.",
    )

    # --- Armonía cromática ---
    esquema_cromatico: MetricaConfianza[str] = Field(
        ...,
        description=(
            "Esquema 'monocromático' (<30°), 'análogo' (<60°), 'complementario' "
            "(par a 180°±30°), 'otro' o 'sin_color' con gate cerrado. 'Monocromático' se refiere "
            "solo a los píxeles cromáticos. valor_norm es None; confianza 1 si "
            "ratio_pixeles_cromaticos >= 0.10."
        ),
    )
    spread_cromatico: float = Field(
        ...,
        ge=0.0,
        le=180.0,
        description=(
            "Máxima distancia circular entre matices dominantes, en grados [0,180]. "
            "Vale 0 con un solo matiz o con el gate cerrado."
        ),
    )
    matices_dominantes: list[float] = Field(
        ...,
        description=(
            "Matices dominantes en grados [0,360), ordenados por masa descendente y redondeados "
            "a una decimal. Lista vacía con el gate cerrado."
        ),
    )
    n_matices_dominantes: int = Field(
        ...,
        ge=0,
        description="Número de elementos de matices_dominantes; 0 con el gate cerrado.",
    )
    ratio_pixeles_cromaticos: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Fracción de píxeles con saturación y valor suficientes, en [0,1]. Abre el gate "
            "cromático desde 0.10; 1 - valor es la fracción acromática."
        ),
    )

    # --- Verificación visual ---
    verificacion_path: str = Field(
        ..., description="Ruta al PNG de verificación de exposición y armonía cromática."
    )


class DiagnosticoLuzTono(BaseModel):
    """Salida del Agente 4: informe determinista y diagnóstico del LLM."""

    informe: InformeLuzTono = Field(
        ..., description="Informe devuelto por la herramienta, transcrito sin alterar."
    )
    diagnostico: str = Field(
        ...,
        description=(
            "Diagnóstico en español de luz y tono, basado en campos escalares citables. "
            "No afirma una paleta con confianza 0 ni convierte las métricas en juicios de calidad."
        ),
    )


class InformeEspacioAislamiento(BaseModel):
    """Informe del Agente 3: espacio negativo y separación figura-fondo."""

    dimensiones_imagen: tuple[int, int] = Field(
        ..., description="Dimensiones (ancho, alto) de la imagen original, en píxeles."
    )

    # --- Espacio negativo ---
    ratio_espacio_negativo: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Fracción del encuadre no ocupada por la máscara del sujeto, en [0,1]. "
            "valor_norm es None y la confianza vale 1 solo si fuente='yolo'."
        ),
    )

    # --- Separación figura-fondo ---
    ratio_nitidez: MetricaConfianza[float] = Field(
        ...,
        description=(
            "Índice de densidad de detalle (var_figura - var_fondo)/(var_figura + var_fondo), "
            "en [-1,1]: positivo favorece a la figura. No mide enfoque. "
            "valor_norm = (valor + 1)/2; confianza 1 si fuente='yolo' y existe textura suficiente."
        ),
    )
    var_laplaciano_figura: float = Field(
        ...,
        ge=0.0,
        description=(
            "Varianza del Laplaciano dentro de la máscara, medida en el espacio de trabajo; "
            "se interpreta solo frente a var_laplaciano_fondo."
        ),
    )
    var_laplaciano_fondo: float = Field(
        ...,
        ge=0.0,
        description=(
            "Varianza del Laplaciano fuera de la máscara, en las mismas condiciones que la "
            "varianza de la figura."
        ),
    )

    # --- Procedencia de la máscara ---
    fuente_mascara: str = Field(
        ...,
        description=(
            "'grabcut' si la máscara se inicializó con el bbox o 'prior_saliencia' si procede "
            "del mapa de saliencia. La vía de segmentación no determina por sí sola la confianza."
        ),
    )

    # --- Verificación visual ---
    verificacion_path: str = Field(
        ..., description="Ruta al PNG de verificación de la máscara y sus métricas."
    )


class DiagnosticoEspacioAislamiento(BaseModel):
    """Salida del Agente 3: informe determinista y diagnóstico del LLM."""

    informe: InformeEspacioAislamiento = Field(
        ..., description="Informe devuelto por la herramienta, transcrito sin alterar."
    )
    diagnostico: str = Field(
        ...,
        description=(
            "Diagnóstico en español sobre espacio y densidad de detalle, basado en campos "
            "escalares citables. Omite valores con confianza 0 y no interpreta detalle como enfoque."
        ),
    )
