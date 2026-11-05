# Juicio de aplicabilidad de un segundo LLM (Claude) sobre el corpus C2

- **Fecha:** 2026-10-17. **Juez:** Claude Opus 5.5 (`claude-opus-5-5`), en Claude Code.
- **Fichero de datos:** [`juicio_claude.csv`](juicio_claude.csv), con una fila por imagen, las ocho preguntas del prompt (`si` / `no` / `dudoso`) y un comentario que justifica la decisión.
- **Integración:** `uv run python experimentos/aplicabilidad_vlm.py --juez experimentos/resultados/aplicabilidad_vlm/juicio_claude.csv` añade las columnas `juez`, `juez_abre` y `comentario_juez` a la hoja *Detalle* y un bloque «juez» a la hoja *Resumen*.

## Qué es y qué no es

Es una **segunda opinión de un LLM juez**, independiente de Gemini y de la heurística. **No es la verdad de referencia.** Claude también es un modelo de visión y lenguaje, así que comparte con Gemini los sesgos propios de esta familia de modelos. Su acuerdo con Gemini indica que la decisión semántica es **estable entre modelos distintos**, pero no demuestra que sea correcta. Los falsos positivos y negativos solo pueden medirse frente a la anotación humana (A2).

## Procedimiento

1. Las 100 imágenes se redujeron a 1024 px de lado mayor, la misma resolución que recibe Gemini.
2. Cada imagen se juzgó **una sola vez, sin conocer** la decisión de la heurística ni la de Gemini, aplicando las mismas ocho definiciones operativas del prompt (hoja *Prompt* del Excel): color, horizonte natural, horizontal estructural, perspectiva, protagonista, espacio, nitidez y anclaje.
3. Se usó «dudoso» solo cuando no era posible decidir. En el cruce, «dudoso» cuenta como no abrir, igual que en Gemini.

## Resultados

| Condición | Abre heurística | Abre Gemini | Abre Claude | Acuerdo Gemini–Claude | Acuerdo heurística–Claude |
|---|---|---|---|---|---|
| color | 80 | 85 | 72 | 87 % | 92 % |
| horizonte_natural | 29 | 28 | 16 | 88 % | 73 % |
| horizonte_amplio | 29 | 69 | 31 | 60 % | 74 % |
| convergencia | 26 | 24 | 12 | 88 % | 80 % |
| espacio | 61 | 72 | 65 | 89 % | 84 % |
| nitidez | 57 | 73 | 65 | 88 % | 78 % |
| composicion | 61 | 73 | 71 | 94 % | 82 % |

**Lectura:** los dos LLM coinciden entre sí en un 87-94 % de las decisiones, salvo en el horizonte amplio. Esa coincidencia es mayor que la de cualquiera de los dos con la heurística en las condiciones semánticas. La excepción es el color, la única condición puramente fotométrica.

### Casos en que la heurística abre y los dos LLM dicen que no aplica

Son los candidatos más sólidos a falso positivo de la heurística, y los primeros que conviene revisar a mano:

- **horizonte_natural (16):** 5, 6, 7, 8, 9, 15, 40, 42, 45, 48, 60, 63, 70, 71, 72, 77. Casi todos son horizontales estructurales: fachadas, persianas, peldaños, bordes de mesa o barandillas. Es el modo de fallo descrito en el apartado 9.7.
- **convergencia (13):** 6, 7, 8, 14, 35, 42, 51, 59, 67, 72, 85, 86, 94. Son fachadas frontales, retratos, animales, bodegones o mesas cenitales sin perspectiva que organice la imagen.
- **espacio (3):** 15, 86, 93. **nitidez (2):** 15, 86. **composición (2):** 15, 86.

### Casos en que la heurística cierra y los dos LLM dicen que aplica

Son candidatos a falso negativo, el error que el diseño tolera:

- **horizonte_natural (7):** 23, 34, 35, 36, 50, 64, 97. Son horizontes reales que la heurística no capta o que el sujeto interrumpe.
- **convergencia (3):** 2, 10, 11.
- **identidad (espacio, nitidez y composición, entre 9 y 14):** 16, 27, 33, 34, 66, 76, 87, 88, 96… Son torres, casas, árboles, faros, relojes de pulsera o joyas, sujetos claros que YOLO no detecta con confianza ≥ 0,75 o que COCO no tiene como clase. Es la limitación de cobertura semántica del apartado 9.7.

### Desacuerdo entre los dos LLM: horizontal estructural

En el horizonte amplio, Gemini abre 69 imágenes y Claude 31. La pregunta «¿hay una horizontal que cruce buena parte del encuadre?» es ambigua: Gemini acepta tejados, forjados o líneas de suelo parciales, y Claude solo las que cruzan casi todo el encuadre. **La definición amplia no es estable entre modelos.** Antes de usarla en la memoria convendría concretarla, por ejemplo con «cruza más de la mitad del ancho».

## Observaciones sobre el corpus

- **imagen61 es un duplicado exacto de imagen55** (mismo MD5, `e86d490f…`). Cuenta dos veces en todas las tasas del corpus.
- Las imágenes 24, 25 y 26 son versiones giradas de 19, 20 y 21, con −7,95°, +7,95° y −2,95°. Los tres decisores coinciden en abrir el horizonte en las seis, que son los casos de control del horizonte desnivelado.

## Limitaciones de este juicio

- Una sola pasada y un solo juez: no se ha medido la reproducibilidad del propio juicio.
- Las definiciones las redactó el mismo asistente que juzga, y siguen la guía A2. Un anotador humano puede interpretar de otra forma los casos límite, como el horizonte tapado por montañas o los bodegones de varios objetos.
- Los comentarios describen lo que se ve a 1024 px. No sustituyen la revisión a resolución completa.
