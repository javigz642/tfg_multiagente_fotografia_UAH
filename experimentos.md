# Validación experimental (capítulos 8 y 9): plan de trabajo

Plan para validar el sistema. Sirve de base para el capítulo 8 (diseño de la validación) y para el
capítulo 9 (resultados). **No contiene resultados.** Fija qué se mide, sobre qué conjunto y con qué
criterio, y todo ello se decide **antes** de ejecutar.

- **Fecha del plan:** 2026-11-05. **Rama:** `agente3-sam`.
- **Convención:**
  - **[verificado]**: comprobado en el código o ejecutando algo;
  - **[pendiente]**: está por hacer.

---

## 1. Qué se valida y qué no

El sistema no emite un juicio estético. Por eso la validación no mide si una crítica acierta
artísticamente. Tampoco demuestra que el vector W sea óptimo, porque W es una hipótesis de diseño. Lo
que se mide son las **garantías de ingeniería** que el sistema dice ofrecer:

| Pregunta | Qué comprueba | Objetivos (§1.4.2) | Experimento | Resultados |
|---|---|---|---|---|
| **Contexto** | ¿El clasificador asigna el contexto correcto y rechaza lo que no reconoce? | 1.º y 3.º | E1 | §9.1 |
| **Aplicabilidad** | ¿Las condiciones de aplicabilidad cierran una métrica cuando su regla no aplica? | 2.º | E2 | §9.2 |
| **Citas** | ¿Las cifras del texto salen del informe, sin alterarse y sin citar métricas cerradas? ¿El crítico respeta el arbitraje? | 2.º y 3.º | E3 | §9.3 |
| **Reproducibilidad y coste** | ¿Los registros son fieles a las herramientas? ¿Cuánto cuesta y tarda un análisis? | 1.º | E4 | §9.4 |
| Interfaz | ¿Es usable? | 4.º | — | No se valida (no hay estudio con usuarios) |

Tampoco se compara con un sistema de agente único.

---

## 2. El sistema que se valida

### 2.1 Tubería

| Nivel | Pieza | Qué es determinista | Qué pasa por un LLM | Salida en disco |
|---|---|---|---|---|
| 1 | `ContextClassifierTool` (ResNet-18) + `SharedPerceptionTool` (YOLO `yolo11m.pt` + saliencia espectral) | Todo el cálculo | El orquestador copia la etiqueta y la percepción | `outputs/ejecucion/{stem}/salida_orquestador.json` |
| 2 | 4 especialistas: composición espacial, líneas y dirección, luz y tono, espacio y aislamiento (máscara del sujeto con **SAM 2.1 base**, en CPU) | Las métricas y sus `confianza` | El LLM **transcribe** el informe y redacta `diagnostico` con citas `[campo = valor]` | `salida_{composicion_espacial,lineas_direccion,luz_tono,espacio_aislamiento}.json` |
| 3 | Crítico: `PrioridadTool` (arbitraje por W) + `LecturaVisualTool` | El arbitraje | Transcribe el arbitraje y redacta `CriticaCompositiva` | `salida_critica.json` |

Configuración:

- **Orquestador y especialistas:** `gemini/gemini-2.5-flash`.
- **Crítico:** `gemini/gemini-2.5-pro`.
- **Lectura visual:** `gemini-2.5-flash`, temperatura 0,2, con caché en `outputs/lectura_visual/`.
- **Ejecución:** las 4 tareas del Nivel 2 son asíncronas.

**Hecho clave.** El `informe` de cada `salida_*.json` **no es la salida directa de la herramienta**.
Es la copia que hace el LLM, validada con `output_pydantic`. Por eso E2 mide la aplicabilidad
ejecutando las herramientas, no leyendo los JSON. Las copias del LLM se evalúan aparte, en E3 y E4.

### 2.2 Condiciones de aplicabilidad

| Condición | Métricas | Abre (`confianza = 1`) si… | Constantes |
|---|---|---|---|
| Cromática | `esquema_cromatico` | `ratio_pixeles_cromaticos ≥ RATIO_CROMATICO_MIN` | `RATIO_CROMATICO_MIN = 0.17` |
| Horizonte | `angulo_horizonte` | Hay candidato, `longitud ≥ 0.1517` y `\|ángulo\| ≤ 10°` | `LONGITUD_MIN_HORIZONTE`, `ANGULO_MAX_HORIZONTE` |
| Convergencia | `score_convergencia` | `n ≥ 2`, `score > score_nulo(n)` y `apertura ≥ 89°` | `MIN_LINEAS_FUGA`, `modelo_nulo_fuga.json`, `APERTURA_MIN_GRADOS` |
| Espacio | `ratio_espacio_negativo` | `fuente == "yolo"` **y** segmentación fiable | `FUENTE_CON_IDENTIDAD`, `SAM_SCORE_MIN`, `RELLENO_MIN`, `RELLENO_MAX` |
| Nitidez | `ratio_nitidez` | Lo mismo que espacio **y** `max(var_fig, var_fondo) ≥ VAR_LAPLACIANO_MIN` (2594) | Las anteriores + `VAR_LAPLACIANO_MIN` |
| Composición | `d_tercios`, `d_centro`, `patron_dominante` | `fuente == "yolo"` (una sola condición para las tres) | `FUENTE_CON_IDENTIDAD` |
| — | `media_L`, `d_equilibrio` | Siempre abiertas | — |

- **Segmentación fiable:** `score_segmentacion ≥ SAM_SCORE_MIN` y
  `RELLENO_MIN ≤ relleno_bbox ≤ RELLENO_MAX`. El score es la calidad que SAM estima para su propia
  máscara. El relleno es la fracción de la caja YOLO que ocupa la máscara.
- **Percepción.** Antes de todas las condiciones de identidad está `CONF_YOLO_THRESHOLD = 0.75`, que
  decide si `fuente = "yolo"`. Se elige la caja de **mayor confianza**, no la más grande ni la más
  central.
- **Recuento.** Hay 8 métricas con condición, pero solo **6 decisiones independientes**, porque las tres
  de composición comparten una.

### 2.3 Entorno y reproducibilidad

- **Toda la validación se ejecuta en una sola máquina:** el equipo Windows con GPU RTX 4060 y CUDA.
- **Versiones:** `torch==2.6.0` y `torchvision==0.21.0`, fijadas en `pyproject.toml` para todas las
  plataformas. El resto está congelado en `uv.lock`.
- **SAM corre en CPU** (`DEVICE_SAM = "cpu"`). La máscara no tiene azar, pero varía en algún píxel suelto
  entre CPU y GPU [verificado].
- **Por qué no basta con fijar versiones:** con las mismas versiones, dos máquinas no dan resultados
  idénticos bit a bit. Con torch 2.6.0 en las dos, YOLO detecta sujeto en 61 fotos del conjunto de
  calibración en un Mac y en 60 en el equipo Windows [verificado]. La versión única es una garantía
  adicional; la regla sigue siendo una sola máquina.

---

## 3. Conjuntos de imágenes

| Conjunto | Dónde está | Tamaño | Papel |
|---|---|---|---|
| **EVA-test** | `Curso_CNN_FineTunning/clasificador_contexto/data/splits/test.csv` + `EVA_together/{image_id}.jpg` | 725 (136 animal, 143 arquitectura, 152 paisaje, 139 producto, 155 retrato) | E1 |
| **EVA-other** (`sort = 6`) | `.../splits/otro.csv` | 273 | E1: sustituto aproximado (*near-OOD*) de imágenes fuera de distribución |
| **EVA-val** | `.../splits/val.csv` | 725 | E1: control del umbral de rechazo |
| **Conjunto de calibración** | `data/imagen{1..100}.jpg` | 100 fotos de Unsplash | **Solo para ajustar umbrales** (§4) y para las fotos de repetición |
| **Conjunto de prueba** | `data_prueba/test{1..40}.jpg` | 40 fotos de Unsplash no vistas | **Todos los resultados de E1 (parte externa), E2, E3 y E4** (§5) |
| **Fotos de repetición** | `imagen62` (animal), `imagen9` (arquitectura), `imagen35` (paisaje), `imagen86` (producto), `imagen50` (retrato), del conjunto de calibración | 5 × 3 repeticiones | E4 |

**Por qué hay dos conjuntos de Unsplash.** Si se calibra y se mide sobre las mismas fotos, las tasas de
error salen optimistas. El conjunto de calibración se usa para ajustar los umbrales y el de prueba, que
nadie ha mirado al calibrar, para medir. Las fotos de repetición salen del conjunto de calibración porque
la reproducibilidad no depende de la calibración.

**Conjunto de calibración.** Son 100 fotos con selección intencional y estratificada: 18 por categoría y
10 de «otro», con subtipos pensados para abrir y cerrar cada condición (Tabla 13 de la memoria). Tienen
entre 2,07 y 25,96 MP. Falta **versionar el manifiesto** `experimentos/manifiesto_calibracion.csv`, con
nombre, MD5, dimensiones, URL y autor [pendiente].

**EVA.** No se copian imágenes al repositorio: la ruta va en la configuración. Los pesos del sistema son
idénticos al checkpoint `fase4_fine_tuning_ligero.pt` del proyecto externo (MD5
`20ce2c480390ccf0a06f3e67bb1225ca`) [verificado]. E1 sobre EVA evalúa, por tanto, exactamente el modelo
desplegado.

---

## 4. Calibración de umbrales (conjunto de calibración)

Se hace **antes** de ejecutar nada sobre el conjunto de prueba, con el método de huecos: se ordena la
variable de decisión y se busca el salto limpio entre las imágenes que deben abrir y las que deben
cerrar. Ante dos huecos igual de limpios se toma el más estricto, porque un falso positivo rompe la
verificabilidad y un falso negativo solo hace callar al sistema. El capítulo 6 muestra la distribución
que justifica cada umbral.

| Componente | Parámetro (valor actual) | Cómo se fija |
|---|---|---|
| Espacio | `SAM_SCORE_MIN = 0.90` (provisional) | Huecos sobre el score en las fotos `yolo` |
| Espacio | `RELLENO_MIN = 0.20` / `RELLENO_MAX = 0.95` (provisionales) | Huecos sobre el relleno; revisar los objetos rectangulares que rozan el techo |
| Espacio | `VAR_LAPLACIANO_MIN = 50` (provisional) | Huecos sobre `var_max` en escala log, en las fotos `yolo` |
| Espacio | `MARGEN_FRONTERA_RATIO = 0.01` | Solo documentar la sensibilidad del índice de nitidez |
| Luz y tono | `RATIO_CROMATICO_MIN = 0.09` | Histograma del ratio con el corte; hay casos pegados a él (0,086; 0,093; 0,099) |
| Luz y tono | `PESO_MIN_MATIZ = 0.05` | Ablación fusión/filtro con el valor actual |
| Luz y tono | `UMBRAL_FUSION_MATIZ = 15`, `SAT_MIN` / `VAL_MIN = 0.15` | Solo documentar |
| Líneas | `LONGITUD_MIN_HORIZONTE = 0.13`, `ANGULO_MAX_HORIZONTE = 10` | Rectángulo de aceptación sobre el conjunto de calibración, con la definición amplia del horizonte (§6.2) |
| Líneas | `APERTURA_MIN_GRADOS = 35` | Histograma de apertura sobre el conjunto de calibración |
| Líneas | `modelo_nulo_fuga.json` | No depende del corpus; solo documentar |
| Percepción | `CONF_YOLO_THRESHOLD = 0.75` | Barrido de identidad según el umbral; documentar |
| Percepción | `BLOB_AREA_MIN/MAX_RATIO` | No se activan en el conjunto de calibración: declararlos como no ejercitados |
| Composición | Escalas de lectura del prompt (margen 0,01/0,15; `d_equilibrio` 0,03/0,12) | Recalcular los percentiles sobre el conjunto de calibración y actualizar `agents.yaml` |
| Clasificador | `umbral_otro = 0.85` | No se toca; se controla con EVA-val (E1) |
| Nivel 3 | W (versión 1.1) | Hipótesis de diseño: no se calibra |

Para calibrar el Agente 3 hace falta revisar la segmentación (`bien` / `parcial` / `mal`) de las fotos
`yolo` del conjunto de calibración. Se revisa sobre imágenes propias, sin cifras (`calibracion.py`).

### 4.1 Regla para elegir cada umbral

Fijada el 2026-11-07, **antes de medir**, para que ningún corte se elija después de ver los datos.

**Qué fotos son positivas y cuáles negativas.** Para cada umbral se ordenan las fotos del conjunto de
calibración por su variable de decisión. Cada foto se marca según la anotación: **positiva** si la regla
aplica y **negativa** si no aplica. Las `dudoso` se excluyen.

| Umbral | Variable | Fotos | Positiva si… (anotación) |
|---|---|---|---|
| `RATIO_CROMATICO_MIN` | `ratio_cromatico` | Las 100 | `color = si` |
| `LONGITUD_MIN_HORIZONTE`, `ANGULO_MAX_HORIZONTE` | `longitud_horizonte`, `angulo_horizonte` | Las 100 | Horizonte amplio = `si` (natural **o** estructural) |
| `APERTURA_MIN_GRADOS` | `apertura_haz` | Las que superan el modelo nulo | `perspectiva = si` |
| `SAM_SCORE_MIN`, `RELLENO_MIN`, `RELLENO_MAX` | `score_sam`, `relleno_bbox` | Las `yolo` | Segmentación `bien` o `parcial` (negativa: `mal`) |
| `VAR_LAPLACIANO_MIN` | `var_max` (escala log) | Las `yolo` con segmentación fiable | `nitidez = si` |

**Cómo se elige el corte:**

1. **Si hay banda vacía** (todas las negativas a un lado y todas las positivas al otro), el umbral se
   pone dentro de la banda, en un valor redondeado lo más cerca posible de su extremo **más exigente
   para abrir**. Así ninguna foto se equivoca en la calibración y queda el mayor margen frente a los
   falsos positivos. Si hay dos bandas igual de limpias, se toma la más estricta.
2. **Si no hay banda vacía** (las clases se cruzan), entre todos los cortes posibles se descartan los
   que dejan abrir más de 1 de cada 10 negativas (tasa de falsos positivos > 10 %, la misma tolerancia
   que el criterio de E2). De los que quedan, se toma el de **mayor sensibilidad**; si empatan, el más
   estricto. Si la sensibilidad resultante baja del 60 %, se declara como punto débil, pero el corte no
   se cambia.
3. **Si hay menos de 3 fotos positivas o menos de 3 negativas**, el umbral no se puede calibrar. Se
   mantiene el valor actual si no deja abrir ninguna negativa; si deja abrir alguna, se endurece lo
   justo para cerrarla. En los dos casos se declara como «comprobado, no calibrado».
4. **Umbrales que actúan juntos** (longitud y ángulo del horizonte; score y relleno de SAM): el candidato
   es la pareja de valores. Una negativa que cae dentro de la zona de apertura cuenta como falso
   positivo, y se aplican las mismas reglas 1 a 3.

Los umbrales que no son de huecos siguen su propio método de la tabla anterior: el barrido de YOLO se
documenta y las escalas de composición son percentiles. **El conjunto de prueba nunca interviene en la
calibración:** si allí una foto cae en el lado equivocado, es un error que mide E2, no un motivo para
mover el corte.

Después de calibrar:

1. quitar la etiqueta «PROVISIONAL» del código y actualizar los comentarios que citan cifras;
2. actualizar la Figura 28, que muestra 0,90 / 0,20 / 0,95 (`docs/diagramas/espacio_aislamiento_pipeline.html`);
3. ejecutar todos los scripts de `tests/`;
4. hacer commit y congelar (§9).

### 4.2 Resultado de la calibración (2026-11-07)

Script `experimentos/analisis_calibracion.py`; detalle por umbral (VP/FP/FN/VN y fotos que fallan) en
`resultados/calibracion/propuesta_umbrales.csv` y gráficas `huecos_*.png`. La tabla del §4 conserva los
valores de partida.

| Umbral | Antes | Ahora | Regla | Tasa FP | Sensibilidad |
|---|---|---|---|---|---|
| `RATIO_CROMATICO_MIN` | 0.09 | **0.17** | 2 | 2/23 | 76/76 |
| `LONGITUD_MIN_HORIZONTE` | 0.13 | **0.1517** | 2, con el ángulo fijo (desviación) | 5/66 | 17/32 (53 %) |
| `ANGULO_MAX_HORIZONTE` | 10 | 10 | sin recalibrar (desviación) | — | — |
| `APERTURA_MIN_GRADOS` | 35 | **89** | 2 | 3/35 | 4/14 (29 %) |
| `SAM_SCORE_MIN`, `RELLENO_MIN/MAX` | 0.90; 0.20–0.95 | sin cambios | 3 | — | 60/60 |
| `VAR_LAPLACIANO_MIN` | 50 | **2594** | 2 | 1/12 | 10/48 (21 %) |
| Escalas de composición | margen 0.01/0.15; `d_equilibrio` 0.03/0.12 | margen 0.01/0.13; `d_equilibrio` 0.02/0.14 | P10 / P90 | — | — |

- **Única desviación de la regla: el horizonte.** La pareja que da la regla es (0.09; 0°), porque 28 de
  86 candidatos miden exactamente 0°. Con un ángulo máximo de 0°, `angulo_horizonte` valdría siempre 0
  al abrir y la métrica no podría señalar un horizonte torcido. Se deja el ángulo en su valor de diseño
  (10°) y se calibra solo la longitud. El criterio para desviarse es que el corte deje la métrica sin
  sentido, no que dé cifras malas.
- **Convergencia y nitidez se aplican tal cual** aunque su sensibilidad quede por debajo del 60 %: son
  puntos débiles declarados (§8). En la convergencia, muchas escenas sin perspectiva dan 80–90° de
  apertura (rejillas frontales). En la nitidez, 7 de las 12 negativas lo son porque la caja de YOLO no es
  el protagonista, y solo un corte muy alto las cierra.
- **No calibrados por huecos:** `CONF_YOLO_THRESHOLD` se queda en 0.75 (con 0.90 se pierden 5 cajas
  buenas para quitar 5 malas, `barrido_yolo.csv`); `BLOB_*` no se activan; la composición abre en las 11
  fotos con caja ≠ protagonista.

---

## 5. El conjunto de prueba

### 5.1 Composición (40 fotos)

La lógica es la misma que la del conjunto de calibración: 7 fotos por categoría y 5 para «otro». En
cada categoría hay casos en los que cada condición **debería abrirse** y casos en los que **debería
cerrarse**. Además se refuerzan los dos puntos débiles de la percepción, que son la elección del sujeto
y la segmentación difícil.

Los códigos de fila (A1…O5) solo identifican el objetivo de selección y aparecen en el manifiesto.

**Arquitectura (7)**

| # | Foto | Qué pone a prueba |
|---|---|---|
| A1 | Calle o pasillo con un punto de fuga claro | Convergencia abre |
| A2 | Fachada frontal plana, sin perspectiva | Convergencia cierra. Con la definición amplia, sus horizontales **sí** cuentan como referencia de nivel |
| A3 | Contrapicado de un edificio alto (verticales que convergen) | Convergencia, caso límite |
| A4 | Interior con perspectiva (estación, iglesia, biblioteca) | Convergencia en interior |
| A5 | Edificio aislado como protagonista (una casa sola) | Sujeto fuera de COCO: falso negativo esperado |
| A6 | Arquitectura en blanco y negro | Condición cromática cierra |
| A7 | Plaza o calle con una persona pequeña de paso | YOLO detecta a alguien que no es el protagonista |

**Paisaje (7)**

| # | Foto | Qué pone a prueba |
|---|---|---|
| P1 | Mar o lago con el horizonte nivelado | Horizonte abre |
| P2 | Horizonte claramente inclinado (más de 10°) | Horizonte cierra por ángulo |
| P3 | Bosque cerrado o montaña sin horizonte visible | Horizonte cierra |
| P4 | Camino, vía o embarcadero que se aleja | Convergencia fuera de la arquitectura |
| P5 | Árbol solitario o faro | Sujeto fuera de COCO; vía de saliencia |
| P6 | Niebla o paisaje casi monocromo | Condición cromática cierra |
| P7 | Prado amplio con un animal o una persona pequeños | Sujeto pequeño: segmentación y espacio negativo alto |

**Retrato (7)**

| # | Foto | Qué pone a prueba |
|---|---|---|
| R1 | Fondo desenfocado y pelo suelto | Espacio y nitidez abren; contorno difícil |
| R2 | Fondo nítido y texturado (calle, ladrillo) | Índice de nitidez negativo |
| R3 | Persona pequeña en un entorno amplio | Espacio negativo alto |
| R4 | Retrato en blanco y negro | Condición cromática cierra |
| R5 | Clave baja, sujeto en parte en sombra | Segmentación difícil |
| R6 | Primer plano que llena el encuadre o cortado por los bordes | Espacio negativo bajo; relleno alto |
| R7 | Dos o más personas | Elección del sujeto |

**Animal (7)**

| # | Foto | Qué pone a prueba |
|---|---|---|
| N1 | Animal de COCO (perro, gato, pájaro) con fondo desenfocado | Espacio y nitidez abren |
| N2 | Animal de COCO en un entorno amplio (caballo, vaca) | Espacio negativo alto; horizonte posible |
| N3 | Pelaje largo o plumas | Contorno difuso |
| N4 | Especie fuera de COCO (zorro, ciervo, lagarto) | YOLO no la detecta o la confunde de clase |
| N5 | Animal camuflado | Saliencia y segmentación difíciles |
| N6 | Animal tapado en parte (valla, ramas) | Máscara fragmentada |
| N7 | Grupo o rebaño | Elección del sujeto |

**Producto / bodegón (7)**

| # | Foto | Qué pone a prueba |
|---|---|---|
| D1 | Objeto de COCO aislado sobre fondo neutro (taza, botella, reloj) | Caso de control fácil |
| D2 | Blanco sobre blanco | Condición de textura cierra |
| D3 | Objeto oscuro sobre fondo negro | Segmentación y textura |
| D4 | Bodegón con varios objetos | Elección del sujeto |
| D5 | Objeto fuera de COCO (joya, perfume, cámara) | Falso negativo esperado |
| D6 | Vidrio o superficie reflectante (copa, botella transparente) | Segmentación difícil |
| D7 | Objeto rectangular que llena su caja (libro, cuadro, tableta) | Techo del relleno (`RELLENO_MAX`) |

**Otro (5)**

| # | Foto | Qué pone a prueba |
|---|---|---|
| O1 | Vehículo sin personas | Rechazo a «otro» y pesos uniformes |
| O2 | Deporte con varias personas en movimiento | Elección del sujeto; movimiento |
| O3 | Fuegos artificiales o escena nocturna abstracta | Sin sujeto discreto |
| O4 | Textura que llena todo el encuadre (hojas, arena, un patrón) | Rama `sin_sujeto_claro`, que **no se da nunca en el conjunto de calibración** |
| O5 | Escena sin dominante (persona y animal a la par, un mercado) | Contexto ambiguo |

**Comprobación de cobertura.** Tras la anotación, cuenta cuántas fotos aplican y cuántas no en cada
condición. Se buscan **al menos 4 de cada lado** en las condiciones cromática, de horizonte, de
convergencia, de identidad y de textura; si alguna no llega, se declara. Si una foto no cumple lo que se
esperaba de ella, **se queda igual**: el subtipo es un objetivo de selección, no una etiqueta. La
etiqueta real la da la anotación ciega.

### 5.2 Reglas de selección

- **Fuente:** Unsplash, con la misma licencia que el conjunto de calibración y, si es posible, de
  fotógrafos distintos.
- **Una búsqueda por fila de la tabla:** se coge la **primera foto que cumpla claramente la
  descripción**, no la que «parece que va a salir bien».
- **Ninguna foto del conjunto de prueba pasa por el sistema, la app ni los tests hasta congelar.**
  - Se guardan en `data_prueba/`, no en `data/`, porque los tests recorren `data/`.
  - `data_prueba/` está en `.gitignore`.
- **No se sustituye ninguna foto después de ver resultados.** Si una resulta inválida (corrupta o
  duplicada del conjunto de calibración), se sustituye con la misma regla y se declara.
- **Variedad** de orientación, y resolución parecida a la del conjunto de calibración (de 2 a 26 MP).
- **Manifiesto** `experimentos/manifiesto_prueba.csv` con: nombre (`test1`…`test40`), fila de la tabla,
  URL, autor, MD5, dimensiones y fecha de descarga.

**Cómo se hizo (2026-11-05)** [verificado]. La selección la ejecutó un asistente con
`experimentos/descargar_prueba.py` y la API oficial de Unsplash. Las consultas se fijaron antes de
buscar; en cada fila se revisaron las 15 primeras candidatas por relevancia y se tomó la primera que
cumplía claramente la descripción. Se descartaron automáticamente las fotos de pago, las que no tenían
entre 2 y 26 MP y las casi idénticas a una del conjunto de calibración (hash perceptual). La foto elegida
y los motivos de cada descarte están en `experimentos/seleccion_prueba.json` y en el manifiesto. Los
nombres `testN` se asignan con una permutación de semilla fija, para que el número no delate la
categoría. **El manifiesto, `seleccion_prueba.json` y `data_prueba/_seleccion/` no se abren hasta
terminar la anotación de contexto y de aplicabilidad.**

- **Excepción: P2 es una foto derivada.** En cuatro consultas (60 candidatas) no apareció ningún
  horizonte inclinado más de 10°. Se tomó la primera candidata válida de la primera consulta de P2, que
  no está en el conjunto de prueba, y se giró 15° con recorte al mayor rectángulo inscrito de la misma
  proporción (sin esquinas vacías). El ángulo se conoce por construcción. Se declara en el §8.3 de la
  memoria.
- **Para interpretar E2:** la guía de aplicabilidad pregunta si hay una referencia horizontal legible,
  no por su ángulo. En P2 la anotación será `si`; si el sistema cierra la métrica por el ángulo, la foto
  cuenta como falso negativo, que es el error tolerable.

### 5.3 Anotación del conjunto de prueba

El autor es el único anotador y sigue la guía del §6.2.

1. **Anotación de contexto y anotación de aplicabilidad, a ciegas**, antes de ejecutar nada.
2. **Anotación de segmentación y protagonista**, solo en las fotos con `fuente = yolo`, después de
   congelar y **antes de mirar ninguna decisión del sistema** (§6.2).
   - Segmentación: `bien` / `parcial` / `mal`.
   - Protagonista: `si` / `no`.
   - **Mismo procedimiento que en la calibración:** `calibracion.py` (`medir` → `plantilla` →
     `exportar`) sobre `data_prueba/`. Las imágenes muestran la foto, la caja y la máscara, pero
     **ni cifras ni decisiones**. Esta anotación no puede ser ciega a la máscara, pero sí lo es a lo
     que decidió el sistema. La de aplicabilidad es ciega del todo.
3. **Opcional: segundo anotador.** Si otra persona anota la aplicabilidad de 15 fotos, se informa el
   acuerdo (κ de Cohen) por condición. Reduce el peso de la limitación del anotador único.

---

## 6. Experimentos y criterios de éxito

### 6.0 Criterios de éxito (Tabla 14 de la memoria)

Se fijan **antes** de ejecutar sobre el conjunto de prueba y no se relajan después. Todos se escriben con
«≥» y «≤». Se juzgan sobre la **estimación puntual**; el IC de Wilson al 95 % se informa siempre al lado,
para que se vea la incertidumbre que deja una muestra de 40 fotos.

| Experimento | Conjunto | Criterios de éxito | Descriptivo (sin umbral) |
|---|---|---|---|
| **E1** Clasificador | EVA-test, EVA-val, EVA-other y conjunto de prueba | (a) F1 ≥ 0,75 en cada clase de EVA-test; (b) exactitud sobre las aceptadas del conjunto de prueba ≥ 0,85; (c) ≤ 20 % de las 35 fotos de las cinco clases enviado a «otro» | Cuántas de las 5 fotos de «otro» se rechazan; tabla de cobertura en EVA-val y EVA-other |
| **E2** Aplicabilidad | Conjunto de prueba anotado | (a) Precisión de apertura global ≥ 90 %; (b) precisión de apertura ≥ 80 % en cada condición; (c) sensibilidad global ≥ 60 % | Tasa de FP y de FN por condición; horizonte con la definición estrecha |
| **E3** Citas | Tanda de prueba (40 análisis) | (a) ≥ 95 % de citas exactas; (b) ≤ 1 % de citas a métricas cerradas, en los especialistas y en el crítico por separado; (c) afirmaciones en el orden del arbitraje en ≥ 95 % de las críticas | Resto de controles del crítico; ejecuciones que fallan al primer intento |
| **E4** Reproducibilidad y coste | Fotos de repetición y tanda de prueba | (a) Capa Python idéntica bit a bit en las 3 ejecuciones (100 %); (b) ≥ 95 % de informes transcritos fieles a la herramienta; (c) arbitraje transcrito igual al recalculado en ≥ 95 % de las críticas | Coste, latencia, memoria, fallos; variabilidad entre las 3 repeticiones del sistema completo |

**Veredicto por experimento:** «sí» si cumple todos sus criterios, «en parte» si cumple alguno y «no» si
no cumple ninguno.

**Por qué estos criterios.** La primera campaña (sobre el conjunto de calibración, antes de SAM) usó
criterios absolutos: 0 falsos positivos, 0 citas a métricas cerradas y 100 % de registros idénticos.
Resultaron inadecuados por dos motivos:

1. **No son medibles con 40 fotos.** Aunque una condición dé 0 falsos positivos sobre 15 negativos, el
   IC al 95 % de su tasa llega al 20 %. «Cero errores» no demuestra lo que pretende y, en cambio, un solo
   error lo tumba.
2. **No son compatibles con un componente no determinista.** El LLM copia los informes y el arbitraje;
   exigirle el 100 % contradice el propio límite declarado en el §8.

Los nuevos criterios distinguen lo que el sistema garantiza **por construcción** (la capa Python, a la
que se le sigue exigiendo el 100 %) de lo que garantiza **con una tolerancia** (las decisiones
heurísticas y las copias del LLM, a las que se les exige un error de como mucho 1 de cada 10 o 1 de
cada 20). La revisión se hizo el 2026-11-05, **antes de abrir el conjunto de prueba**, y se declara en
el §8.5 de la memoria.

**Justificación de cada umbral:**

- **E1.** Se mantienen los de la primera campaña. El criterio (c) se calcula ahora sobre las 35 fotos de
  las cinco clases: las 5 de «otro» están elegidas para ser rechazadas y ocuparían por sí solas 12,5 de
  los 20 puntos.
- **E2.** La **precisión de apertura** (VP / (VP + FP)) mide la garantía que se ofrece al usuario: «si el
  sistema cita una métrica, su regla aplica». El 90 % global admite como mucho 1 apertura indebida de cada
  10; el 80 % por condición tolera que una condición concreta, con pocas aperturas, tenga algún error
  más. La sensibilidad mínima evita que el criterio se cumpla cerrándolo todo. Una condición con menos de
  5 aperturas se informa, pero no se juzga por separado; sí entra en el global.
- **E3.** Una cita a una métrica cerrada de cada 100 es el mismo orden de tolerancia que el resto. El
  95 % de citas exactas y de críticas en orden equivale a como mucho 1 error de cada 20 (2 de las 40
  críticas).
- **E4.** La capa Python es determinista en una misma máquina, así que ahí el 100 % sí es justo: una
  diferencia sería un fallo real. La fidelidad de las copias se mide sobre la tanda de prueba (160
  informes y 40 arbitrajes) y no sobre las 5 fotos de repetición, que dan una muestra demasiado pequeña
  para un umbral. Las repeticiones quedan como dato descriptivo.

### 6.1 E1: clasificador de contexto → §9.1

- **No usa LLM.** Se hace entero en un notebook en pocos minutos de GPU.
- **Procedimiento:**
  1. **EVA-test, sin umbral:** exactitud, F1 por clase, matriz de confusión e IC del 95 %.
     - Transform: Resize 256 → CenterCrop 224 → normalización ImageNet.
     - Sirve también de comprobación de integridad: debe dar 0,8510.
  2. **EVA-test + EVA-other, con 0,85:** cobertura, exactitud sobre las aceptadas, errores capturados,
     aciertos perdidos y % de EVA-other enviado a «otro».
  3. **EVA-val + EVA-other:** la misma tabla, para comprobar que 0,85 no depende de haberse elegido con
     EVA-test.
  4. **Conjunto de prueba:** exactitud frente a la anotación de contexto (6 etiquetas), exactitud sobre
     las aceptadas, % de las 35 fotos de las cinco clases enviado a «otro», rechazos entre las 5 de
     «otro», matriz de confusión 6×6 y confianza frente a EVA.
  5. **Grad-CAM:** un acierto y un error del par que más se confunde (producto → arquitectura).
- Los pasos 1 a 3 no dependen del segmentador ni del conjunto de prueba. Si ya hay predicciones guardadas
  con los mismos pesos (mismo MD5), no hace falta repetirlos.
- **Resultados:** `experimentos/resultados/E1/predicciones_{eva_test,eva_other,eva_val,prueba}.csv`,
  `metricas.json`, matrices de confusión y Grad-CAM.
- **Limitaciones:** el umbral se eligió con EVA-test; EVA-other es *near-OOD*; el conjunto de prueba lo
  anota un único anotador.

### 6.2 E2: condiciones de aplicabilidad → §9.2

- **No usa API.** Se ejecutan las cuatro herramientas sobre el conjunto de prueba y se cruza cada decisión
  con la anotación.
- **Guía de anotación de aplicabilidad.** Se responde `si` / `no` / `dudoso` a cada pregunta (Tabla 15):

  | Condición | Métricas | La regla aplica si… |
  |---|---|---|
  | Cromática | `esquema_cromatico` | La imagen tiene color analizable (no es B/N, nocturna ni desaturada) |
  | Horizonte (**amplia, principal**) | `angulo_horizonte` | Hay una referencia horizontal legible para juzgar si la foto está nivelada: un horizonte natural **o** una horizontal estructural de la escena (cornisa, línea del suelo, orilla, borde de mesa) que cruza más de la mitad del ancho |
  | Horizonte (estrecha, secundaria) | `angulo_horizonte` | Hay un horizonte natural legible (cielo contra tierra o mar) |
  | Convergencia | `score_convergencia` | La profundidad se organiza por perspectiva lineal |
  | Espacio | `ratio_espacio_negativo` | Hay sujeto discreto y, si la fuente es YOLO, la segmentación no es `mal`. En la variante estricta, además, tampoco puede ser `parcial`: tiene que ser `bien` |
  | Nitidez | `ratio_nitidez` | Lo mismo que espacio |
  | Composición | `d_tercios`, `d_centro`, `patron_dominante` | Hay sujeto discreto y, si la fuente es YOLO, el objeto detectado es el protagonista |

  Se anotan por separado las dos preguntas de horizonte (`horizonte_natural` y `horizontal_estructural`);
  la amplia es `si` si cualquiera de las dos lo es.

  **Por qué la definición amplia es la principal.** `angulo_horizonte` mide si la foto está nivelada.
  En arquitectura o en bodegón, una cornisa o el borde de una mesa son una referencia legítima para eso,
  así que abrir la métrica con ellas no es un error. La definición estrecha penalizaba ese comportamiento
  y concentró la mayoría de los falsos positivos de horizonte de la primera campaña. Se informa como
  análisis secundario.

  La anotación de contexto usa animal / arquitectura / paisaje / producto / retrato / otro.

- **Definición de error:**
  - **Falso positivo** (grave): la métrica está abierta y la regla no aplica.
  - **Falso negativo** (tolerable): la métrica está cerrada y la regla aplica.
  - Las imágenes `dudoso` se excluyen de la matriz y se cuentan aparte.
- **Medidas:**
  - **Precisión de apertura** = VP / (VP + FP): de las veces que la métrica se abre, cuántas aplica.
    Global (sumando las 6 condiciones) y por condición.
  - **Sensibilidad** = VP / (VP + FN): de las veces que la regla aplica, cuántas se abre. Global.
  - Tasa de FP = FP / (FP + VN) y tasa de FN, por condición, con IC de Wilson.
- **Procedimiento:**
  0. **Primero, la anotación de segmentación y protagonista (§5.3)**, sin abrir el CSV de medidas.
     Motivo: esa anotación es parte de la verdad de referencia de espacio, nitidez y composición. Si
     se hiciera después de ver qué abrió el sistema, dejaría de ser independiente.
  1. Ejecutar los 4 engines sobre las 40 fotos. Extraer por imagen la `confianza`, el motivo de cierre
     y la variable de decisión de cada métrica.
  2. Replicar cada decisión a partir de los campos del informe y comprobar con `assert` que coincide
     con la del código (40 × 6 = 240 decisiones).
  3. **Tabla 1:** % de fotos con cada métrica abierta (con IC), reparto de `fuente`
     (`yolo` / `saliencia` / `sin_sujeto_claro`), cobertura por dimensión y motivos de cierre.
  4. **Tabla 2:** por condición, VP, FP, FN, VN y dudosas; precisión de apertura, tasa de FP con IC y
     tasa de FN; lista de imágenes de cada error. Una fila adicional con el horizonte estrecho y otra con
     la variante estricta de espacio y nitidez.
- **Resultados:** `experimentos/resultados/E2/gates_prueba.csv` (una fila por imagen y métrica),
  `tabla1.csv`, `tabla2.csv`, `fp_fn.json` y hojas de revisión de cada falso positivo.

### 6.3 E3: fidelidad de las citas → §9.3

- **Tanda de prueba:** las 40 fotos del conjunto de prueba, una vez cada una, con lectura visual. Las
  ejecuciones las hace el lanzador (§7) y el análisis un notebook.
- **Verificador** (`experimentos/comun.py`):
  - **Extracción:** cada cita `[campo = valor]` de los diagnósticos y de la crítica, incluidos los
    valores tupla de los campos `coord_*`.
  - **Resolución:** en los especialistas, `metrica.campo` o `campo`. En el crítico,
    `dimension.metrica.campo`, más una gramática explícita para `W(dim)`, `vector_pesos.dim` y
    `cobertura_por_dimension.dim`, que no deben contar como «campo inexistente».
  - **Clase de cada cita:** `exacta` (redondeo correcto a los decimales escritos, máximo 4) /
    `truncada` / `alterada` / `campo_inexistente` / `null_citado` / `texto_distinto`.
  - **Cita a métrica cerrada:** es violación citar `valor`, `valor_norm` o un campo de evidencia que
    depende de la métrica:

    | Métrica cerrada | Campos que tampoco se pueden citar |
    |---|---|
    | `esquema_cromatico` | `spread_cromatico`, `matices_dominantes[*]`, `n_matices_dominantes` |
    | `score_convergencia` | `coord_punto_fuga`, `apertura_haz`, `n_lineas_inliers` |
    | `ratio_espacio_negativo` / `ratio_nitidez` | `var_laplaciano_figura`, `var_laplaciano_fondo` |
    | `d_tercios` / `d_centro` / `patron_dominante` | `coord_centroide`, `coord_p_cercano`, `margen_patron` |

    Citar `confianza`, `score_segmentacion`, `relleno_bbox` o `segmentacion_fiable` **no** es
    violación: son la justificación del cierre, no la métrica.
  - **Controles del crítico:**
    - afirmaciones en el orden de dimensiones del arbitraje (criterio de éxito);
    - una salvedad por cada métrica cerrada;
    - etiqueta igual a la del orquestador;
    - síntesis sin cifras fuera de las citas;
    - `campos_citados` coherente con lo citado.
- **Métricas:**
  - por autor (4 especialistas y crítico): nº de citas, % exactas, truncadas, alteradas y citas a
    cerradas;
  - cumplimiento de cada control del crítico;
  - ejecuciones que fallan en el primer intento.
- **Una ejecución que falla se repite y el fallo se contabiliza.**

### 6.4 E4: reproducibilidad y coste → §9.4

- **Determinismo de la capa Python (sin API, criterio a):** ejecutar los engines 3 veces sobre las fotos
  de repetición, en el mismo proceso y en procesos distintos, y comparar los informes bit a bit. Separa
  la variabilidad de las herramientas de la del LLM.
- **Fidelidad de las copias (tanda de prueba, criterios b y c):**
  - se recalcula cada herramienta y se compara campo a campo con el `informe` del JSON (160 informes).
    En el Agente 3 son 10 campos: uno booleano y dos que pueden ser `null`;
  - se recalcula el arbitraje y se compara con el que transcribe el crítico (40 críticas).
- **Repeticiones del sistema completo (descriptivo):**
  - Las 5 fotos de repetición se analizan 3 veces, en una tanda aparte.
  - La lectura visual se reutiliza de la primera repetición, así que la variabilidad medida es la de
    los especialistas y el crítico.
  - Se informa: informes transcritos idénticos a la herramienta; arbitraje y etiqueta iguales en las 3
    repeticiones; qué parte de las citas se repite.
- **Coste y latencia (tanda de prueba, descriptivo):**
  - tiempo total y por agente;
  - tokens de entrada, salida y razonamiento por modelo;
  - USD por análisis con las tarifas oficiales consultadas en una fecha concreta [pendiente: consultarlas
    antes de lanzar];
  - VRAM máxima y **RAM máxima del proceso** (SAM corre en CPU y no aparece en la VRAM);
  - fallos y reintentos.

  La primera ejecución de cada proceso incluye la carga de los modelos y se excluye de la latencia.
- **Presupuesto:** (40 + 15) análisis × ~0,13 USD ≈ **7,4 USD** sin descontar la caché.

### 6.5 Casos ilustrativos → §9.5

Tres casos del conjunto de prueba, **elegidos a partir de los resultados de E2 y E3**, no antes:

- un acierto típico;
- un cierre correcto o un falso negativo por la cobertura de COCO;
- un error del LLM como transcriptor, si lo hay.

Se muestran las imágenes de verificación del sistema y un extracto de la crítica con sus citas.

### 6.6 Resumen → §9.6

Tabla final: pregunta → criterio → resultado (sí / en parte / no) → cifra clave. Se genera leyendo
`experimentos/resultados/E*/metricas.json`, no se escribe a mano.

---

## 7. Instrumentos que hay que preparar

| Instrumento | Qué hace | Estado |
|---|---|---|
| `experimentos/config_experimentos.yaml` | Rutas (EVA externo, `data/`, `data_prueba/`), fotos de repetición, criterios de éxito y tarifas | [pendiente] |
| `experimentos/comun.py` | Utilidades: localizar la raíz, leer CSV con codificación y separador robustos, medir las condiciones, verificador de citas, IC de Wilson, comprobar la congelación | [pendiente] |
| `experimentos/ejecutar_lote.py` | Lanzador por lotes (abajo) | [pendiente] |
| Plantilla de contexto + aplicabilidad | `experimentos/plantilla_anotacion_prueba.py`: `plantilla` genera `anotaciones/anotacion_prueba.xlsx` (ciega, sobre `data_prueba/`); `exportar` la valida y escribe `anotaciones/anotacion_prueba.csv` | [verificado] |
| `experimentos/calibracion.py` | `medir`: variables de decisión de las 100 fotos de `data/` (`resultados/calibracion/medidas_calibracion.csv`, `entorno_calibracion.json`) e imágenes de revisión sin cifras; `plantilla` / `exportar`: revisión de la segmentación y del protagonista de las fotos `yolo` (`anotaciones/segmentacion_calibracion.csv`) | [verificado en 3 fotos; falta ejecutarlo en Windows] |
| Revisión de segmentación/protagonista del conjunto de prueba | La misma `calibracion.py` con una opción `--conjunto prueba` que lea `data_prueba/` y escriba en ficheros propios (`resultados/prueba/`, `anotaciones/segmentacion_prueba.csv`) sin pisar los de calibración. Se añade antes de congelar y se ejecuta después | [pendiente] |

**Lanzador por lotes.** Para cada imagen de una lista:

1. Borrar `outputs/ejecucion/{stem}`.
2. Lanzar `kickoff`.
3. Medir el tiempo total y, con `task_callback`, el instante en que termina cada tarea. Con tareas
   asíncronas, la duración por tarea es aproximada.
4. Guardar los tokens por modelo, la VRAM y la RAM máximas, y los errores y reintentos.
   - Los tokens de `LecturaVisualTool` no pasan por CrewAI. Hay que guardarlos desde la herramienta
     (`usage_metadata`).
5. Copiar la carpeta a `outputs/experimentos/{tanda}/{rep}/{stem}/`, junto con un `registro.json`
   que incluya el commit.
6. Ser **reanudable**: si ya existe `registro.json`, saltar la imagen.

Este paso de copia hace falta porque cada ejecución se escribe en `outputs/ejecucion/{stem}`. Sin él,
las repeticiones se pisan entre sí.

`tracing=True` envía trazas y puede añadir latencia. Hay que desactivarlo durante las tandas y
declararlo.

**Estructura:**

```
experimentos/
├── config_experimentos.yaml
├── congelacion.json              # generado al congelar
├── manifiesto_calibracion.csv / manifiesto_prueba.csv
├── anotaciones/                  # contexto, aplicabilidad y segmentación del conjunto de prueba
├── comun.py
├── ejecutar_lote.py
├── 00_preparacion.ipynb          # manifiestos, plantillas, validación de anotaciones, congelación
├── 01_calibracion.ipynb          # apoyo al cap. 6: distribuciones con el umbral y tabla de constantes
├── E1_clasificador.ipynb
├── E2_aplicabilidad.ipynb
├── E3_citas.ipynb
├── E4_reproducibilidad_coste.ipynb
├── E5_casos_y_resumen.ipynb
└── resultados/{E1,E2,E3,E4,E5}/
```

---

## 8. Límites de la validez (§8.6)

- **Anotador único.** El contexto y la aplicabilidad los anota el autor (salvo el segundo anotador
  opcional del §5.3).
- **Tamaño del conjunto de prueba.** Con 40 imágenes, los intervalos de confianza son anchos; por eso los
  criterios son tolerancias y no «cero errores».
- **La convergencia no tiene veredicto propio.** Solo hay 4 fotos con perspectiva en el conjunto de
  prueba, por debajo del mínimo de 5 aperturas para juzgar una condición por separado (§6.0). Cuenta en
  el resultado global y sus cifras se informan, pero es la condición más débil (48 % de precisión de
  apertura en la primera campaña) y su evidencia es escasa.
- **La segmentación fiable no se pudo calibrar.** En el conjunto de calibración no hubo ninguna
  segmentación `mal` (59 `bien`, 1 `parcial`), así que `SAM_SCORE_MIN` y `RELLENO_MIN/MAX` quedan como
  «comprobados, no calibrados» (regla 3 del §4.1). Su eficacia solo se observa en el conjunto de prueba.
- **Tres umbrales con sensibilidad baja en calibración (§4.2).** Horizonte 53 %, convergencia 29 % y
  nitidez 21 %: sus variables no separan bien las fotos en las que la regla aplica de las que no, y la
  regla del §4.1 prima los falsos negativos. Esas condiciones callan a menudo cuando podrían hablar.
- **Desviación declarada en el horizonte (§4.2).** El ángulo máximo se dejó en 10° porque el corte que
  daba la regla (0°) anulaba la métrica.
- **La caja de YOLO no siempre es el protagonista.** 11 de las 60 fotos `yolo` de calibración; ningún
  umbral de los calibrados lo corrige, y abre composición (y a veces espacio y nitidez) en esas fotos.
- **Selección intencional.** Los dos conjuntos de Unsplash contienen ejemplos claros de cada categoría y
  no representan las fotos de un usuario cualquiera.
- **EVA-other es solo una aproximación** a un conjunto fuera de distribución.
- **El umbral del clasificador se eligió con EVA-test.** Se controla con EVA-val.
- **LLM no determinista.** Hay una sola ejecución por imagen; la variabilidad solo se mide en las fotos de
  repetición y con la lectura visual fija.
- **Una foto derivada.** P2 se giró a propósito porque no apareció ningún horizonte inclinado natural.
- **Una sola máquina.** Los resultados bit a bit no son portables a otro hardware (§2.3).
- **Coste aproximado.** Se calcula con las tarifas de una fecha y sin descontar la caché, así que es una
  cota superior.

---

## 9. Congelación antes de ejecutar

Se guarda en `experimentos/congelacion.json`:

- [ ] Commit con el árbol limpio, con los umbrales ya calibrados y sin la etiqueta «PROVISIONAL».
- [ ] MD5 de:
  - `models/clasificador_contexto/*`;
  - `models/yolo/yolo11m.pt` (`2c2bcbb54c3829b20d31a32524ee0f03`);
  - `models/sam/sam2.1_b.pt` (`e57666d5a782119e02af8f0de6713e32`);
  - `parametros/*.json` y `config/*.yaml`;
  - `manifiesto_calibracion.csv` y `manifiesto_prueba.csv`.
- [ ] Valores de todas las constantes del §4.
- [ ] Modelos de Gemini, `FLAG_CRITICO_VE_IMAGEN` y la versión del prompt de la lectura visual.
- [ ] Máquina, GPU, CUDA, Python y versiones de los paquetes (`torch 2.6.0+cu124`,
      `ultralytics 8.4.102`…).
- [ ] Criterios de éxito del §6.0 escritos en `config_experimentos.yaml`.
- [ ] Conjunto de prueba cerrado y anotaciones de contexto y aplicabilidad hechas.

---

## 10. Orden de trabajo

1. ~~Elegir y descargar el conjunto de prueba en `data_prueba/` y crear el manifiesto, **sin pasar
   ninguna foto por el sistema**.~~ Hecho el 2026-11-05.
2. Anotar contexto y aplicabilidad del conjunto de prueba a ciegas.
3. Calibrar sobre el conjunto de calibración (§4): actualizar el código, los comentarios, `agents.yaml` y
   la Figura 28, y ejecutar `tests/`.
4. Preparar los instrumentos (§7) y congelar (§9).
5. Sin API: primero anotar la segmentación y el protagonista de las fotos `yolo` del conjunto de prueba
   (`calibracion.py --conjunto prueba`, sin abrir las medidas); después E1, E2 y el determinismo de E4.
6. Con API, en la máquina Windows: tanda de prueba y después las repeticiones.
7. E3, E4, casos ilustrativos y resumen. Actualizar la memoria (§11).

---

## 11. Cambios en la memoria

| Dónde | Qué cambia |
|---|---|
| 4.x Tecnologías | Añadir SAM 2.1 base, en CPU por reproducibilidad, con su cita (Ravi et al., 2024) |
| 6.x Agente 3 | Máscara con SAM; condición de segmentación fiable (score y relleno); informe de 10 campos. Sustituir la **Figura 28** por `docs/diagramas/espacio_aislamiento_pipeline.png` |
| 6.x Umbrales | Distribuciones y valores calibrados sobre el conjunto de calibración |
| 8.3 Conjuntos (Tablas 12 y 13) | Conjunto de calibración (`data/`, 100 fotos) y conjunto de prueba (`data_prueba/`, 40 fotos), con su composición (§5.1) y la foto derivada P2 |
| 8.4 Instrumentos | Añadir la RAM máxima al registro |
| 8.5 Protocolos (Tabla 14) | Sustituir por la tabla del §6.0, con «≥» / «≤», y una frase que declare la revisión de los criterios antes de abrir el conjunto de prueba |
| 8.5 Guía de anotación (Tabla 15) | Horizonte con la definición amplia como principal y la estrecha como secundaria |
| 8.6 Límites | Sustituir «umbrales ajustados sobre el mismo corpus» por «tamaño del conjunto de prueba», «una foto derivada» y «una sola máquina» |
| 9 | Todos los resultados sobre el conjunto de prueba; precisión de apertura en E2; regenerar las tablas y las Figuras 35 a 37. Tabla 24 con «sí / en parte / no» |
| 9.7 y 10 | Quitar del trabajo futuro la «condición de relleno de la máscara» y la «segmentación de propósito general» |

**Por qué SAM y no GrabCut.** Basta una frase en el apartado del Agente 3 (o en 4.x), como decisión de
diseño:

> Para segmentar el sujeto se usa SAM 2.1 [X] en lugar de un método clásico como GrabCut porque, además
> de la máscara, estima su propia calidad; esa estimación permite cerrar las métricas cuando la
> segmentación no es fiable, mientras que GrabCut siempre devuelve una máscara, aunque sea vacía o
> errónea, sin indicar si lo es.

La frase justifica la elección por una propiedad de los métodos, no por un experimento. Por eso no
debe afirmar que SAM «segmenta mejor» en general.
