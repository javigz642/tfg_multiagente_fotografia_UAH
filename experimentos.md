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

| Pregunta | Qué comprueba | Objetivos (§1.4.2) | Resultados |
|---|---|---|---|
| **P1 Contexto** | ¿El clasificador asigna el contexto correcto y rechaza lo que no reconoce? | 1.º y 3.º | §9.1 |
| **P2 Aplicabilidad** | ¿Las condiciones de aplicabilidad cierran una métrica cuando su regla no aplica? | 2.º | §9.2 |
| **P3 Citas** | ¿Toda cifra del texto sale del informe, sin alterarse y sin citar métricas cerradas? ¿El crítico respeta el arbitraje? | 2.º y 3.º | §9.3 |
| **P4 Coste y reproducibilidad** | ¿Cuánto cuesta y tarda un análisis, y es reproducible? | 1.º | §9.4 |
| Interfaz | ¿Es usable? | 4.º | No se valida (no hay estudio con usuarios) |

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
| Cromática | `esquema_cromatico` | `ratio_pixeles_cromaticos ≥ RATIO_CROMATICO_MIN` | `RATIO_CROMATICO_MIN = 0.09` |
| Horizonte | `angulo_horizonte` | Hay candidato, `longitud ≥ 0.13` y `\|ángulo\| ≤ 10°` | `LONGITUD_MIN_HORIZONTE`, `ANGULO_MAX_HORIZONTE` |
| Convergencia | `score_convergencia` | `n ≥ 2`, `score > score_nulo(n)` y `apertura ≥ 35°` | `MIN_LINEAS_FUGA`, `modelo_nulo_fuga.json`, `APERTURA_MIN_GRADOS` |
| Espacio | `ratio_espacio_negativo` | `fuente == "yolo"` **y** segmentación fiable | `FUENTE_CON_IDENTIDAD`, `SAM_SCORE_MIN`, `RELLENO_MIN`, `RELLENO_MAX` |
| Nitidez | `ratio_nitidez` | Lo mismo que espacio **y** `max(var_fig, var_fondo) ≥ VAR_LAPLACIANO_MIN` | Las anteriores + `VAR_LAPLACIANO_MIN` |
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
  idénticos bit a bit. Con torch 2.6.0 en las dos, YOLO detecta sujeto en 61 fotos de C2 en un Mac y
  en 60 en el equipo Windows [verificado]. La versión única es una garantía adicional; la regla sigue
  siendo una sola máquina.

---

## 3. Conjuntos de imágenes

| Conjunto | Dónde está | Tamaño | Papel |
|---|---|---|---|
| **EVA, prueba** | `Curso_CNN_FineTunning/clasificador_contexto/data/splits/test.csv` + `EVA_together/{image_id}.jpg` | 725 (136 animal, 143 arquitectura, 152 paisaje, 139 producto, 155 retrato) | E1 |
| **EVA, other** (`sort = 6`) | `.../splits/otro.csv` | 273 | E1: sustituto aproximado (*near-OOD*) de imágenes fuera de distribución |
| **EVA, validación** | `.../splits/val.csv` | 725 | E1: control del umbral de rechazo |
| **C2, calibración** | `data/imagen{1..100}.jpg` | 100 fotos de Unsplash | **Solo para ajustar umbrales** (§4) y para la reproducibilidad |
| **C3, prueba** | `data_c3/test{1..40}.jpg` | 40 fotos de Unsplash no vistas | **Resultados de E1 (parte externa), E2, E3 y E4** (§5) |
| **R, reproducibilidad** | `imagen62` (animal), `imagen9` (arquitectura), `imagen35` (paisaje), `imagen86` (producto), `imagen50` (retrato) de C2 | 5 × 3 repeticiones | E4 |

**Por qué hay dos conjuntos de Unsplash.** Si se calibra y se mide sobre las mismas fotos, las tasas de
error salen optimistas. C2 se usa para ajustar los umbrales y C3, que nadie ha mirado al calibrar, para
medir. R se toma de C2 porque la reproducibilidad no depende de la calibración.

**C2.** Son 100 fotos con selección intencional y estratificada: 18 por categoría y 10 de «otro», con
subtipos pensados para abrir y cerrar cada condición (Tabla 13 de la memoria). Tienen entre 2,07 y
25,96 MP. Falta **versionar el manifiesto** `experimentos/manifiesto_C2.csv`, con nombre, MD5,
dimensiones, URL y autor [pendiente].

**EVA.** No se copian imágenes al repositorio: la ruta va en la configuración. Los pesos del sistema son
idénticos al checkpoint `fase4_fine_tuning_ligero.pt` del proyecto externo (MD5
`20ce2c480390ccf0a06f3e67bb1225ca`) [verificado]. E1 sobre EVA evalúa, por tanto, exactamente el modelo
desplegado.

---

## 4. Calibración sobre C2

Se hace **antes** de abrir C3, con el método de huecos: se ordena la variable de decisión y se busca
el salto limpio entre las imágenes que deben abrir y las que deben cerrar. Ante dos huecos igual de
limpios se toma el más estricto, porque un falso positivo rompe la verificabilidad y un falso negativo
solo hace callar al sistema. El capítulo 6 muestra la distribución que justifica cada umbral.

| Componente | Parámetro (valor actual) | Cómo se fija |
|---|---|---|
| Espacio | `SAM_SCORE_MIN = 0.90` (provisional) | Huecos sobre el score en las fotos `yolo` |
| Espacio | `RELLENO_MIN = 0.20` / `RELLENO_MAX = 0.95` (provisionales) | Huecos sobre el relleno; revisar los objetos rectangulares que rozan el techo |
| Espacio | `VAR_LAPLACIANO_MIN = 50` (provisional) | Huecos sobre `var_max` en escala log, en las fotos `yolo` |
| Espacio | `MARGEN_FRONTERA_RATIO = 0.01` | Solo documentar la sensibilidad del índice de nitidez |
| Luz y tono | `RATIO_CROMATICO_MIN = 0.09` | Histograma del ratio con el corte; hay casos pegados a él (0,086; 0,093; 0,099) |
| Luz y tono | `PESO_MIN_MATIZ = 0.05` | Ablación fusión/filtro con el valor actual |
| Luz y tono | `UMBRAL_FUSION_MATIZ = 15`, `SAT_MIN` / `VAL_MIN = 0.15` | Solo documentar |
| Líneas | `LONGITUD_MIN_HORIZONTE = 0.13`, `ANGULO_MAX_HORIZONTE = 10` | Rectángulo de aceptación sobre C2 |
| Líneas | `APERTURA_MIN_GRADOS = 35` | Histograma de apertura sobre C2 |
| Líneas | `modelo_nulo_fuga.json` | No depende del corpus; solo documentar |
| Percepción | `CONF_YOLO_THRESHOLD = 0.75` | Barrido de identidad según el umbral; documentar |
| Percepción | `BLOB_AREA_MIN/MAX_RATIO` | No se activan en C2: declararlos como no ejercitados |
| Composición | Escalas de lectura del prompt (margen 0,01/0,15; `d_equilibrio` 0,03/0,12) | Recalcular los percentiles sobre C2 y actualizar `agents.yaml` |
| Clasificador | `umbral_otro = 0.85` | No se toca; se controla con la partición de validación (E1) |
| Nivel 3 | W (versión 1.1) | Hipótesis de diseño: no se calibra |

Para calibrar el Agente 3 hace falta revisar la segmentación (`bien` / `parcial` / `mal`) de las fotos
`yolo` de C2 sobre las imágenes de verificación del sistema actual.

Después de calibrar:

1. quitar la etiqueta «PROVISIONAL» del código y actualizar los comentarios que citan cifras;
2. actualizar la Figura 28, que muestra 0,90 / 0,20 / 0,95 (`docs/diagramas/espacio_aislamiento_pipeline.html`);
3. ejecutar todos los scripts de `tests/`;
4. hacer commit y congelar (§9).

---

## 5. C3: el conjunto de prueba

### 5.1 Composición (40 fotos)

La lógica es la misma que la de C2: 7 fotos por categoría y 5 para «otro». En cada categoría hay
casos en los que cada condición **debería abrirse** y casos en los que **debería cerrarse**. Además se
refuerzan los dos puntos débiles de la percepción, que son la elección del sujeto y la segmentación
difícil.

**Arquitectura (7)**

| # | Foto | Qué pone a prueba |
|---|---|---|
| A1 | Calle o pasillo con un punto de fuga claro | Convergencia abre |
| A2 | Fachada frontal plana, sin perspectiva | Convergencia cierra; sus horizontales no deben abrir el horizonte |
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
| O4 | Textura que llena todo el encuadre (hojas, arena, un patrón) | Rama `sin_sujeto_claro`, que **no se da nunca en C2** |
| O5 | Escena sin dominante (persona y animal a la par, un mercado) | Contexto ambiguo |

**Comprobación de cobertura.** Antes de cerrar la lista, cuenta cuántas fotos deberían abrir y cuántas
deberían cerrar cada condición. Se buscan **al menos 4 de cada lado** en las condiciones cromática, de
horizonte, de convergencia, de identidad y de textura. Si una foto no cumple lo que se esperaba de
ella, **se queda igual**: el subtipo es un objetivo de selección, no una etiqueta. La etiqueta real la
da la anotación ciega.

### 5.2 Reglas de selección

- **Fuente:** Unsplash, con la misma licencia que C2 y, si es posible, de fotógrafos distintos.
- **Una búsqueda por fila de la tabla:** se coge la **primera foto que cumpla claramente la
  descripción**, no la que «parece que va a salir bien».
- **Ninguna foto de C3 pasa por el sistema, la app ni los tests hasta congelar.**
  - Se guardan en `data_c3/`, no en `data/`, porque los tests recorren `data/`.
  - `data_c3/` está en `.gitignore`.
- **No se sustituye ninguna foto después de ver resultados.** Si una resulta inválida (corrupta o
  duplicada de C2), se sustituye con la misma regla y se declara.
- **Variedad** de orientación, y resolución parecida a la de C2 (de 2 a 26 MP).
- **Manifiesto** `experimentos/manifiesto_C3.csv` con: nombre (`test1`…`test40`), fila de la tabla,
  URL, autor, MD5, dimensiones y fecha de descarga.

### 5.3 Anotación de C3

El autor es el único anotador y sigue la guía del §6.2.

1. **A1 (contexto) y A2 (aplicabilidad), a ciegas**, antes de ejecutar nada.
2. **Segmentación y protagonista**, solo en las fotos con `fuente = yolo`, después de A2.
   - Segmentación: `bien` / `parcial` / `mal`.
   - Protagonista: `si` / `no`.
   - Se revisan sobre las imágenes de verificación del Agente 3. Esta anotación no puede ser ciega a la
     máscara; A2 sí lo es.

---

## 6. Experimentos

Criterios de éxito (Tabla 14 de la memoria). Se fijan **antes** de ejecutar y no se relajan después:

| Experimento | Conjunto | Qué se mide | Criterio de éxito |
|---|---|---|---|
| **E1** Clasificador | EVA (prueba, validación, other) y C3 | Exactitud y F1 por clase; cobertura y exactitud con el umbral de 0,85; % enviado a «otro» | F1 ≥ 0,75 en cada clase de EVA prueba; exactitud sobre las aceptadas de C3 ≥ 0,85; ≤ 20 % de C3 enviado a «otro» |
| **E2** Aplicabilidad | C3 anotado | Matriz abierta/cerrada × aplica/no aplica por condición | 0 falsos positivos por condición |
| **E3** Citas | T40 (C3 analizado) | Clase de cada cita; controles del crítico; fidelidad de la transcripción | 0 citas a métricas cerradas; ≥ 95 % de citas exactas; 100 % de críticas con las afirmaciones en orden |
| **E4** Coste y reproducibilidad | T40 y R × 3 | Latencia, tokens, coste, memoria y fallos; identidad entre repeticiones | 100 % de informes transcritos idénticos a la herramienta y arbitraje idéntico en las 3 repeticiones |

> La memoria escribe los criterios de E1 con «>» y «<» en la Tabla 14 y con «≥» y «≤» en la Tabla 24.
> Hay que unificarlos (se recomienda «≥» y «≤»).

### 6.1 E1: clasificador de contexto (P1) → §9.1

- **No usa LLM.** Se hace entero en un notebook en pocos minutos de GPU.
- **Procedimiento:**
  1. **EVA prueba, sin umbral:** exactitud, F1 por clase, matriz de confusión e IC del 95 %.
     - Transform: Resize 256 → CenterCrop 224 → normalización ImageNet.
     - Sirve también de comprobación de integridad: debe dar 0,8510.
  2. **EVA prueba + other, con 0,85:** cobertura, exactitud sobre las aceptadas, errores capturados,
     aciertos perdidos y % de other enviado a «otro».
  3. **EVA validación + other:** la misma tabla, para comprobar que 0,85 no depende de haberse elegido
     con la partición de prueba.
  4. **C3:** exactitud frente a A1 con 6 etiquetas, exactitud sobre las aceptadas, % enviado a
     «otro», matriz de confusión 6×6 y confianza en C3 frente a EVA.
  5. **Grad-CAM:** un acierto y un error del par que más se confunde (producto → arquitectura).
- Los pasos 1 a 3 no dependen del segmentador ni de C3. Si ya hay predicciones guardadas con los mismos
  pesos (mismo MD5), no hace falta repetirlos.
- **Resultados:** `experimentos/resultados/E1/predicciones_{prueba,other,val,C3}.csv`, `metricas.json`,
  matrices de confusión y Grad-CAM.
- **Limitaciones:** el umbral se eligió con la partición de prueba; other es *near-OOD*; C3 lo anota un
  único anotador.

### 6.2 E2: condiciones de aplicabilidad (P2) → §9.2

- **No usa API.** Se ejecutan las cuatro herramientas sobre C3 y se cruza cada decisión con la anotación.
- **Guía de anotación A2.** Se responde `si` / `no` / `dudoso` a cada pregunta (Tabla 15):

  | Condición | Métricas | La regla aplica si… |
  |---|---|---|
  | Cromática | `esquema_cromatico` | La imagen tiene color analizable (no es B/N, nocturna ni desaturada) |
  | Horizonte | `angulo_horizonte` | Hay un horizonte natural legible |
  | Convergencia | `score_convergencia` | La profundidad se organiza por perspectiva lineal |
  | Espacio | `ratio_espacio_negativo` | Hay sujeto discreto y, si la fuente es YOLO, la segmentación no es `mal`. En la definición amplia, además, tampoco puede ser `parcial`: tiene que ser `bien` |
  | Nitidez | `ratio_nitidez` | Lo mismo que espacio |
  | Composición | `d_tercios`, `d_centro`, `patron_dominante` | Hay sujeto discreto y, si la fuente es YOLO, el objeto detectado es el protagonista |

  Para el contexto, A1 usa animal / arquitectura / paisaje / producto / retrato / otro.

- **Definición de error:**
  - **Falso positivo** (grave): la métrica está abierta y la regla no aplica.
  - **Falso negativo** (tolerable): la métrica está cerrada y la regla aplica.
  - Las imágenes `dudoso` se excluyen de la matriz y se cuentan aparte.
- **Procedimiento:**
  1. Ejecutar los 4 engines sobre las 40 fotos. Extraer por imagen la `confianza`, el motivo de cierre
     y la variable de decisión de cada métrica.
  2. Replicar cada decisión a partir de los campos del informe y comprobar con `assert` que coincide
     con la del código (40 × 6 = 240 decisiones).
  3. **Tabla 1:** % de fotos con cada métrica abierta (con IC), reparto de `fuente`
     (`yolo` / `saliencia` / `sin_sujeto_claro`), cobertura por dimensión y motivos de cierre.
  4. **Tabla 2:** por condición, VP, FP, FN, VN y dudosas; tasa de FP con IC de Wilson al 95 %; tasa de
     FN; lista de imágenes de cada error.
- **Resultados:** `experimentos/resultados/E2/gates_C3.csv` (una fila por imagen y métrica), `tabla1.csv`,
  `tabla2.csv`, `fp_fn.json` y hojas de revisión de cada falso positivo.

### 6.3 E3: fidelidad de las citas (P3) → §9.3

- **Tanda T40:** las 40 fotos de C3, una vez cada una, con lectura visual. Las ejecuciones las hace el
  lanzador (§7) y el análisis un notebook.
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
    - arbitraje transcrito igual al recalculado;
    - afirmaciones en el orden de dimensiones del arbitraje;
    - una salvedad por cada métrica cerrada;
    - etiqueta igual a la del orquestador;
    - síntesis sin cifras fuera de las citas;
    - `campos_citados` coherente con lo citado.
  - **Fidelidad de transcripción:** se recalcula cada herramienta y se compara campo a campo con el
    `informe` del JSON. En el Agente 3 son 10 campos: uno booleano y dos que pueden ser `null`.
- **Métricas:**
  - por autor (4 especialistas y crítico): nº de citas, % exactas, truncadas, alteradas y citas a
    cerradas;
  - cumplimiento de cada control del crítico;
  - informes transcritos fieles;
  - ejecuciones que fallan en el primer intento.
- **Una ejecución que falla se repite y el fallo se contabiliza.**

### 6.4 E4: coste, latencia y reproducibilidad (P4) → §9.4

- **Coste y latencia (T40):**
  - tiempo total y por agente;
  - tokens de entrada, salida y razonamiento por modelo;
  - USD por análisis con las tarifas oficiales consultadas en una fecha concreta [pendiente: consultarlas
    antes de lanzar];
  - VRAM máxima y **RAM máxima del proceso** (SAM corre en CPU y no aparece en la VRAM);
  - fallos y reintentos.

  La primera ejecución de cada proceso incluye la carga de los modelos y se excluye de la latencia.
  Coste y latencia son descriptivos y no tienen umbral.
- **Reproducibilidad (R × 3):**
  - Las 5 fotos de R se analizan 3 veces, en una tanda aparte.
  - La lectura visual se reutiliza de la primera repetición, así que la variabilidad medida es la de
    los especialistas y el crítico.
  - Se mide: informes transcritos idénticos a la herramienta; arbitraje y etiqueta iguales en las 3
    repeticiones; qué parte de las citas se repite.
- **Determinismo de la capa Python (sin API):** ejecutar los engines 3 veces sobre R, en el mismo
  proceso y en procesos distintos, y comparar los informes bit a bit. Separa la variabilidad de las
  herramientas de la del LLM.
- **Presupuesto:** (40 + 15) análisis × ~0,13 USD ≈ **7,4 USD** sin descontar la caché.

### 6.5 Casos ilustrativos → §9.5

Tres casos de C3, **elegidos a partir de los resultados de E2 y E3**, no antes:

- un acierto típico;
- un cierre correcto o un falso negativo por la cobertura de COCO;
- un error del LLM como transcriptor, si lo hay.

Se muestran las imágenes de verificación del sistema y un extracto de la crítica con sus citas.

### 6.6 Resumen → §9.6

Tabla final: pregunta → criterio → resultado (sí / no / en parte) → cifra clave. Se genera leyendo
`experimentos/resultados/E*/metricas.json`, no se escribe a mano.

---

## 7. Instrumentos que hay que preparar

| Instrumento | Qué hace | Estado |
|---|---|---|
| `experimentos/config_experimentos.yaml` | Rutas (EVA externo, `data/`, `data_c3/`), lista R, criterios de éxito y tarifas | [pendiente] |
| `experimentos/comun.py` | Utilidades: localizar la raíz, leer CSV con codificación y separador robustos, medir las condiciones, verificador de citas, IC de Wilson, comprobar la congelación | [pendiente] |
| `experimentos/ejecutar_lote.py` | Lanzador por lotes (abajo) | [pendiente] |
| Plantillas de anotación | CSV de A1, A2 y segmentación/protagonista, con la guía del §6.2 | [pendiente] |

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
├── manifiesto_C2.csv / manifiesto_C3.csv
├── anotaciones/                  # A1, A2 y segmentación de C3 (CSV utf-8, separador ',')
├── comun.py
├── ejecutar_lote.py
├── 00_preparacion.ipynb          # manifiestos, plantillas, validación de anotaciones, congelación
├── 01_calibracion_C2.ipynb       # apoyo al cap. 6: distribuciones con el umbral y tabla de constantes
├── E1_clasificador.ipynb
├── E2_aplicabilidad.ipynb
├── E3_citas.ipynb
├── E4_coste_reproducibilidad.ipynb
├── E5_casos_y_resumen.ipynb
└── resultados/{E1,E2,E3,E4,E5}/
```

---

## 8. Límites de la validez (§8.6)

- **Anotador único.** El contexto y la aplicabilidad los anota el autor.
- **Tamaño de C3.** Con 40 imágenes, los intervalos de confianza son anchos.
- **Selección intencional.** C2 y C3 contienen ejemplos claros de cada categoría y no representan las
  fotos de un usuario cualquiera.
- **other es solo una aproximación** a un conjunto fuera de distribución.
- **El umbral del clasificador se eligió con la partición de prueba de EVA.** Se controla con la de
  validación.
- **LLM no determinista.** Hay una sola ejecución por imagen; la variabilidad solo se mide en R y con la
  lectura visual fija.
- **Anotación acotada.** Solo se anota el horizonte natural, no las horizontales estructurales.
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
  - los manifiestos de C2 y C3.
- [ ] Valores de todas las constantes del §4.
- [ ] Modelos de Gemini, `FLAG_CRITICO_VE_IMAGEN` y la versión del prompt de la lectura visual.
- [ ] Máquina, GPU, CUDA, Python y versiones de los paquetes (`torch 2.6.0+cu124`,
      `ultralytics 8.4.102`…).
- [ ] Criterios de éxito escritos en `config_experimentos.yaml`.
- [ ] Lista de C3 cerrada y anotaciones A1 y A2 de C3 hechas.

---

## 10. Orden de trabajo

1. Elegir y descargar C3 en `data_c3/` y crear el manifiesto, **sin pasar ninguna foto por el sistema**.
2. Anotar A1 y A2 de C3 a ciegas.
3. Calibrar sobre C2 (§4): actualizar el código, los comentarios, `agents.yaml` y la Figura 28, y
   ejecutar `tests/`.
4. Preparar los instrumentos (§7) y congelar (§9).
5. Sin API: E1, E2 y el determinismo de E4. Anotar la segmentación y el protagonista de las fotos `yolo`
   de C3.
6. Con API, en la máquina Windows: T40 y después R × 3.
7. E3, E4, casos ilustrativos y resumen. Actualizar la memoria (§11).

---

## 11. Cambios en la memoria

| Dónde | Qué cambia |
|---|---|
| 4.x Tecnologías | Añadir SAM 2.1 base, en CPU por reproducibilidad, con su cita (Ravi et al., 2024) |
| 6.x Agente 3 | Máscara con SAM; condición de segmentación fiable (score y relleno); informe de 10 campos. Sustituir la **Figura 28** por `docs/diagramas/espacio_aislamiento_pipeline.png` |
| 6.x Umbrales | Distribuciones y valores calibrados sobre C2 |
| 8.3 Conjuntos (Tablas 12 y 13) | C2 como conjunto de calibración; C3 como conjunto de prueba, con su composición (§5.1) |
| 8.4 Instrumentos | Añadir la RAM máxima al registro |
| 8.5 Protocolos (Tabla 14) | Los experimentos se miden sobre C3; unificar «≥» / «≤» |
| 8.6 Límites | Sustituir «umbrales ajustados sobre el mismo corpus» por «tamaño de C3» y «una sola máquina» |
| 9 | Todos los resultados sobre C3; regenerar las tablas y las Figuras 35 a 37 |
| 9.7 y 10 | Quitar del trabajo futuro la «condición de relleno de la máscara» y la «segmentación de propósito general» |

**Por qué SAM y no GrabCut.** Basta una frase en el apartado del Agente 3 (o en 4.x), como decisión de
diseño:

> Para segmentar el sujeto se usa SAM 2.1 [X] en lugar de un método clásico como GrabCut porque, además
> de la máscara, estima su propia calidad; esa estimación permite cerrar las métricas cuando la
> segmentación no es fiable, mientras que GrabCut siempre devuelve una máscara, aunque sea vacía o
> errónea, sin indicar si lo es.

La frase justifica la elección por una propiedad de los métodos, no por un experimento. Por eso no
debe afirmar que SAM «segmenta mejor» en general.
