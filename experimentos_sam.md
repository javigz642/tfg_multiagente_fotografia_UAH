# Revalidación tras sustituir GrabCut por SAM 2.1 (campaña 2)

Diseño experimental para evaluar el sistema después de cambiar el segmentador del especialista de
espacio y aislamiento (Agente 3). **No contiene resultados**: fija qué se mide, sobre qué, con qué
instrumentos y con qué criterio, **antes** de ejecutar, igual que el capítulo 8.

- **Fecha del diseño:** 2026-11-05. **Rama:** `agente3-sam`.
- **Relación con lo anterior:** la campaña 1 (ejecutada el 2026-09-28, capítulos 8 y 9 de la memoria,
  plan en `experimentos.md`) evaluó el sistema con GrabCut. Este documento solo cambia lo que el nuevo
  segmentador obliga a cambiar; todo lo demás se hereda tal cual.
- **Convención:** **[verificado]** comprobado en el código o ejecutando algo; **[pendiente]** no existe
  todavía; **[decisión]** lo tiene que fijar el autor antes de ejecutar.

---

## 1. Qué ha cambiado en el sistema

| Pieza | Campaña 1 (GrabCut) | Campaña 2 (SAM) |
|---|---|---|
| Máscara con `sujeto_discreto = True` | `cv2.grabCut`, 5 iteraciones, `setRNGSeed(0)` | SAM 2.1 base (`models/sam/sam2.1_b.pt`) con el rect como prompt, `DEVICE_SAM = "cpu"` |
| Máscara con `sujeto_discreto = False` | Otsu sobre el mapa de saliencia | Sin cambios |
| Gates de espacio y nitidez | Identidad (+ textura en nitidez) | Identidad **y segmentación fiable** (+ textura en nitidez) |
| Gate nuevo | — | `segmentacion_fiable`: `score_segmentacion ≥ SAM_SCORE_MIN` y `RELLENO_MIN ≤ relleno_bbox ≤ RELLENO_MAX` (0,90 / 0,20 / 0,95, **provisionales**) |
| `InformeEspacioAislamiento` | 7 campos | 10 campos (`score_segmentacion`, `relleno_bbox`, `segmentacion_fiable`) |
| Prompt del Agente 3 | 2 motivos de cierre | 3 motivos; salvedad propia cuando la segmentación no es fiable |

El fallo que motiva el cambio está documentado en la campaña 1: en la Tabla 19, los falsos positivos de
espacio y nitidez se deben a detecciones que no son el protagonista **o a segmentaciones fallidas**,
incluida una máscara vacía que abría las dos métricas. Con la definición amplia (segmentación parcial =
la regla no aplica) suben a 31 y 30.

## 2. Qué experimentos hay que repetir

| Experimento de la campaña 1 | ¿Le afecta el cambio? | Qué se hace |
|---|---|---|
| E1 Clasificador de contexto (§9.1) | No: el Nivel 1 no cambia [verificado] | **Se hereda** sin repetir |
| E2 Aplicabilidad: cromática, horizonte, convergencia, composición | No: sus engines no usan la máscara [verificado] | **Se heredan** |
| E2 Aplicabilidad: espacio y nitidez | Sí: cambian la máscara y la definición del gate | **Se repite** (§5) |
| E3 Fidelidad de citas (§9.3) | Sí: cambian el informe, el prompt del Agente 3 y, con ellos, la entrada del crítico | **Se repite la tanda T100 completa** (§6) |
| E4 Coste, latencia y reproducibilidad (§9.4) | Sí: SAM añade cómputo en CPU y el informe es más largo | **Se repite** (§7) |
| Experimento `aplicabilidad_vlm` (línea 10.4) | Solo si cambia alguna decisión de gate de espacio o nitidez | Comprobar en la tanda congelada; si no cambia ninguna, se hereda |
| — | — | **Experimento nuevo E-SEG**: calidad de la segmentación (§4) |

La tanda con API se repite entera aunque solo cambie un especialista: el sistema que se presenta en la
memoria tiene que ser el sistema evaluado, y el crítico recibe otro diagnóstico de espacio en las 100
imágenes.

## 3. Pregunta nueva

Se añade una fila a la Tabla 11:

| Aspecto | Pregunta | Objetivos | Resultados |
|---|---|---|---|
| Segmentación | ¿SAM segmenta al sujeto mejor que GrabCut, y su gate de calidad detecta las segmentaciones malas? | 2º | 9.x (nuevo) |

---

## 4. E-SEG: calidad de la segmentación (experimento nuevo)

- **Pregunta:** la de §3. Es una comparación **pareada**: mismas imágenes, misma caja y mismo espacio de
  trabajo de 1024 px. Lo único que cambia es el segmentador.
- **Conjunto:** las imágenes del corpus de Unsplash con `fuente = "yolo"`, que son las únicas que llegan
  a la vía de segmentación con identidad. Hay que fijar la lista en la tanda congelada **[pendiente]**:
  la campaña 1 contó 60 (en Windows/CUDA) y el desarrollo en Mac cuenta 61, así que alguna imagen está
  pegada a `CONF_YOLO_THRESHOLD = 0,75` y hay que identificarla.
- **No usa API.** Solo los engines.

### 4.1 Instrumento: anotación A3, segmentación a ciegas

La anotación de segmentación de la campaña 1 se hizo viendo solo la máscara de GrabCut. Para comparar
sin sesgo, **se anotan de nuevo los dos métodos, en la misma sesión y sin saber cuál es cuál**:

1. Para cada imagen se generan las dos máscaras y se guardan en caché (`mascaras_grabcut.npz` y
   `mascaras_sam.npz`, en `outputs/`).
2. Una hoja de revisión (extensión de `experimentos/plantilla_revision.py`) muestra por imagen la
   original y dos paneles, **A** y **B**, con el contorno de cada método. El orden A/B se sortea por
   imagen con una semilla fija y la clave se guarda en un fichero aparte que el anotador no abre.
3. La hoja **no muestra** ni `score_segmentacion`, ni el relleno, ni ninguna métrica: ver la puntuación
   sesgaría el juicio que luego sirve para evaluarla.
4. Columnas por imagen:
   - `seg_A` y `seg_B`: `bien` / `parcial` / `mal`, con la misma escala que la campaña 1;
   - `preferencia`: `A` / `B` / `igual`;
   - `comentario`.

Guía (fijarla antes de anotar) **[decisión]**:

| Valor | El contorno… |
|---|---|
| `bien` | rodea al sujeto completo; lo que sobra o falta no cambia el área de forma apreciable |
| `parcial` | deja fuera, o mete dentro, una parte apreciable (un brazo, media cabeza de un animal vecino), pero el sujeto se reconoce |
| `mal` | rodea a otro objeto, está vacío, se queda con un fragmento pequeño o rellena la caja entera |

La anotación A2 de la campaña 1 (`protagonista`, `espacio`, `nitidez`) **se reutiliza**: se hizo a ciegas
y sin ver ninguna máscara, así que no depende del segmentador.

### 4.2 Procedimiento y métricas

- **E-SEG a, calidad.** Por método, el reparto `bien` / `parcial` / `mal` con su IC 95 % (Wilson).
  Comparación pareada con la prueba exacta de McNemar sobre `bien` frente a `no bien`, y sobre `mal`
  frente a `no mal`. Recuento de preferencias A/B (prueba de signos, sin contar los empates).
- **E-SEG b, máscaras degeneradas.** Por método, nº de máscaras vacías, con relleno < 0,20 y con
  relleno > 0,95. Es el modo de fallo que la campaña 1 no podía detectar.
- **E-SEG c, efecto en las métricas.** Por imagen, `Δ = SAM − GrabCut` de `ratio_espacio_negativo` y de
  `ratio_nitidez`. Se da la mediana y el máximo de `|Δ|`, el nº de cambios de signo del índice, y el nº
  de imágenes en las que el índice cambia de banda en la escala de lectura del prompt (doble, triple,
  nueve veces…). Es descriptivo: no hay verdad de referencia para el valor de la métrica, solo para la
  máscara.
- **E-SEG d, el gate de calidad.** ¿Separa `score_segmentacion` las máscaras malas de las buenas?
  - AUROC de `score_segmentacion` para `mal` frente a `bien`, y para `mal ∪ parcial` frente a `bien`;
    lo mismo con el relleno.
  - Matriz 2×2 de `segmentacion_fiable` (true/false) frente a la anotación, con las dos definiciones.
- **E-SEG e, coste.** Tiempo por imagen de cada método en CPU (mediana y p95), en la misma máquina.

### 4.3 Criterios de éxito (se fijan antes de ejecutar)

| Criterio | Tipo |
|---|---|
| Proporción de `bien` con SAM mayor que con GrabCut (McNemar exacto, α = 0,05) | Principal |
| 0 máscaras vacías o que rellenan la caja **con el gate de espacio abierto** | Obligatorio |
| El gate de calidad no rechaza ninguna máscara anotada `bien` | Secundario |
| AUROC del score para `mal` frente a `bien` ≥ 0,90 | Secundario **[decisión]** |

---

## 5. E2 revisado: aplicabilidad de espacio y nitidez

Mismo protocolo que §8.5.2 de la memoria: se ejecutan los engines (no se leen los JSON), se replica cada
decisión desde los campos publicados y se cruza con la anotación.

- **Definición de «la regla aplica» (Tabla 15), sin cambios**: espacio aplica si hay sujeto discreto y,
  si la fuente es YOLO, la segmentación no es mala. La diferencia es que la segmentación ahora es la de
  SAM, anotada en A3. Se mantienen las dos versiones, la estricta (`mal`) y la amplia (`mal ∪ parcial`).
- **Réplica:** las decisiones son ahora 3 condiciones por imagen en el Agente 3 (identidad,
  segmentación y textura). Se comprueba que la réplica coincide con el código en todas.
- **Tablas:**
  - Tabla 18 actualizada, con un motivo de cierre nuevo: «segmentación no fiable».
  - Tabla 19 actualizada para espacio y nitidez, **en dos columnas: campaña 1 (GrabCut) y campaña 2
    (SAM)**, sobre las mismas imágenes y la misma anotación A2, más la lista de imágenes cuyo veredicto
    cambia (FP→VN, VP→FN…).
- **Criterio: el mismo de la campaña 1, 0 falsos positivos por condición.** No se relaja: cambiar el
  criterio después de ver los resultados de la campaña 1 invalidaría la comparación.

## 6. E3 revisado: fidelidad de citas

Tanda **T100** completa con el sistema congelado de la campaña 2, con el mismo protocolo y los mismos
criterios (0 citas a cerradas; ≥ 95 % exactas; 100 % de críticas en orden). Cambios en el verificador:

- **Campos nuevos.** `score_segmentacion`, `relleno_bbox` y `segmentacion_fiable` son evidencia **del
  gate**, igual que `confianza`: citarlos **no** es citar una métrica cerrada. Los campos condicionados
  de espacio y nitidez siguen siendo `var_laplaciano_figura` y `var_laplaciano_fondo`.
- **Fidelidad de transcripción** con 10 campos, incluidos un booleano y dos que pueden ser `null`: hay
  que comprobar que el LLM no convierte `null` en `0` ni `false` en `"false"`.
- **Control nuevo del Agente 3: salvedad de segmentación.** Cuando `segmentacion_fiable = false` con
  identidad, la salvedad tiene que hablar del contorno y no de la escena (`tasks.yaml`, punto 5). Se
  comprueba que cita `[segmentacion_fiable = false]` y que no usa la plantilla de la escena («paisaje»,
  «fachada», «interés distribuido»). Es una heurística de texto y se declara como tal. Si en C2 no hay
  ningún caso con el gate cerrado por segmentación y la identidad abierta, el control queda **sin
  ejercitar** y se dice así, sin fabricar un caso.

## 7. E4 revisado: coste, latencia y reproducibilidad

- **Latencia y coste (T100):** mismo registro que la campaña 1, con dos añadidos:
  - el tiempo de `mascara_sam` por imagen, medido dentro del engine. Los especialistas corren en
    paralelo, así que el tiempo total no basta para atribuirle el coste a SAM;
  - la **memoria RAM máxima del proceso**: SAM corre en CPU, así que no aparece en la VRAM que medía la
    campaña 1.
- **E4a, determinismo de la capa Python (sin API).** Tres niveles, de menor a mayor exigencia:

  | Nivel | Qué se compara | Criterio |
  |---|---|---|
  | Mismo proceso | 3 análisis seguidos de las imágenes de R | Informe idéntico bit a bit (100 %) |
  | Procesos distintos | 3 procesos nuevos sobre R | Informe idéntico bit a bit (100 %) |
  | Entre máquinas | Windows/CUDA frente a Mac, las dos con `DEVICE_SAM = "cpu"` | **Descriptivo**: IoU de las máscaras y `|Δ|` de cada métrica. Tolerancia que se fija antes **[decisión]**: propuesta IoU ≥ 0,99 y `|Δ| < 0,01` |

  El tercer nivel es necesario porque `uv.lock` instala **torch 2.6.0+cu124 en Windows y torch 2.14.0 en
  macOS** [verificado en `uv.lock`]: aunque las dos usen CPU, no ejecutan el mismo código numérico. En
  desarrollo se midió que cambiar de versión de ultralytics altera del 0,1 al 0,3 % de los píxeles de la
  máscara.
- **E4c, reproducibilidad con LLM (R × 3):** las mismas 5 imágenes de la campaña 1 (`imagen62`, `9`,
  `35`, `86` y `50`) y el mismo criterio.

---

## 8. Calibración de los umbrales nuevos (capítulo 6)

Los umbrales del gate de calidad **se fijan sin mirar la anotación**, con el mismo método de huecos que
`VAR_LAPLACIANO_MIN`: se ordena la distribución de `score_segmentacion` (y del relleno) de las imágenes
`yolo` en escala logarítmica y se corta en un hueco limpio. La anotación A3 se usa **después**, solo
para evaluar. Así la evaluación de E-SEG d no queda contaminada por la calibración.

| Parámetro | Valor provisional | Qué hacer |
|---|---|---|
| `SAM_SCORE_MIN` | 0,90 | Método de huecos sobre el score; documentarlo en el capítulo 6 |
| `RELLENO_MIN` / `RELLENO_MAX` | 0,20 / 0,95 | Ídem sobre el relleno; revisar los objetos rectangulares que rozan el techo |
| `VAR_LAPLACIANO_MIN` | 50 | **Recalibrar**: se fijó con máscaras de GrabCut |
| `MARGEN_FRONTERA_RATIO` | 0,01 | Repetir la sensibilidad (la de la campaña 1 incluía las iteraciones de GrabCut, que ya no existen) |
| `DEVICE_SAM` | `cpu` | No se calibra; se justifica (§7) |
| Modelo | `sam2.1_b` | Opcional: ablación con `sam2.1_t`/`s`/`l` (tiempo frente a calidad en A3). No es necesaria para validar |

**Declaración obligatoria en §8.6.** Los valores provisionales se orientaron con una prueba exploratoria
sobre este mismo corpus (imágenes 5, 86 y 87). Aunque la calibración definitiva no use la anotación, el
autor ya había visto esas imágenes.

**Mejora opcional, recomendada:** un **conjunto de prueba nuevo C3** de 30 a 40 fotografías de Unsplash
con sujeto detectable por YOLO (unas 7 por contexto), elegidas después de congelar los umbrales y nunca
usadas para calibrar. E-SEG a y d sobre C3 dan la única estimación no optimista del gate de calidad.
Sin C3, esa limitación va en §8.6.

## 9. Amenazas nuevas (añadir a §8.6)

- **Umbrales provisionales orientados en C2** (§8).
- **Anotador único que sabe que hay dos métodos**, aunque no cuál es cada uno. El orden A/B aleatorio
  reduce el sesgo, pero no lo elimina si un método tiene un estilo de contorno reconocible.
- **Dependencia del entorno numérico:** la máscara depende de la versión de ultralytics y de torch (§7).
- **El gate de calidad solo existe en la vía SAM.** La vía de saliencia sigue sin medida de calidad;
  en C2 ya la cierra el gate de identidad, pero un sujeto de saliencia con identidad no está contemplado.
- **La puntuación de SAM mide la coherencia de la máscara con la caja, no si YOLO acertó con el
  protagonista.** Un objeto bien segmentado que no es el protagonista sigue siendo un falso positivo de
  composición y de espacio, y lo sigue midiendo `protagonista` en A2.

## 10. Congelación de la campaña 2

Añadir a la lista de congelación de la campaña 1 (`experimentos.md`, §11):

- [ ] Commit de `agente3-sam` (o de `main` tras fusionarla) con el árbol limpio.
- [ ] MD5 de `models/sam/sam2.1_b.pt` (`e57666d5a782119e02af8f0de6713e32`) y de `models/yolo/yolo11m.pt`.
- [ ] Valores finales de `SAM_SCORE_MIN`, `RELLENO_MIN`, `RELLENO_MAX`, `VAR_LAPLACIANO_MIN`,
      `MARGEN_FRONTERA_RATIO` y `DEVICE_SAM`, sin la etiqueta «provisional» en el código.
- [ ] Versiones de ultralytics y torch **de cada máquina** que ejecute algo.
- [ ] Lista fija de imágenes `yolo` de E-SEG y clave del orden A/B.
- [ ] Criterios de §4.3, §5, §6 y §7 escritos antes de lanzar.

## 11. Orden de trabajo

1. **Sin API, antes de mirar nada:** generar la caché de las dos máscaras y la hoja A/B ciega.
2. Anotar A3.
3. Calibrar los umbrales por huecos (sin la anotación); actualizar el código, quitar «provisional» y
   ejecutar `tests/`.
4. Congelar (§10).
5. Sin API: E-SEG (a–e), E2 de espacio y nitidez, E4a (los tres niveles).
6. Con API: T100 y R × 3 con el lanzador de la campaña 1.
7. E3 y E4 sobre las tandas nuevas.
8. Actualizar la memoria (§12).

## 12. Cambios en la memoria

| Dónde | Qué cambia |
|---|---|
| 2.x Estado del arte | GrabCut pasa a antecedente; se añade la segmentación guiada por prompt (SAM 2, Ravi et al., 2024 [X]) |
| 4.x Tecnologías y modelos | SAM 2.1 base en CPU, con la justificación de reproducibilidad |
| 6.x Agente 3 | Cálculo (SAM y relleno), condiciones de aplicabilidad (tercer gate), **Figura 28** (redibujar: la actual muestra GrabCut) y el número de campos de evidencia |
| 8 | Tabla 11 (pregunta nueva), Tabla 14 (protocolo E-SEG), Tabla 15 (la segmentación es la de SAM), anotación A3 en Instrumentos, amenazas de §9 |
| 9 | Apartado nuevo para E-SEG; Tablas 18-23 regeneradas; Figuras 35 y 36 regeneradas con la tanda nueva |
| 9.7 y 10 | «Condición de relleno de la máscara» y «segmentación de propósito general» dejan de ser trabajo futuro y pasan a aportaciones. Lo mismo en `guia_redaccion_9.7_cap10.md` y `redaccion_definitiva_9.7_cap10.md` |
