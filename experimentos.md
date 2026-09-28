# Experimentos de los capítulos 8 y 9: contexto y plan de reproducción

> **Estado actual y cómo continuar: §15 (versión acotada) y §14** (actualizado el 2026-09-27). Si
> retomas el trabajo en un chat nuevo, lee primero §15, después §14, y consulta el resto según lo
> necesites. **§15 manda sobre todo lo anterior**: el alcance se recortó el 2026-09-27 por falta de
> tiempo (una semana) y de presupuesto de API (25 €). Donde §1-§14 hablen de V20, de R = 10, de
> `horizontal_estructural`, de `vegetacion_dominante`, de E4a, de `01_calibracion_C2` o de los
> notebooks E5 y E9, vale lo que dice §15.

Documento de trabajo para preparar y rehacer la validación experimental del TFG. Recoge el análisis
del índice «Validación experimental y Resultados (cap. 8 y 9)», el estado real del sistema y de los
notebooks, y el plan para construir un notebook por experimento. Está pensado para que la siguiente
tarea (crear los `.ipynb`) no tenga que volver a revisar el proyecto.

- **Fecha del análisis:** 2026-09-26. **Commit de referencia:** `8e611ed` (rama `main`, árbol limpio).
- **Convención de fiabilidad** usada en todo el documento:
  - **[verificado]**: comprobado en el código, en los ficheros o ejecutando algo durante este análisis.
  - **[según notebook]**: lo dice una salida o un texto guardado en un notebook; puede estar desactualizado.
  - **[pendiente]**: no existe o no se ha encontrado; no se inventa nada.

---

## 1. Objetivo general de la validación

La validación no mide gusto ni calidad estética (§1.5 y §3.1 de la memoria), y no demuestra que el
vector W sea óptimo, porque W es una hipótesis de diseño (§4.6). Mide las **garantías de ingeniería**
que el sistema dice ofrecer:

1. **Contexto correcto (P1):** el clasificador asigna la etiqueta adecuada y rechaza lo que no reconoce.
2. **Cierres correctos (P2):** las condiciones de aplicabilidad (los *gates*) cierran una métrica cuando
   la regla no aplica, y sobre todo no la dejan abierta cuando no aplica.
3. **Cifras trazables (P3):** toda cifra del texto sale del informe, sin alterarse y sin citar métricas
   cerradas, y el crítico respeta el arbitraje.
4. **Coste acotado y reproducible (P4):** cuánto cuesta y cuánto tarda un análisis, y si se repite.

El capítulo 8 fija las preguntas, los conjuntos, los instrumentos, los protocolos y los criterios de éxito
**antes** de ejecutar. El capítulo 9 da los números. Cada sección del 9 empieza con «Según el protocolo 8.5.x…».

---

## 2. El sistema que se valida (resumen operativo)

### 2.1 Tubería

| Nivel | Pieza | Qué es determinista | Qué pasa por un LLM | Salida en disco |
|---|---|---|---|---|
| 1 | `ContextClassifierTool` (ResNet-18, `tools/cnn_context.py`) + `SharedPerceptionTool` (YOLO `yolo11m.pt` + saliencia espectral) | Todo el cálculo | El orquestador copia la etiqueta y la percepción en `SalidaOrquestador` | `outputs/ejecucion/{stem}/salida_orquestador.json` |
| 2 | 4 especialistas: composición espacial, líneas y dirección, luz y tono, espacio y aislamiento | Las métricas y sus `confianza` (herramientas Python) | El LLM **transcribe** el informe y redacta `diagnostico` con citas `[campo = valor]` | `salida_{composicion_espacial,lineas_direccion,luz_tono,espacio_aislamiento}.json` |
| 3 | Crítico: `PrioridadTool` (arbitraje por W) + `LecturaVisualTool` opcional | El arbitraje (`tools/prioridad_tool.py`) | Transcribe el arbitraje y redacta `CriticaCompositiva` | `salida_critica.json` |

Datos de configuración [verificado]:

- **LLM de los especialistas y el orquestador:** `MODEL=gemini/gemini-2.5-flash` (en `.env`).
- **LLM del crítico:** `MODELO_CRITICO = "gemini/gemini-2.5-pro"` (`crew.py`).
- **Lectura visual:** `gemini-2.5-flash`, temperatura 0.2, `VERSION_PROMPT = 1`, con caché en `outputs/lectura_visual/{stem}_lectura.json`.
- **Interruptor de la lectura visual:** `FLAG_CRITICO_VE_IMAGEN = True` (constante de módulo en `crew.py`).
- **Ejecución:** `Process.sequential` con las 4 tareas del Nivel 2 en `async_execution=True`, y `tracing=True`.
- **Punto de entrada:** `main.py` solo ejecuta una imagen fija (`data\imagen39.jpg`). No hay lanzador por lotes ni registro de tiempos o tokens.
- **Entorno:** crewai 1.15.13, ultralytics 8.4.102, torch 2.6.0+cu124, opencv-contrib 5.0.0.93, scikit-learn 1.9.0, pandas 3.0.5, grad-cam 1.5.5; GPU RTX 4060 Laptop con CUDA. `ipykernel` está instalado; `jupyter`, `nbformat` y `nbconvert` no.

**Hecho clave [verificado]:** el `informe` que aparece en cada `salida_*.json` **no es la salida directa
de la herramienta**. Es la copia que hace el LLM (`tasks.yaml`: «Transcribe su resultado con exactitud…»),
validada con `output_pydantic`. Lo mismo ocurre con el `arbitraje` dentro de `salida_critica.json`.
Por eso «informes idénticos» (E4) también prueba si la transcripción del LLM es fiel, y la
aplicabilidad (E2) debe calcularse con los *engines* directamente, no leyendo los JSON.

### 2.2 Métricas con condición de aplicabilidad (gates)

| Dimensión | Métrica (`MetricaConfianza`) | Gate (condición para `confianza = 1`) | Constantes |
|---|---|---|---|
| luz_tono | `media_L` | Ninguno (siempre 1) | — |
| luz_tono | `esquema_cromatico` | `ratio_pixeles_cromaticos ≥ RATIO_CROMATICO_MIN` | `RATIO_CROMATICO_MIN = 0.09` |
| lineas_direccion | `angulo_horizonte` | Existe candidato, `longitud ≥ 0.13` y `|ángulo| ≤ 10°` | `LONGITUD_MIN_HORIZONTE`, `ANGULO_MAX_HORIZONTE` |
| lineas_direccion | `score_convergencia` | `n ≥ 2`, `score > score_nulo(n)` (estricto) y `apertura ≥ 35°` | `MIN_LINEAS_FUGA`, `modelo_nulo_fuga.json`, `APERTURA_MIN_GRADOS` |
| espacio_aislamiento_sujeto | `ratio_espacio_negativo` | `fuente == "yolo"` | `FUENTE_CON_IDENTIDAD` |
| espacio_aislamiento_sujeto | `ratio_nitidez` | `fuente == "yolo"` y `max(var_fig, var_fondo) ≥ 50` | `VAR_LAPLACIANO_MIN = 50` |
| composicion_espacial | `d_tercios`, `d_centro`, `patron_dominante` | `fuente == "yolo"` (**un solo gate compartido por las tres**) | `FUENTE_CON_IDENTIDAD` |
| composicion_espacial | `d_equilibrio` | Ninguno (siempre 1) | — |

Aguas arriba de todos los gates de identidad está `CONF_YOLO_THRESHOLD = 0.75`
(`shared_perception_tool.py`): decide si `fuente = "yolo"`. Se elige la caja YOLO de **mayor
confianza**, no la más grande ni la más central. Hay 8 métricas con gate pero solo **6 decisiones de
gate independientes** (las 3 de composición comparten una). E2 debe contarlas así.

---

## 3. Conjuntos de datos (estado real)

| Conjunto | Dónde está | Tamaño | Estado |
|---|---|---|---|
| **C1**: prueba de EVA | `Documents/Curso_CNN_FineTunning/clasificador_contexto/data/splits/test.csv` + imágenes en `Documents/Curso_CNN_FineTunning/EVA_together/{image_id}.jpg` | 725 (136 animal, 143 arquitectura, 152 paisaje, 139 producto, 155 retrato) | [verificado] Existe y es legible |
| **O**: `other` de EVA (`sort=6`) | `.../splits/otro.csv` (misma carpeta de imágenes) | 273 | [verificado] Es el conjunto con el que se calculó el 64,8 % |
| **val** de EVA | `.../splits/val.csv` | 725 | [verificado] Útil para recalibrar el umbral sin tocar C1 (ver E1) |
| **C2**: corpus de desarrollo | `data/imagen{1..100}.jpg` (fuera de git) | 100 JPG RGB, de 2,07 a 25,96 MP (mediana 4,83 MP) | [verificado] **100 imágenes únicas** tras sustituir `imagen61.jpg` el 2026-09-26 (MD5 nuevo `7a522ec1bd26fb24df379e0b8226db9d`). Antes `imagen61` duplicaba a `imagen55` |
| **R**: subconjunto de reproducibilidad | 10 imágenes de C2 | 10 | [pendiente] Falta elegir las imágenes (propuesta en §6.4) |
| **V20**: imágenes con y sin lectura visual | 20 imágenes de C2 | 20 | [pendiente] Falta elegir las imágenes (propuesta en §6.3) |

Notas sobre C2:

- **No comparte imágenes con el corpus antiguo de 40** (`Curso_CNN_FineTunning/data/`) [verificado por MD5]. La única coincidencia era la antigua `imagen40`, que era la `imagen55` actual y la duplicada `imagen61`.
- **Faltan origen, licencia y criterio de selección** de las 100 imágenes [pendiente]: los necesita §8.3.
- `data/`, `outputs/` y `notebooks/` están en `.gitignore`. Para que C2 sea reproducible hay que **versionar un manifiesto** con nombre, MD5, dimensiones y origen de cada imagen.
- **La sustitución de `imagen61` deja datos antiguos** que hay que regenerar:
  - `outputs/notebooks_cache/espacio_aislamiento/percepcion.json` y `mascaras.npz`;
  - la fila 61 de `notebooks/revision_*.csv`;
  - cualquier salida guardada de los notebooks 01 a 05.

---

## 4. Valoración del índice propuesto

### 4.1 Lo que es coherente con el sistema actual

- Las cuatro preguntas corresponden a garantías que el código implementa (gates, citas `[campo = valor]`, arbitraje determinista, herramientas deterministas).
- La regla de reparto 8/9 y «un lugar en el 8 y otro en el 9 por pregunta» funciona. Encaja con la aclaración de §6: en el capítulo 6 va la distribución que justifica cada umbral; en §9.2, las tasas de apertura y los falsos positivos y negativos.
- Las cifras del clasificador que cita el índice coinciden con los artefactos externos: exactitud 0,8510, 27,4 % y 64,8 %, umbral 0,85 [verificado, §9].
- El umbral cromático del código es 0,09 [verificado], como dice la revisión cruzada (punto 3).
- Los pesos del sistema (`models/clasificador_contexto/pesos_fine_tuning_ligero.pt`) son **idénticos** al checkpoint `fase4_fine_tuning_ligero.pt` del proyecto externo (MD5 `20ce2c480390ccf0a06f3e67bb1225ca`) [verificado]. E1 sobre C1 evalúa exactamente el modelo desplegado.

### 4.2 Ajustes necesarios para que sea reproducible, medible y útil

1. **Resuelto el punto 2 de la revisión cruzada.** El 64,8 % se midió con las 273 imágenes de `otro.csv` (el `other` de EVA, `sort=6`) como sustituto de «fuera de distribución» [verificado en `fase5_evaluacion.ipynb`, celda 18]. Es un conjunto *near-OOD*: mismo concurso y misma estética. Hay que decirlo en §8.5.1 y matizarlo en §5.1.2.

2. **El umbral 0,85 se eligió mirando C1.** La curva de umbrales de la fase 5 usa `probs_test`, es decir, `test.csv` [verificado]. Por tanto C1 **no** está limpio para las métricas que dependen del umbral:
   - Sí está limpio para la exactitud sin umbral: el checkpoint se eligió con `val.csv`.
   - Consecuencia: la cobertura y la exactitud sobre las aceptadas medidas en C1 con 0,85 son optimistas.
   - Propuesta: (a) declararlo en §8.6; (b) repetir la curva en `val.csv` + `otro.csv` para ver si 0,85 cae en la misma zona; (c) usar C2 como comprobación limpia del umbral, porque el clasificador nunca lo ha visto.

3. **El 27,4 % no es «mal rechazado».** Cuenta **todas** las imágenes de prueba con probabilidad máxima < 0,85, aciertos y fallos incluidos. Rechazar un fallo es bueno. En §9.1 hay que dar:
   - la exactitud sin umbral (0,8510);
   - la cobertura (72,6 % = 100 − 27,4);
   - la exactitud sobre las aceptadas [pendiente de calcular];
   - la fracción de errores capturados por el rechazo [pendiente];
   - como métrica sin umbral, el AUROC `test` frente a `otro` [pendiente].

4. **El criterio de E1 («ninguna clase con F1 < 0,75 en C1») se conocía de antemano.** La F1 mínima ya medida es 0,7687 (producto). Ese criterio no es una hipótesis previa, sino una descripción. Opciones: presentarlo como descriptivo, o fijar un criterio nuevo que todavía no se haya medido. Por ejemplo, sobre C2: «exactitud sobre las aceptadas ≥ X» o «% enviado a *otro* ≤ Y».

5. **E2 se calcula con los engines, no con las 100 ejecuciones del LLM.** Las `confianza` son deterministas y no dependen del LLM. Leerlas de `salida_*.json` añadiría el ruido de la transcripción, que ya mide E4. Así E2 no necesita API y se puede ejecutar ya.

6. **E2 necesita una definición operativa de «la regla aplica» por gate**, escrita antes de anotar (ver §6.2). La más delicada es la del horizonte: el gate abre con horizontales estructurales, como escaleras, mesas o fachadas [según notebook 03]. Propuesta: anotar por separado `horizonte_natural` y `horizontal_estructural`, y dar los falsos positivos con las dos definiciones.

7. **El criterio «0 falsos positivos» de E2 probablemente no se cumplirá tal como está el código.** Hay indicios previos:
   - la `imagen51`: YOLO la detecta, GrabCut devuelve una máscara vacía y las dos métricas de espacio abren con confianza 1 [según notebook 02, §11.4];
   - falsos positivos de convergencia en escenas planas (86, 51, 85, 59), con etiquetas de ejemplo que no son del autor [según notebook 03].

   Hay que decidir **antes** de ejecutar: (a) mantener el criterio y aceptar un «no»; o (b) corregir los gates, por ejemplo con el gate de relleno propuesto en el notebook 02. La opción (b) calibra sobre el mismo C2, así que se declara en §8.6. No vale cambiar el criterio después de ver los resultados.

8. **E3 necesita reglas precisas para el verificador**, que los auditores actuales no tienen:
   - Citas de tuplas como `coord_centro_masa = (0.4830, 0.5899)`: los auditores las marcan como «no escalar» aunque sean correctas [según notebook 04].
   - Citas a W y a la cobertura (`[W(luz_tono) = 0.37]`, `[cobertura_por_dimension.X = 0.0]`): hoy salen como «campo inexistente» [según notebook 05].
   - Citar `X.confianza = 0.0` de una métrica cerrada: el control 4 lo marca como cita a cerrada, pero es la forma natural de redactar una salvedad. Propuesta: solo cuenta como violación citar `valor`, `valor_norm` o un campo de evidencia condicionado por esa métrica (tabla en §6.3).
   - «Truncada» debe separarse de «mal redondeada»: hoy son la misma categoría.
   - «Síntesis en el orden del arbitraje» solo se puede comprobar de forma exacta sobre `afirmaciones`, que es una lista estructurada con `dimension`. Sobre `sintesis`, que es prosa, solo cabe una heurística. Hay que decirlo.

9. **E4 necesita instrumentación que hoy no existe:**
   - un lanzador por lotes;
   - tiempos por tarea (vía `task_callback`);
   - tokens por modelo (`CrewOutput.token_usage` da el total; `agent.llm.get_token_usage_summary()` debería dar el reparto por agente y modelo, a verificar al implementarlo);
   - VRAM pico (`torch.cuda.max_memory_allocated`);
   - una tabla de precios de Gemini con fecha y fuente [pendiente].

   Además, `LecturaVisualTool` llama a `google-genai` directamente: **sus tokens no pasan por CrewAI ni se guardan** [verificado]. Para medirlos hay que modificar la herramienta (guardar `usage_metadata`) o declarar que el coste de la lectura visual queda fuera.

10. **Las ejecuciones se sobrescriben.** `PrioridadTool` lee los informes de `outputs/ejecucion/{stem}`, que devuelve `dir_ejecucion()`, sin opción de cambiar la ruta. Por eso cada ejecución debe hacerse en esa carpeta y copiarse después a una carpeta de experimento (`outputs/experimentos/{tanda}/{rep}/{stem}/`). Si no, las repeticiones de R y las 20 sin lectura visual se pisan entre sí.

11. **La caché de la lectura visual afecta a R.** Si la caché existe, las tres repeticiones reciben la misma lectura, y esa parte es idéntica por construcción. Hay que decidirlo y declararlo (recomendado: mantener la caché y decir que la variabilidad medida es la del crítico y los especialistas).

12. **El prompt contiene cifras del corpus antiguo** [verificado]. `agents.yaml` (líneas ~289-300) dice «escalas empíricas observadas sobre las 40 imágenes del corpus de calibración» y «solo 7 de las 40 imágenes abren el gate». Las escalas de `d_equilibrio` (0,03 / 0,12) son también de entonces. El prompt condiciona el texto que mide E3, así que hay que **actualizarlo y congelarlo antes** de las ejecuciones. El índice no lo incluye en «Cambios en otros capítulos».

13. **Hay que congelar el sistema antes de ejecutar** (§11): commit, constantes, prompts, modelos, W, pesos y versiones. Los notebooks muestran que las constantes cambiaron después de sus salidas guardadas. Por ejemplo, `PESO_MIN_MATIZ` vale 0,05 en el código y 0,15 en el texto y las salidas del notebook 01, y el gate cromático pasó de 0,10 a 0,09.

14. **Errata del índice:** en «Cambios en otros capítulos», el resumen extendido habla de «las cifras de E1–E7», pero ahora solo hay E1–E4.

15. **La rama `sin_sujeto_claro` no se da nunca en C2** [según notebook 02]: 61 imágenes `yolo`, 39 `saliencia` y 0 `sin_sujeto_claro`. La Tabla 1 de §9.2 debe decirlo, porque esa vía queda sin validar con fotos reales.

16. **Modelo nulo de fuga.** Es cierto que no depende del corpus. Dos matices:
    - El script que lo generó no está en el repositorio. La reproducción del notebook 03 coincide en la mayoría de los `n`, pero no en todos: con n = 10 da 0,40 frente a 0,50 en el JSON [según notebook 03].
    - Según el notebook 03 (§12.4, sin salida guardada), el redondeo a 4 decimales convierte un empate 4/7 en apertura.

    Ninguno obliga a regenerarlo, pero conviene mencionarlo en la amenaza de reproducibilidad.

---

## 5. Capítulo 8: qué necesita cada apartado

El capítulo 8 no contiene resultados. Lo que sí se puede generar con un notebook son las **tablas
descriptivas del diseño**: la composición de C2 y la lista de parámetros congelados.

| Apartado | Contenido | Qué hace falta | Fuente o notebook |
|---|---|---|---|
| 8.1 Qué se puede validar y qué no | Prosa | Nada | — |
| 8.2 Preguntas P1…P4 | Tabla pregunta ↔ objetivo específico de §1.4.2 | El texto de §1.4.2 de la memoria [pendiente: no está en el repo] | — |
| 8.3 Conjuntos | Tabla C1/C2/R (+ O, val y V20); composición de C2 (origen, licencia, criterio de selección, reparto por contexto, casos límite) | Manifiesto de C2; anotación de contexto (A1); anotación de casos límite; origen y licencia [pendiente] | `00_preparacion` |
| 8.4 Instrumentos | Verificador (qué controla y con qué reglas), registro de tiempo, tokens y coste, protocolo de anotación | Reglas del verificador (§6.3); diseño del lanzador (§6.4); guía de anotación (§6.2) | `comun.py`, `lanzador` |
| 8.5.1–8.5.4 Protocolos E1–E4 | Qué se ejecuta, cuántas veces, sobre qué conjunto y con qué criterio | Criterios fijados en `config_experimentos.yaml` antes de ejecutar | §6 de este documento |
| 8.6 Amenazas | Umbrales calibrados sobre C2; anotador único; LLM no determinista; sin comparación con un agente único; **umbral del clasificador elegido con C1**; *near-OOD* como sustituto de OOD; rama `sin_sujeto_claro` sin ejercitar; caché de la lectura visual | Nada nuevo | — |

---

## 6. Capítulo 9: experimentos

Formato de cada experimento: pregunta, conjuntos, datos y artefactos, procedimiento, métricas,
resultados que produce, criterio, qué se reutiliza, limitaciones y si se puede hacer en un notebook.

### 6.1 E1: clasificador de contexto (P1) → §9.1

- **Pregunta:** ¿el clasificador asigna el contexto correcto y rechaza lo que no reconoce, también fuera de EVA?
- **¿Se puede hacer en un notebook?** Sí, entero. No usa LLM. Tiempo de GPU: minutos.
- **Datos y artefactos:**
  - Modelo: `models/clasificador_contexto/{pesos_fine_tuning_ligero.pt, class_to_idx.json, umbral.json}` y la clase `ContextClassifier` de `tools/cnn_context.py`.
  - Para C1, O y val: los splits y `EVA_together/` del proyecto externo. La ruta debe ir en la configuración, sin rutas fijas en el código.
  - Para C2: `data/` y la anotación A1 (contexto verdadero con 6 etiquetas, incluida `otro`) [pendiente].
- **Procedimiento:**
  1. **E1a (C1, sin umbral):** inferir las 725 imágenes con el transform del sistema (Resize 256 → CenterCrop 224 → normalización ImageNet, idéntico al de la fase 5 [verificado]). Guardar logits y probabilidades por imagen. Reproducir la exactitud de 0,8510 y el `classification_report` como comprobación de integridad.
  2. **E1b (C1 + O, umbral):** calcular la curva cobertura/exactitud-selectiva (riesgo-cobertura) para umbrales de 0,50 a 0,99, el AUROC test-frente-a-otro, y los valores en 0,85: cobertura, exactitud sobre las aceptadas, errores capturados y % de O rechazado (el 64,8 %).
  3. **E1c (val + O, control del umbral):** la misma curva sobre `val.csv`, para comprobar si 0,85 no es un artefacto de haberlo elegido con test.
  4. **E1d (C2):** `predict()` sobre las 100 imágenes. Calcular exactitud frente a A1 con 6 etiquetas, % enviado a `otro`, matriz de confusión 6×6, exactitud sobre las aceptadas, y distribución de la confianza en C2 frente a C1 (cambio de dominio EVA → fotos nuevas).
  5. **E1e (fidelidad del Nivel 1, opcional):** tras la tanda de 100 ejecuciones, comprobar que `salida_orquestador.json:etiqueta_contexto` coincide con `predict()`. El orquestador es un LLM que copia la etiqueta.
  6. **Grad-CAM:** dos mapas del par que más se confunde en C1. Es producto → arquitectura, con 18 casos; el segundo es arquitectura → paisaje, con 15 [verificado en la matriz de la fase 5]. Uno de acierto y otro de error.
- **Métricas:** exactitud; precisión, recall y F1 por clase; F1 macro; matriz de confusión; cobertura; exactitud selectiva; AUROC; % a `otro`; histogramas de confianza.
- **Resultados que produce:**
  - `resultados/E1/predicciones_C1.csv`, `predicciones_O.csv`, `predicciones_val.csv` y `predicciones_C2.csv` (id, etiqueta, predicción, confianza, probabilidades);
  - `resultados/E1/metricas.json`;
  - figuras: matriz de confusión C1, matriz de confusión C2, curva riesgo-cobertura, histogramas de confianza y Grad-CAM.
- **Criterio del índice:** ninguna clase con F1 < 0,75 en C1. Ver §4.2-4: ya se conocía (mínimo 0,7687). Hay que proponer un criterio adicional sobre C2 y fijarlo antes de etiquetar.
- **Se reutiliza:** el código de evaluación de `fase5_evaluacion.ipynb` (celdas 6, 11, 15 y 18), `src/dataset.py` (`EVADataset`, `ImagenesSinEtiquetaDataset`), la celda de guardado de Grad-CAM de `fase6_gradcam.ipynb` y `ContextClassifier.cam` en el propio sistema.
- **Limitaciones:** el umbral se eligió con test; O es *near-OOD*; C2 lo etiqueta un único anotador; el `other` de EVA no es el `otro` del sistema (§5.1.2).

### 6.2 E2: condiciones de aplicabilidad (P2) → §9.2

- **Pregunta:** ¿los gates cierran cuando la regla no aplica, sin falsos positivos?
- **¿Se puede hacer en un notebook?** Sí. Solo usa los engines y las anotaciones, sin API. Construir la caché de percepción tarda unos 8 minutos [según notebook 02].
- **Datos:** C2; caché de percepción regenerada (`percepcion.json` con todas las cajas YOLO + `mascaras.npz`); anotación A2.
- **Anotación A2** (guía que hay que fijar antes de anotar; una columna por pregunta y valores `si`/`no`/`dudoso`):

  | Gate | Pregunta al anotador | Estado actual |
  |---|---|---|
  | `esquema_cromatico` | ¿La escena tiene color analizable, es decir, no es B/N, nocturna o desaturada? | [pendiente] Sin columna |
  | `angulo_horizonte` | `horizonte_natural`: ¿hay una línea de horizonte legible? `horizontal_estructural`: ¿hay una horizontal de la escena que cruza buena parte del encuadre? | [pendiente] La columna `horizonte_real` existe en `revision_lineas_direccion.csv`, pero **vacía (0/100)** |
  | `score_convergencia` | ¿Hay perspectiva lineal que organiza la profundidad? | [pendiente] La columna `perspectiva_real` existe, pero vacía (0/100) |
  | `ratio_espacio_negativo`, `ratio_nitidez` | `segmentacion`: ok / parcial / mal. `protagonista`: si / no | [según CSV] `revision_espacio_aislamiento.csv`: 70/100 con `segmentacion` (34 ok, 32 parcial, 4 mal), 69/100 con `protagonista`. **Faltan 30 y revisar la fila 61** |
  | composición (3 métricas) | ¿El anclaje (centro del bbox) está sobre un objeto real que es el protagonista? | Se puede derivar de `protagonista` + `segmentacion`; decidirlo en la guía |
  | (contexto, E1) | Contexto verdadero: animal / arquitectura / paisaje / producto-still_life / retrato-humano / otro | [pendiente] Sin columna |
  | (casos límite, §8.3) | B/N, sin sujeto, vegetación, horizonte, perspectiva | [pendiente] |

  Existe además `revision_composicion_espacial.csv` con `patron_percibido` para las 100 imágenes (49 centrada, 28 otro, 23 tercios). **No es una anotación de aplicabilidad**: sirve para un análisis complementario del arbitraje tercios/centrada (acuerdo con el veredicto frente a `margen_patron`), que puede ir en §9.5 o en limitaciones. Hay que revisar su fila 61.

- **Definición de error** (fijarla en §8.5.2):
  - **Falso positivo** (grave): la métrica está abierta y la regla no aplica.
    - En espacio: abierta y `segmentacion = mal` en la versión estricta, o `∈ {mal, parcial}` en la amplia.
    - En nitidez: la misma condición que en espacio, o la escena sin textura según la anotación.
  - **Falso negativo** (tolerable): la métrica está cerrada y la regla aplica.
- **Procedimiento:**
  1. Regenerar la caché (subproceso, reutilizando la celda 4 del notebook 02).
  2. Pasar los 4 engines por las 100 imágenes y extraer por imagen `confianza`, motivo y variable de decisión de cada métrica (reutilizando `pasada()` del NB03, la tabla `C` del NB02, `evaluar_corpus()` del NB01 y `medir()` del NB04).
  3. Comprobar la réplica contra el engine real con `assert`, como hacen los notebooks.
  4. **Tabla 1** (automática): % de las 100 imágenes con cada métrica abierta; reparto de `fuente` (yolo / saliencia / sin_sujeto_claro); cobertura por dimensión (reutilizando la celda 33 del NB05); motivo de cierre (reutilizando `motivo_corto` del NB03).
  5. **Tabla 2** (central): por gate, matriz 2×2 abierta/cerrada × aplica/no aplica; falsos positivos y negativos con la lista de imágenes; tasa de falsos positivos con intervalo de confianza (Wilson o Clopper-Pearson, porque n es pequeño).
- **Resultados que produce:** `resultados/E2/gates_C2.csv` (una fila por imagen y métrica), `tabla1.csv`, `tabla2.csv`, `fp_fn.json` y hojas de revisión de los falsos positivos (paneles).
- **Criterio del índice:** 0 falsos positivos. Ver §4.2-7.
- **Se reutiliza:** la celda 4 del NB02 (caché), la celda 37 (tabla `C`), las celdas 47 a 49 (hojas de revisión y CSV), la celda 63 (`simular`, si se evalúan gates alternativos), las celdas 27 y 43 del NB03 (`pasada`, `matrices`), la celda 49 del NB01 (`evaluar_corpus`) y la celda 10 del NB04 (`medir`).
- **Limitaciones:** anotador único; los umbrales se calibraron sobre este mismo C2; la rama `sin_sujeto_claro` no se ejercita; YOLO elige la caja más confiable, no la protagonista.

### 6.3 E3: fidelidad de citas y disciplina del crítico (P3) → §9.3

- **Pregunta:** ¿toda cifra del texto sale del informe, sin alterarse y sin citar métricas cerradas? ¿El crítico respeta el arbitraje?
- **¿Se puede hacer en un notebook?** El análisis sí. **Las ejecuciones no** (ver el lanzador en §6.4): dependen de la API de Gemini, tienen coste, son largas y conviene poder reanudarlas.
- **Datos:** tanda T100 (100 ejecuciones con lectura visual) y tanda V20 (20 imágenes sin lectura visual, `FLAG_CRITICO_VE_IMAGEN = False`).
  - Selección de V20 propuesta: estratificada por contexto (A1) y por cobertura de la dimensión líder. Guardarla como lista fija en la configuración.
- **El verificador** (`experimentos/comun.py`) unifica los auditores existentes con estas correcciones:
  - **Extracción de citas:** la regex actual `\[([^\[\]]*(?:\[\d+\][^\[\]]*)*)\]`, con división por `,` antes de `campo =`. Hay que añadir soporte de valores tupla `(a, b)` o `a, b` para los campos `coord_*`.
  - **Resolución de rutas:** en especialistas, `metrica.campo` o `campo`. En el crítico, `dimension.metrica.campo` o `metrica.campo` (lógica de `resolver` del NB05, celda 27).
  - **Gramática explícita para citas del arbitraje:** `W(dim)`, `vector_pesos.dim` y `cobertura_por_dimension.dim` se resuelven contra `arbitraje` y no cuentan como «campo inexistente».
  - **Clasificación de cada cita:** `exacta` (redondeo correcto a los decimales escritos, máximo 4 según `tasks.yaml`) / `truncada` (coincide con el truncamiento pero no con el redondeo) / `alterada` / `campo_inexistente` / `null_citado` / `texto_distinto`.
  - **Cita a métrica cerrada:** cuenta como violación citar `valor`, `valor_norm` o un campo de evidencia condicionado por esa métrica. Citar `.confianza` no es violación. Tabla propuesta de campos de evidencia, **a verificar contra `agents.yaml`**:

    | Métrica cerrada | Campos que tampoco se pueden citar |
    |---|---|
    | `esquema_cromatico` | `spread_cromatico`, `matices_dominantes[*]`, `n_matices_dominantes` (`ratio_pixeles_cromaticos` sí se puede) |
    | `angulo_horizonte` | lado de la inclinación (no es campo; solo comprobable con una heurística de texto) |
    | `score_convergencia` | `coord_punto_fuga` (vale `null` si está cerrado), `apertura_haz`, `n_lineas_inliers` |
    | `ratio_espacio_negativo` / `ratio_nitidez` | `var_laplaciano_figura`, `var_laplaciano_fondo` |
    | `d_tercios` / `d_centro` / `patron_dominante` | `coord_centroide`, `coord_p_cercano`, `margen_patron` |

  - **Controles del crítico:** los 9 de la celda 27 del NB05:
    1. arbitraje recomputado igual al transcrito;
    2. el campo citado existe;
    3. la cifra está bien redondeada;
    4. no se cita una métrica cerrada;
    5. `campos_citados` coincide con lo citado;
    6. `afirmaciones` sigue `orden_dimensiones`;
    7. hay una salvedad por métrica cerrada;
    8. la síntesis no trae cifras nuevas;
    9. vocabulario prohibido y jurisdicción de las observaciones visuales.

    Añadir la **fidelidad de transcripción** del informe: la herramienta se recalcula y se compara campo a campo con el `informe` del JSON.
- **Métricas:**
  - Por autor (4 especialistas + crítico): nº de citas; % con campo existente; % exactas; % truncadas; % alteradas; nº de citas a cerradas.
  - Crítico: % de críticas con `afirmaciones` en orden (exacto) y % con la síntesis en orden (heurística, marcada como tal); % con salvedades = nº de cerradas; nº de tensiones cuyas citas cubren ≥ 2 dimensiones; incidencias por control.
  - Con y sin lectura visual (V20): las mismas cifras en dos columnas, sobre las mismas 20 imágenes.
- **Resultados que produce:** `resultados/E3/citas.csv` (una fila por cita: tanda, imagen, autor, campo, escrito, real, clase), `controles_critico.csv`, `tabla_por_autor.csv`, `tabla_critico.csv`, `comparativa_V20.csv` y ejemplos de cada tipo de incidencia.
- **Criterio del índice:** 0 citas a cerradas; ≥ 95 % exactas; 100 % en orden. Fijar antes de ejecutar si «en orden» se refiere a `afirmaciones` (recomendado).
- **Se reutiliza:** `resolver`/`auditar` (NB01 celda 59, NB02 celda 43, NB03 celda 38, NB04 celda 29) y `auditar` con 9 controles (NB05 celdas 27 y 29).
- **Fallos conocidos de esos auditores:**
  - NB03: `TypeError` con los campos lista.
  - NB04: marca como «no escalar» citas de tupla correctas.
  - NB05: W y cobertura salen como inexistentes; `confianza = 0.0` cuenta como cita a cerrada.
- **Fixtures para desarrollar el verificador:** las ejecuciones que ya hay en `outputs/ejecucion/` (6 de C2: imagen6, 23, 51 incompleta, 65, 67 y 77; y 8 externas de Unsplash). Se hicieron con código y prompts anteriores. **No valen como resultados.**
- **Limitaciones:** el verificador comprueba cifras, no la graduación de adjetivos (por ejemplo, «clave baja extrema» con `media_L = 19` [según notebook 01]); la heurística de orden sobre la síntesis.

### 6.4 E4: coste, latencia y reproducibilidad (P4) → §9.4

- **Pregunta:** ¿cuánto cuesta y tarda un análisis, y es reproducible?
- **¿Se puede hacer en un notebook?** El análisis sí. Las ejecuciones van en un **script lanzador** (`experimentos/ejecutar_lote.py`), que se puede invocar desde un notebook.
- **Diseño del lanzador:** por cada imagen de una lista:
  1. borrar `outputs/ejecucion/{stem}`;
  2. `kickoff` con `construir_inputs()` de `main.py`;
  3. medir el tiempo de pared total y registrar la marca de tiempo de fin de cada tarea con `task_callback`. Con tareas asíncronas, la duración por tarea es aproximada; hay que decirlo;
  4. guardar `CrewOutput.token_usage` y `agent.llm.get_token_usage_summary()` por agente (API a verificar), la VRAM pico y los errores y reintentos;
  5. copiar la carpeta a `outputs/experimentos/{tanda}/{rep}/{stem}/` con un `registro.json` que incluya commit, modelos, flag y versión del prompt;
  6. **reanudable**: si ya existe `registro.json`, saltar la imagen.

  Para la tanda V20, fijar `crew.FLAG_CRITICO_VE_IMAGEN = False` antes de instanciar `TfgMultiagenteFotografia()` (el método `critico_agent` lee la global al construirse; a verificar).
- **Tandas:** T100 (C2 × 1); R (10 imágenes × 3 repeticiones, en una tanda aparte para que el intervalo entre repeticiones sea corto); V20 (20 imágenes × 1 sin lectura visual). Unas **150 ejecuciones** en total.
- **Selección de R propuesta:** 2 imágenes por contexto de A1, mezclando gates abiertos y cerrados, e incluyendo alguna con un valor cerca de un umbral. Guardarla como lista fija.
- **Subexperimentos:**
  - **E4a, determinismo de la capa Python (sin API, notebook):** ejecutar los engines 3 veces sobre R, en el mismo proceso y en procesos distintos, y comparar bit a bit los informes (sin rutas). Separa el no determinismo de Python (YOLO en CUDA, BLAS en k-means; el notebook 01 documenta variaciones de 1e-14 resueltas con el redondeo) del del LLM.
  - **E4b, coste y latencia (T100):** tiempo total y por tarea (media y p95); tokens por modelo (flash / pro / lectura visual si se instrumenta); €/imagen con la tabla de precios fechada; VRAM pico; tasa de fallos y reintentos.
  - **E4c, reproducibilidad (R × 3):** informes transcritos idénticos entre repeticiones y frente a la herramienta; `arbitraje` idéntico; etiqueta del orquestador idéntica; en el texto, % de citas (`campo = valor`) repetidas entre ejecuciones (índice de Jaccard) y nº de citas por ejecución.
- **Resultados que produce:** `resultados/E4/registro_T100.csv`, `latencia.csv`, `tokens_coste.csv`, `determinismo_python.csv`, `reproducibilidad_R.csv` y figuras (distribución de la latencia, tokens por modelo).
- **Criterio del índice:** informes 100 % idénticos. Precisar si se refiere a la herramienta (E4a, esperado 100 %) o a la transcripción del LLM (E4c).
- **Dependencias:** `GEMINI_API_KEY` con cuota suficiente; tabla de precios [pendiente]; instrumentar la lectura visual [pendiente de decidir]. `tracing=True` envía trazas y puede añadir latencia: decidir si se mantiene y declararlo.

### 6.5 Casos ilustrativos → §9.5

- Tres casos con sus paneles: un acierto típico, un cierre correcto y un falso negativo por la limitación de COCO. En la anotación de espacio ya hay candidatos para este último: 8 (edificio), 33 (árbol), 76 (reloj), 88 (joya) y 99 (humo) [según CSV].
- **Notebook:** `E5_casos_ilustrativos.ipynb`. Elige los casos a partir de los resultados de E2 y E3, no a mano antes, y compone paneles con el original, los PNG de verificación de los 4 especialistas, Grad-CAM, percepción y un extracto de la crítica con sus citas.
- **Se reutiliza:** `verOutputs.ipynb` (localización de los PNG por sufijo: `composicion`, `lineas`, `luztono`, `espacio`). Hay que corregir sus rutas relativas.

### 6.6 Qué demuestran los resultados y limitaciones → §9.6

Es prosa. Un notebook de resumen (`E9_resumen_criterios.ipynb`) lee `resultados/E*/metricas.json` y
genera la tabla final pregunta → criterio → resultado (sí / no / en parte) → cifra, con el tamaño real
de cada amenaza de §8.6.

---

## 7. Parámetros y componentes que requieren calibración

Estos parámetros se calibran (o se documenta su calibración) sobre C2 **antes** de congelar el sistema.
El capítulo 6 muestra la distribución que justifica cada umbral. Sus tasas de acierto van en §9.2.

| Componente | Parámetro (valor actual) | Evidencia o estado | Qué hacer |
|---|---|---|---|
| Luz y tono | `RATIO_CROMATICO_MIN = 0.09` | Comentario desactualizado («4 imágenes en 0.000 y la siguiente en 0.287»). Con 100 imágenes había casos pegados al corte: img77 0,086, img28 0,093, img74 0,099, y el siguiente abierto en 0,171 [según notebook 01, con el umbral anterior de 0,10] | Histograma del ratio con el corte; decidir si se mueve a la banda vacía; actualizar comentario, esquema y Tabla 2 de la memoria |
| Luz y tono | `PESO_MIN_MATIZ = 0.05` (antes 0,15) | La nota del autor en el NB01 (celda 40) dice que se cambió; el comentario del código sigue citando 17/25 imágenes | Rehacer la ablación fusión/filtro (NB01, celda 53) con el valor actual |
| Luz y tono | `UMBRAL_FUSION_MATIZ = 15`, `SAT_MIN`/`VAL_MIN = 0.15` | Documentados en el código | Solo documentar |
| Espacio | `VAR_LAPLACIANO_MIN = 50` | Comentario desactualizado (33 imágenes; «imagen14 = producto blanco», que ahora es un tren) [según notebook 02] | Diagrama de dispersión en escala log de `var_max` sobre imágenes `yolo` (NB02, celda 55) |
| Espacio | Gate de relleno del bbox (**no existe**) | Propuesto por `imagen51` (máscara vacía que abre las dos métricas) | Decisión previa (§4.2-7) |
| Espacio | `MARGEN_FRONTERA_RATIO = 0.01`, `ITER_GRABCUT = 5` | Cambian el índice de nitidez unos 0,19 de media y lo hacen cambiar de signo en algunas imágenes [según notebook 02, texto] | Documentar la sensibilidad (NB02, celdas 57 y 59) |
| Líneas | `LONGITUD_MIN_HORIZONTE = 0.13`, `ANGULO_MAX_HORIZONTE = 10` | Comentarios del corpus de 40 | Rectángulo de aceptación (NB03, celda 45) |
| Líneas | `APERTURA_MIN_GRADOS = 35` | El comentario cita «42-90° reales, 9-30° paralelismo» del corpus antiguo | Histograma de apertura (NB03, celda 47) |
| Líneas | Detección (σ = 2, Canny 40/70, Hough) | Documentado | Solo documentar |
| Líneas | `modelo_nulo_fuga.json` | No depende del corpus; generador fuera del repo; posible empate por redondeo | Documentar; no regenerar salvo que cambie el núcleo (la guarda lo exige) |
| Percepción | `CONF_YOLO_THRESHOLD = 0.75`; pesos `yolo11m.pt` | Barrido de identidad según umbral (NB02, celda 61) | Documentar (afecta a 2 especialistas); §5.2.1: comparar pesos sobre C2 o no citar corpus |
| Percepción | `BLOB_AREA_MIN/MAX_RATIO` | Nunca se activan en C2 | Declarar como no ejercitados |
| Composición | Sin umbrales; escalas de lectura del prompt (margen 0,01/0,15; `d_equilibrio` 0,03/0,12) | Rango real de `d_equilibrio` 0,0072-0,2438; 0,03 = percentil 27 y 0,12 = percentil 84 [según notebook 04] | Recalcular sobre C2 (§6.2.3) y actualizar `agents.yaml` |
| Clasificador | `umbral_otro = 0.85` | Elegido con test + otro | Control con val (E1c) |
| Nivel 3 | W (versión 1.1) | Hipótesis, no se calibra | Solo el análisis de sensibilidad del orden (NB05, celda 31), si se quiere |

**Notebook de calibración** (apoyo al capítulo 6, no a los capítulos 8 y 9): `01_calibracion_C2.ipynb`
genera los histogramas con el umbral marcado y la tabla de constantes congeladas (reutilizando las
«chuletas» del NB01 celda 86, NB02 celda 81 y NB03 celda 68, que leen `getattr(modulo, CONSTANTE)`).

---

## 8. Inventario reutilizable de `/notebooks`

Los cinco notebooks son **didácticos y de calibración**: importan el código real (`lt`, `ea`, `ld`, `ce`,
`pt`) y comprueban sus réplicas con `assert`. Sus salidas guardadas son anteriores a cambios de
constantes y a la nueva `imagen61`, así que ninguna cifra suya vale como resultado sin volver a ejecutarlo.

| Notebook | Elementos reutilizables (celda) | Para qué | Adaptación necesaria |
|---|---|---|---|
| `01_agente_luz_tono` | `paso_a_paso` + `desde_kmeans` (24); `evaluar_corpus` (49); histograma y sensibilidad del gate (50); ablación fusión/filtro (53); `constantes()` (65); `comparar()` (66); `resolver`/`auditar` (59); chuleta (86) | E2 (color), calibración, E3 | `desde_kmeans` usa las constantes del módulo: verificar tras el cambio a 0,05. Ojo: `UMBRAL_FUSION_MATIZ` es un argumento por defecto y cambiarlo en caliente no tiene efecto |
| `02_agente_espacio_aislamiento` | Caché en subproceso con todas las cajas YOLO (4); tabla `C` con gates (37); plantilla CSV + `hoja_revision` (47-49); señales y `barrido` (51); `simular()` (63); sensibilidad margen/iteraciones (57, 59); barrido de umbral YOLO y caja alternativa (61); detección de duplicados (66) | E2 (espacio, fuente), caché común, calibración | Regenerar la caché (`RECALCULAR=True`). La celda 51 falló con `NameError` por el orden de ejecución. Separadores y codificación de los CSV mezclados (`;` frente a `,`; notas con mojibake): leer con `sep=None, engine="python"` y probar `utf-8-sig` y `cp1252` |
| `03_agente_lineas_direccion` | `pasada()` sobre el corpus en unos 6 s (27); curva del nulo frente al corpus (28); `generar_modelo_nulo` (30); prueba de la guarda (32); `motivo_corto` (34); `matrices()` falsos positivos/negativos (43); rectángulo del horizonte (45); histograma de apertura (47); empate por redondeo (49); `experimento()` (51) | E2 (líneas), calibración | La celda 38 falla con listas; la 45 con `NameError ETIQ`. Etiquetas `EJEMPLO` hechas por el asistente, no por el autor: **no usarlas** |
| `04_agente_composicion_espacial` | `medir()` (10); simulación del sesgo estructural (19); percentiles de `d_equilibrio` (33); bandas de margen (35); `mapa_decision` (37); matriz `patron_percibido` frente a `patron` (39, con 100 etiquetas del autor) | E2 (composición), calibración §6.2.3, §9.5 | Revisar la fila 61 del CSV |
| `05_agente_critico` | Recomputar el arbitraje; orden por contexto (10); trampas de cobertura (17); prueba de la guarda de procedencia (21); **auditor con 9 controles** (27) y cuadro de mando (29); sensibilidad de W (31); cobertura sobre el corpus `COB` (33-34) | E3 (núcleo del verificador), Tabla 1 de E2 | Correcciones de §6.3 |
| `verOutputs.ipynb` (raíz) | Localizar los PNG por sufijo | §9.5 | Rutas relativas fijas: usar `RAIZ` |
| `revision_*.csv` | Anotaciones del autor (espacio 70/100; composición 100/100; líneas 0/100) | E2, §9.5 | Completar y revisar la fila 61 |

Patrón común que conviene conservar en los notebooks nuevos:

- localizar `RAIZ` buscando `pyproject.toml` y añadir `src` a `sys.path`;
- construir la caché pesada en un **subproceso**, para que el kernel no cargue YOLO;
- comprobar con `assert` que la réplica coincide con el engine;
- dejar las decisiones del anotador en un CSV.

---

## 9. Recursos externos (`Documents/Curso_CNN_FineTunning`)

La carpeta se llama `Curso_CNN_FineTunning`, con doble «n». Existe también `Documents/eva-dataset/`
(con `LICENSE`, `data/`, `images/` y `readme.md`), que no se ha examinado.

| Recurso | Ruta (relativa a `Curso_CNN_FineTunning/`) | Experimento | Qué extraer |
|---|---|---|---|
| Splits | `clasificador_contexto/data/splits/{train,val,test,otro}.csv` (columnas `image_id, sort, label`) | E1 | `test.csv` (C1), `otro.csv` (O), `val.csv` (control del umbral). Estratificados 70/15/15 con `SEED = 42` y sin fuga entre splits [verificado en la fase 1] |
| Imágenes EVA | `EVA_together/{image_id}.jpg` (5101 ficheros) | E1 | Entrada de la inferencia. **No copiarlas al repo:** poner su ruta en la configuración |
| Métricas de referencia | `clasificador_contexto/outputs/metrics/{classification_report.json, umbral.json, matriz_de_confusión.png}` | E1 | Valores esperados para comprobar la integridad (0,8510; matriz `[[125,0,5,5,1],[0,122,15,6,0],[1,11,132,5,3],[7,18,8,103,3],[3,3,4,10,135]]`) |
| Evaluación | `clasificador_contexto/notebooks/fase5_evaluacion.ipynb` (celdas 6, 11, 15, 18, 19) | E1 | Bucle de inferencia, informe, matriz, curva de umbrales y ganancia marginal |
| Grad-CAM | `clasificador_contexto/notebooks/fase6_gradcam.ipynb`; `outputs/gradcam/gradcam_*.png` | E1 | Procedimiento de acierto frente a error del par producto → arquitectura |
| Datasets | `clasificador_contexto/src/dataset.py` (`EVADataset`, `ImagenesSinEtiquetaDataset`) | E1 | Copiar la lógica (unas 30 líneas) en `comun.py` en lugar de importar del proyecto externo |
| Checkpoints | `clasificador_contexto/outputs/checkpoints/fase4_fine_tuning_ligero.pt` | — | Idéntico a los pesos del repo; no hace falta |
| Metadatos EVA | `EVA_data/image_content_category.csv` (5101 filas, `image_id, sort`) | E1, §8.3 | Mapa `sort` → categoría: 1 animal, 2 arquitectura, 3 humano, 4 natural/rural, 5 still life, 6 other |
| Ejecuciones antiguas | `outputs_imagenes/ejecucion/*` | — | Corpus antiguo. **No usar** |

---

## 10. Dependencias pendientes y decisiones abiertas

**Datos y anotación**

1. Origen, licencia y criterio de selección de las 100 imágenes de C2 (§8.3).
2. Anotación A1: contexto verdadero de las 100 imágenes.
3. Anotación A2: completar color, horizonte, perspectiva y segmentación (las 30 que faltan); revisar la fila 61 en los tres CSV.
4. Anotación de casos límite de C2 (§8.3).
5. Texto de §1.4.2 (objetivos específicos) para la tabla de §8.2.

**Instrumentación**

6. Lanzador por lotes con registro (§6.4).
7. Tokens de la lectura visual: modificar `LecturaVisualTool` o dejarlos fuera.
8. Tabla de precios de Gemini 2.5 Flash/Pro con fecha y fuente.
9. Cuota de API para unas 150 ejecuciones.

**Decisiones del autor** (entre paréntesis, la recomendación)

10. Definición de «la regla aplica» por gate, sobre todo en el horizonte (dos columnas: natural y estructural).
11. ¿Corregir antes de ejecutar los falsos positivos conocidos (gate de relleno) o mantener el código y aceptar el resultado? (Decidirlo y declararlo en §8.6; no cambiarlo después.)
12. Criterios de E1 sobre C2 y precisión de «en orden» (E3) e «idénticos» (E4).
13. Caché de la lectura visual en R (mantenerla y declararlo).
14. `tracing=True` durante las tandas (desactivarlo para medir la latencia y declararlo).
15. Actualizar y congelar las escalas del corpus antiguo en `agents.yaml` antes de ejecutar.
16. Dónde guardar los notebooks: `notebooks/` está en `.gitignore`, así que no se versionarían (propuesta en §12).

---

## 11. Lista de congelación antes de las tandas

Guardar en `experimentos/congelacion.json`, generado por `00_preparacion`:

- [ ] Commit del código (`git rev-parse HEAD`) con el árbol limpio.
- [ ] MD5 de `data/*.jpg` (manifiesto de C2), `models/clasificador_contexto/*`, `models/yolo/yolo11m.pt`, `parametros/*.json`, `config/*.yaml`.
- [ ] Valores de todas las constantes de §7 (con `getattr`).
- [ ] `MODEL`, `MODELO_CRITICO`, `FLAG_CRITICO_VE_IMAGEN`, modelo, temperatura y `VERSION_PROMPT` de la lectura visual.
- [ ] Versiones de los paquetes (`importlib.metadata`) y de Python; GPU y CUDA.
- [ ] Criterios de éxito de E1–E4 escritos en `config_experimentos.yaml` **antes** de lanzar.
- [ ] Listas fijas de R y V20.

---

## 12. Propuesta de estructura para los notebooks

Carpeta nueva **versionada** `experimentos/`. `notebooks/`, `data/` y `outputs/` están ignoradas. Los
resultados ligeros (CSV, JSON y PNG) van en `experimentos/resultados/`; las ejecuciones completas,
en `outputs/experimentos/` (ignorada).

```
experimentos/
├── config_experimentos.yaml     # rutas (EVA externo), semillas, listas R y V20, criterios, umbrales de verificación
├── congelacion.json             # generado por 00
├── manifiesto_C2.csv            # nombre, md5, dimensiones, origen, licencia
├── anotaciones/                 # A1 contexto, A2 aplicabilidad, casos límite (CSV utf-8, separador ',')
├── comun.py                     # utilidades compartidas (ver abajo)
├── ejecutar_lote.py             # lanzador reanudable (T100, R×3, V20)
├── 00_preparacion.ipynb         # manifiesto, congelación, caché de percepción, plantillas de anotación,
│                                # validación de las anotaciones, tablas descriptivas de §8.3
├── 01_calibracion_C2.ipynb      # apoyo al cap. 6: histogramas con umbral y tabla de constantes
├── E1_clasificador_contexto.ipynb       # §9.1 (E1a–E1e + Grad-CAM)
├── E2_condiciones_aplicabilidad.ipynb   # §9.2 (Tabla 1, Tabla 2)
├── E3_fidelidad_citas_critico.ipynb     # §9.3 (T100 + V20)
├── E4_coste_latencia_reproducibilidad.ipynb  # §9.4 (E4a sin API; E4b, E4c tras las tandas)
├── E5_casos_ilustrativos.ipynb          # §9.5
├── E9_resumen_criterios.ipynb           # §9.6: tabla pregunta→criterio→resultado
└── resultados/{E1,E2,E3,E4,E5}/
```

**`comun.py`** (funciones previstas):

- `raiz()`, `cargar_config()`, `comprobar_congelacion()`: avisa si el commit o los MD5 difieren de `congelacion.json`.
- `imagenes_C2()`, `leer_csv_robusto(ruta)`: detecta separador y codificación.
- `construir_cache_percepcion(recalcular=False)`: subproceso, derivado de la celda 4 del NB02.
- `medir_gates_C2()`: devuelve el DataFrame por imagen y métrica con confianza, motivo y variable de decisión.
- Verificador: `extraer_citas`, `resolver_ruta`, `clasificar_cita`, `CAMPOS_CONDICIONADOS`, `auditar_especialista`, `auditar_critica`, `fidelidad_transcripcion`.
- `cargar_tanda(nombre)`, `guardar_resultado(exp, nombre, obj)`, `intervalo_wilson(k, n)`.

**Plantilla de cada notebook E\*:**

1. **Propósito y protocolo:** pregunta, referencia a §8.5.x y criterio, leído de la configuración.
2. **Configuración y comprobación de congelación.**
3. **Carga de datos y anotaciones**, con validación de que están completas.
4. **Ejecución del experimento:** cachea resultados intermedios y es idempotente.
5. **Métricas:** tablas.
6. **Figuras.**
7. **Criterio:** sí / no / en parte, calculado y no escrito a mano.
8. **Exportación** a `resultados/E*/`.
9. **Notas de limitaciones** propias del experimento.

---

## 13. Orden de trabajo propuesto

1. Decisiones de §10 (10-16) y criterios en `config_experimentos.yaml`.
2. `00_preparacion`: manifiesto, regenerar la caché (por la nueva `imagen61`) y plantillas de anotación.
3. Anotar A1, A2 y casos límite (usando las hojas de revisión).
4. `01_calibracion_C2`: si se cambia algún umbral, hacerlo **ahora**; actualizar comentarios, esquemas y `agents.yaml`; ejecutar `tests/`; commit.
5. Congelar (`congelacion.json`).
6. Sin API: `E1` (entero), `E2` (entero) y `E4a`.
7. Con API: lanzar T100 → R×3 → V20 con `ejecutar_lote.py`.
8. `E3`, `E4b`/`E4c`, `E5`, `E9`.
9. Aplicar los «Cambios en otros capítulos» del índice (corrigiendo «E1–E7» por «E1–E4»).

---

## 14. Estado y traspaso (actualizado el 2026-09-27)

> **Si vienes de otro chat, lee antes §15**: recorta el alcance y sustituye la tabla de estado
> (14.1) y la lista de lo que falta (14.6). Lo demás de esta sección (API de `comun.py`, forma de
> trabajo, convenciones y hechos verificados) sigue vigente. Las secciones 1-13 son el análisis y
> el plan originales; siguen vigentes salvo donde §14 o §15 digan otra cosa.

### 14.1 Estado por pieza (antes del recorte; el estado vigente está en §15.3)

| Pieza | Estado | Nota |
|---|---|---|
| `config_experimentos.yaml` | Hecho | Criterios de E1 sobre C2 a `null` [pendiente del autor]; `listas` R y V20 vacías |
| `comun.py` | Hecho (lo que necesitan 00, E1 y E2) | Falta el verificador de E3 y la carga de tandas (ver 14.6) |
| `_cache_percepcion.py`, `_cache_gates.py` | Hechos y ejecutados | Cachés en `outputs/experimentos/cache/{percepcion,gates}/` (461 s y 413 s) |
| `_ejecutar_notebook.py` | Hecho | Valida un notebook sin Jupyter (ver 14.3) |
| `manifiesto_C2.csv` | Generado | `origen`, `autor`, `licencia` y `url` vacíos [pendiente del autor] |
| `anotaciones/GUIA_ANOTACION.md` | Borrador | El autor debe revisarla y poner la fecha de fijación |
| `anotaciones/A1_contexto.csv` | Plantilla vacía (0/100) | Anotación pendiente del autor |
| `anotaciones/A2_aplicabilidad.csv` | Parcial | `segmentacion` 60/60 `yolo`; `protagonista` 59/60 (falta imagen48); columnas ciegas 0/100 |
| `anotaciones/patron_percibido.csv` | Migrada | 99/100 (imagen61 vaciada) |
| `00_preparacion.ipynb` | Hecho y probado | Congelación con `CONGELAR = False` |
| `E1_clasificador_contexto.ipynb` | Hecho y probado | E1d con verdad (A1) y E1e (T100) pendientes de datos |
| `E2_condiciones_aplicabilidad.ipynb` | Hecho y probado | Tabla 1 lista; Tabla 2 pendiente de A2 |
| `01_calibracion_C2.ipynb` | **No existe** | Ver 14.6 |
| `ejecutar_lote.py` | **No existe** | Ver 14.6 |
| `E3`, `E4`, `E5`, `E9` | **No existen** | Ver 14.6 |
| `congelacion.json` | No existe | Se genera en 00 §7 al final |
| `experimentos/resultados/` | Vacío a propósito | Los resultados de prueba se borraron: el sistema no está congelado |

**Nada de `experimentos/` está commiteado todavía.**

### 14.2 Ficheros de `experimentos/` y API de `comun.py`

Los notebooks importan `comun` después de localizar la raíz (primera celda de cada uno). Funciones:

- **Rutas y configuración:** `raiz()`, `paquete()`, `preparar_sys_path()`, `cargar_config()`, `ruta_config(clave)`, `dir_resultados(exp)`, `relativa(p)`, `guardar_json(obj, ruta)` (convierte NumPy y pasa NaN a `null`), `md5(ruta)` (memorizada), `imagenes_C2()`, `leer_csv_robusto(ruta)` (`,` o `;`, utf-8 o cp1252, todo como texto), `intervalo_wilson(k, n)`.
- **Anotaciones:** `ESQUEMA_A1`, `ESQUEMA_A2`, `ESQUEMA_PATRON`, `A2_SOLO_YOLO` (`segmentacion`, `protagonista`), `normalizar_valor`, `leer_anotacion(nombre, esquema)` (indexada por `imagen`), `validar_anotacion(df, esquema, requeridas)`, `revisiones_obsoletas(a2, percepcion)`.
- **Constantes del sistema sin importar el código:** `constantes_modulo(ruta)` y `constantes_sistema(modulos)` (vía `ast`; los prompts largos van como MD5), `md5_codigo(modulos)`, `modelo_llm()` (solo la línea `MODEL` de `.env`).
- **Procedencia y congelación:** `procedencia()`, `ficheros_congelados()`, `estado_sistema()`, `diferencias_estado()`, `comprobar_congelacion()`. Un commit posterior solo cuenta como diferencia si cambia `src/`, `models/`, `pyproject.toml` o `uv.lock`.
- **Cachés:**
  - `CACHES`, `construir_cache(nombre)`, `vigencia_cache(nombre)`, `cargar_cache(nombre)`, `meta_cache(nombre)`;
  - atajos de percepción: `construir_cache_percepcion`, `cargar_cache_percepcion()` → `(meta, percepcion, mascaras)`;
  - la vigencia compara el MD5 de las imágenes, el MD5 del código y las constantes de `MODULOS_CACHE` / `MODULOS_GATES`, los pesos de YOLO (percepción) y la fecha de la caché de percepción (gates);
  - si algo cambia, la caché se reconstruye sola en un subproceso.
- `DIMENSIONES` sigue el orden y las claves de W: `composicion_espacial`, `lineas_direccion`, `espacio_aislamiento_sujeto`, `luz_tono`.

Contenido de las cachés:

- **`percepcion.json`:** por imagen, la salida de `SharedPerceptionEngine.detect()`, más todas las cajas YOLO, `md5`, `rect`, `fuente_mascara` y `cobertura_mascara`.
- **`mascaras.npz`:** la máscara de cada imagen (clave = stem).
- **`gates.json`:** por imagen, `informes` (los 4 `model_dump(mode="json")`) y `arbitraje` (el de `PrioridadEngine` con etiqueta `otro`).

### 14.3 Cómo se trabaja

- **Kernel:** el Python de `.venv`. `jupyter`, `nbformat`, `nbconvert` y `nbclient` **no** están instalados.
- **Validar un notebook** tras editarlo: `.venv/Scripts/python.exe experimentos/_ejecutar_notebook.py experimentos/<nb>.ipynb` (con `--hasta N` se ejecutan solo N celdas). Escribe resultados como una ejecución normal: bórralos después si son de prueba.
- **Editar notebooks:** directamente sobre el `.ipynb` (NotebookEdit). Los tres se generaron desde fuentes intermedias que ya no existen; no hay que regenerarlos desde ningún otro sitio.
- **Tiempos:**
  - E1: ~1 min la primera vez (inferencia de ~1.800 imágenes en GPU) y segundos con las predicciones en caché;
  - 00 y E2: segundos con las cachés vigentes;
  - reconstruir las cachés: ~8 min (percepción) + ~7 min (gates).
- **Estilo de los notebooks** (mantenerlo en los nuevos):
  - estructura de §12: protocolo, configuración y congelación, datos, ejecución, métricas, figuras, criterio calculado, exportación y limitaciones;
  - `assert` de la réplica contra el código real;
  - criterios leídos de la configuración;
  - paleta de figuras: azul `#2a78d6`, naranja `#eb6834` y aqua `#1baf7a`, en ese orden fijo;
  - figuras sin bordes superior y derecho, con rejilla `#e1e0d9`.

### 14.4 Decisiones y convenciones ya tomadas

1. **Anotación ciega:**
   - A1 y las columnas ciegas de A2 se rellenan con las hojas `ciega_*.png` antes de ver `segmentacion_*.png`;
   - la Tabla 2 de E2 no se calcula para un gate hasta que están completas sus columnas;
   - **no enseñar al autor `resultados/E2/gates_C2.csv`, ni listas de gates por imagen, antes de que termine las columnas ciegas.**
2. **Los notebooks nunca sobrescriben anotaciones** ni las columnas manuales del manifiesto: solo crean lo que falta.
3. **Resultados de prueba:** se borran mientras el sistema no esté congelado. Los números de 14.5 son provisionales.
4. **E2 mide con los engines**, no con los `salida_*.json` de las ejecuciones: así no entra el ruido de la transcripción del LLM.
5. **Definición de «la regla aplica» por gate** (estricta y amplia): está en `GUIA_ANOTACION.md` y codificada en la función `aplica()` de E2; las dos deben coincidir.
   - Se combinan con lógica de tres valores: `no` domina, después `dudoso`.
   - Una imagen `dudoso` sale de la matriz 2×2 y se cuenta aparte.
   - `E2.definicion_criterio: estricta` y `fp_max_por_gate: 0`.
6. **Réplica del clasificador con tolerancia TF32 de 1e-3** (el sistema corre con TF32 activado, como en la fase 5).
7. **Revisiones de segmentación versionadas:** `fuente_revisada` y `clase_revisada` guardan lo que se vio al revisar; si la percepción cambia, la fila pasa a obsoleta.
8. **R y V20:**
   - 00 §6 los propone estratificados por contexto de A1 (sin `otro`), alternando `yolo` y `saliencia`, con la semilla de la configuración;
   - el autor pega el bloque en `config_experimentos.yaml` (`listas:`) antes de congelar.
9. **Congelación:**
   - se niega si hay cambios sin commit, la caché está obsoleta, el manifiesto no coincide o ya existe `congelacion.json`;
   - con pendientes (anotaciones, guía sin fecha, criterios `null`, listas vacías) solo congela con `FORZAR = True`, y lo registra en el JSON.

### 14.5 Hechos verificados en esta fase

- **Percepción de C2:** 60 `yolo`, 40 `saliencia` y 0 `sin_sujeto_claro`. La nueva `imagen61` va por saliencia (YOLO solo ve `dog` 0,40).
- **Máscaras degeneradas** (cobertura < 0,01): imagen8, 15, 35, 51 (vacía) y 100 con `yolo`; imagen3 y 28 con `saliencia`. Son candidatas a falso positivo de espacio y nitidez.
- **Tras la migración:** 69 revisiones de segmentación vigentes y 0 obsoletas.
- **Orientación EXIF:** ninguna imagen de C2 la tiene distinta de 1, así que el clasificador (PIL, sin rotación EXIF) y los especialistas (cv2, con rotación) ven la misma imagen.
- **E1** (provisional):
  - **Integridad:** se reproducen la exactitud 0,8510, la matriz, el 27,4 % y el 64,8 %.
  - **C1, umbral 0,85:** cobertura 72,6 %; exactitud de las aceptadas 0,943; errores capturados 72,2 % (78/108); aciertos perdidos 19,6 %; AUROC C1 frente a O 0,754.
  - **val:** exactitud 0,854; cobertura 70,3 %; exactitud de las aceptadas 0,951; AUROC 0,735. Con la regla de ganancia marginal de la fase 5, **0,85 es el óptimo estricto en val** (en C1 lo era 0,80).
  - **C2:** 10/100 van a `otro` (IC 95 % 5,5-17,4 %). Las demás, 18 por clase. Mediana de la confianza: 0,984 en C2 y 0,965 en C1.
  - **Grad-CAM:** acierto EVA 527582 (0,998); error producto → arquitectura EVA 14852 (0,993, aceptado). De los 18 errores de ese par, 5 superan el umbral.
  - **Ganancia marginal:** en C1 da 2,94 / 1,70 frente a 2,85 / 1,72 de la salida guardada de la fase 5; las conclusiones no cambian.
- **TF32:** la confianza entre el lote de 64 y `predict()` difiere hasta 4,4e-4 (4e-7 sin TF32). Hay 5 imágenes de EVA a menos de 1e-3 del umbral. Es un dato para E4 y §8.6.
- **Grad-CAM de `predict()`** (`cnn_context.py:113`): superpone el mapa del recorte central sobre la imagen entera redimensionada a 224×224, desalineado en fotos no cuadradas. **Sin corregir**: decisión del autor.
- **E2 Tabla 1** (provisional): réplica 600/600; máscara de espacio igual a la de la caché; cobertura igual a la del arbitraje.
  - Métricas abiertas: esquema cromático 80 %, horizonte 26 %, convergencia 25 %, espacio 60 %, nitidez 55 %, composición 60 %.
  - Cierres del horizonte: 58 fragmento corto, 12 sin candidatos, 4 inclinado.
  - Cierres de la convergencia: 31 no supera el azar, 26 haz casi paralelo, 18 pocas líneas.
  - Nitidez: 5 cierres por falta de textura.
  - Cobertura media: composición 0,70, líneas 0,255, espacio 0,575, luz 0,90.
  - Tabla 2 probada con anotaciones simuladas en memoria; cuadra con un cálculo independiente.

### 14.6 Qué falta (SUPERADO: la lista vigente está en §15.4)

**A. Del autor (bloquean la Tabla 2 de E2, E1d y la congelación)**

1. Revisar y fijar `anotaciones/GUIA_ANOTACION.md` (poner la fecha de fijación).
2. Anotar A1 (100) y las columnas ciegas de A2 (100 × 6) con `outputs/experimentos/hojas_revision/ciega_*.png`. Después, `protagonista` de imagen48 y repasar la fila 61 de `patron_percibido.csv`.
3. Completar `origen`, `autor`, `licencia` y `url` en `manifiesto_C2.csv`.
4. Fijar en la configuración los criterios de E1 sobre C2 (`exactitud_aceptadas_min_C2`, `pct_otro_max_C2`) **antes** de anotar A1.
5. Decidir §4.2-7: añadir el gate de relleno (imagen51 y las otras máscaras vacías) o aceptar el «no» de E2. Decidirlo antes de ver la Tabla 2.
6. Resto de decisiones de §10: 13 (caché de la lectura visual en R), 14 (`tracing`), 15 (escalas del corpus antiguo en `agents.yaml`).
7. Hacer commit de `experimentos/` y de los cambios de comentarios en `src/tools/{composicion_espacial,lineas_direccion,luz_tono}_tool.py`. Son solo comentarios (quitar «§»), hechos por el autor el 2026-09-27; sin ese commit no se puede congelar.

**B. De implementación (en este orden)**

1. **`ejecutar_lote.py`**: lanzador reanudable de T100, R×3 y V20, según el diseño de §6.4. Detalles comprobados:
   - `main.py` solo lanza una imagen fija; hay que reutilizar `construir_inputs()`;
   - `PrioridadTool` lee de `rutas.dir_ejecucion(stem)` → `outputs/ejecucion/{stem}`, así que cada ejecución se hace ahí y se copia a `outputs/experimentos/{tanda}/{rep}/{stem}/` con `registro.json`;
   - `FLAG_CRITICO_VE_IMAGEN` y `MODELO_CRITICO` son constantes de `crew.py` (V20 necesita `FLAG = False` antes de instanciar la crew: verificar que el método del agente la lee al construirse);
   - tokens por agente con `agent.llm.get_token_usage_summary()` (a verificar); VRAM con `torch.cuda.max_memory_allocated`;
   - la lectura visual (`google-genai` directo) no pasa por CrewAI.
2. **`01_calibracion_C2.ipynb`** (apoyo al capítulo 6, §7): histogramas de cada variable de decisión con su umbral (ya están en `resultados/E2/gates_C2.csv`, columnas `v_*`) y tabla de constantes. Se puede hacer ya sin anotaciones, porque no muestra aciertos por imagen. Ojo: si se cambia un umbral, las cachés se invalidan y se recalculan solas.
3. **E4a** (determinismo de la capa Python, sin API): se puede hacer ya. Ejecutar los engines 3 veces sobre R (o sobre una muestra fija mientras R no esté fijada), en el mismo proceso y en procesos distintos, y comparar los informes bit a bit. Se puede reutilizar `_cache_gates.py`.
4. **Verificador de E3 en `comun.py`**: `extraer_citas`, `resolver_ruta`, `clasificar_cita`, `CAMPOS_CONDICIONADOS`, `auditar_especialista`, `auditar_critica` y `fidelidad_transcripcion`, con las correcciones de §6.3. Para desarrollarlo sirven las ejecuciones antiguas de `outputs/ejecucion/`, que no valen como resultados. La fidelidad de transcripción puede comparar contra `gates.json`, que tiene los informes de la herramienta.
5. **E3, E4b/E4c, E5 y E9**, tras las tandas (§6.3-§6.6).
6. **E1e**, cuando exista T100 (el código ya está en E1).

**C. Orden recomendado a partir de aquí**

(autor) guía → criterios de E1 → A1 y A2 → origen y licencia → decisión §4.2-7
∥ (implementación) `ejecutar_lote.py` → `01_calibracion_C2` → E4a → verificador de E3
→ calibrar si hace falta → commit → congelar (00 §7) → E1 y E2 definitivos → tandas con API → E3/E4/E5/E9.

---

## 15. Versión acotada (2026-09-27) — MANDA SOBRE TODO LO ANTERIOR

El autor dispone de una semana y de 25 € de saldo en la API de Gemini (Google AI Studio). Se
recorta el alcance a lo esencial **sin cambiar la estructura de los capítulos 8 y 9** (siguen
E1-E4, §9.5 y §9.6): lo que se reduce es la profundidad de cada experimento.

### 15.1 Qué se mantiene y qué se retira

| Pieza | Versión acotada | Retirado respecto a §6 |
|---|---|---|
| **E1** (§9.1) | Igual: E1a-E1d + Grad-CAM. E1d con A1 (100 etiquetas de contexto). E1e si hay T100 | Nada |
| **E2** (§9.2) | Tabla 1 (hecha) + Tabla 2 con 4 columnas ciegas: `color_analizable`, `horizonte_natural`, `perspectiva`, `sin_sujeto` (+ `segmentacion` y `protagonista`, ya revisadas) | `horizontal_estructural` (y con ella la definición amplia del horizonte) y `vegetacion_dominante` (caso límite «vegetación») |
| **E3** (§9.3) | Sobre T100: citas a métricas cerradas, % de citas exactas/truncadas/alteradas, `afirmaciones` en el orden del arbitraje, salvedades | La tanda **V20** y la comparativa con/sin lectura visual |
| **E4** (§9.4) | E4b (tiempo, tokens, € por imagen, VRAM, fallos) sobre T100; E4c sobre **R = 5 imágenes × 3** | **E4a** (determinismo de la capa Python como notebook aparte); R pasa de 10 a 5 |
| §9.5 casos | 2-3 casos elegidos a partir de E2/E3, con los PNG que copia el lanzador. Sin notebook | Notebook `E5_casos_ilustrativos` |
| §9.6 resumen | La tabla pregunta → criterio → resultado se escribe en la memoria a mano con las cifras de E1-E4 | Notebook `E9_resumen_criterios` |
| Cap. 6 | Sin cambios de umbrales | Notebook `01_calibracion_C2` y la ablación de `PESO_MIN_MATIZ` |

Si el piloto (15.4, paso 5) muestra que 100 + 15 ejecuciones no caben en el presupuesto, **T100 se
reduce a 50 imágenes** (10 por contexto de A1) y R a 3 × 3; se declara en §8.6.

### 15.2 Decisiones tomadas (antes de anotar y de ver resultados)

1. **Criterios de E1 sobre C2:** exactitud sobre las aceptadas ≥ 0,85 y ≤ 20 % de C2 enviado a `otro`
   (`config_experimentos.yaml`, fijados el 2026-09-27).
2. **§4.2-7, falsos positivos conocidos:** **no** se añade el gate de relleno. Si E2 da falsos
   positivos (imagen51 y las máscaras degeneradas de §14.5 son candidatas), se informan tal cual y
   se discuten como limitación. `fp_max_por_gate: 0` no cambia.
3. **§10-13, caché de la lectura visual en R:** se mantiene (`lectura_visual: cache` en la tanda R).
   La variabilidad medida en E4c es la de los especialistas y el crítico. En T100 se regenera
   (`lectura_visual: nueva`) para medir sus tokens.
4. **§10-14, `tracing`:** apagado durante las tandas. `crew.py` tiene ahora la constante
   `TRACING = True` y el lanzador la pone a `False` antes de construir el crew.
5. **§10-15, escalas del corpus antiguo en `agents.yaml`:** actualizadas con C2 (bloque del
   especialista de líneas: «100 imágenes del corpus de desarrollo»; horizonte abierto en 26/100,
   longitud 0,137-0,752, mediana 0,21; convergencia abierta en 25/100, apertura 36°-90°, mediana
   81°; score 0,17-1,00, mediana 0,43). Se quitó la frase de los falsos positivos por paralelismo
   (≤ 30°), porque dependía de etiquetas del corpus antiguo. Las escalas de `d_equilibrio` (0,03 /
   0,12) y de `margen_patron` (0,01 / 0,15) no se tocan: en C2, 0,03 es el percentil 27 y 0,12 el 82.
6. **Origen de C2:** las 100 fotos son de **Unsplash**, con la **Unsplash License**. El manifiesto
   tiene `origen` y `licencia` rellenos; `autor` y `url` quedan vacíos (el autor no guardó los enlaces).
   En §8.3 basta con una frase sobre la fuente y el criterio de selección.
7. **«En orden» (E3)** se comprueba sobre `afirmaciones`. **«Informes idénticos» (E4)**: el informe
   transcrito por el LLM coincide con el de la herramienta (`gates.json`) y el arbitraje coincide
   entre las 3 repeticiones de R. Criterios en `config_experimentos.yaml` (`E3`, `E4`).
8. **Guía de anotación fijada** el 2026-09-27 con las 4 columnas ciegas.

### 15.3 Estado por pieza (sustituye a 14.1)

| Pieza | Estado | Nota |
|---|---|---|
| `config_experimentos.yaml` | Hecho | Criterios de E1-E4 fijados; `tandas:` (piloto, T100, R); `precios:` a `null` [pendiente del autor]; `listas.R` a `null` hasta anotar A1 |
| `comun.py` | Hecho para 00, E1, E2 y el lanzador | `ESQUEMA_A2` reducido; `COLUMNAS_HOJA`; `coste_usd(registro, precios)`. Falta el verificador de E3 |
| `anotaciones/ANOTAR_AQUI.xlsx` | Creado, **vacío** | Hoja de Excel del autor: 100 filas, 5 columnas con desplegable (`contexto` + las 4 ciegas), enlaces a la hoja ciega y a la foto, pestaña de instrucciones |
| `importar_anotaciones.py` | Hecho y probado | Pasa la hoja a `A1_contexto.csv` y a las 4 columnas ciegas de `A2_aplicabilidad.csv` (sobrescribe solo esas columnas) y avisa de valores fuera de lista |
| `anotaciones/GUIA_ANOTACION.md` | Fijada | Reducida a las 4 columnas ciegas |
| `A2_aplicabilidad.csv` | Parcial | Columnas `horizontal_estructural` y `vegetacion_dominante` eliminadas (estaban vacías). `segmentacion` 60/60 `yolo`; `protagonista` 59/60 (**falta imagen48**, `person`, en `segmentacion_02.png`) |
| `manifiesto_C2.csv` | Hecho | `origen = Unsplash`, `licencia = Unsplash License` |
| `ejecutar_lote.py` | Hecho; probado sin API (`--estado`, guardas de congelación y de precios) | **No se ha lanzado ninguna ejecución con API** |
| `crew.py`, `lectura_visual_tool.py` | Modificados | `TRACING`; la lectura visual guarda sus tokens en `outputs/lectura_visual/{stem}_lectura_uso.json`, aparte de la lectura (el crítico no los ve). Probado con la API simulada |
| `agents.yaml` | Modificado | Escalas de C2 (15.2-5) |
| `00_preparacion.ipynb` | Adaptado y probado | Sin V20; casos límite sin vegetación; propone R con 1 imagen por contexto; la congelación exige también `precios:` |
| `E1_clasificador_contexto.ipynb` | Probado | Sin cambios de código; ya lee los criterios fijados |
| `E2_condiciones_aplicabilidad.ipynb` | Adaptado y probado | Horizonte solo con `horizonte_natural` (amplia = estricta) |
| E3/E4 (un solo notebook) | **No existe** | Tras las tandas |
| `experimentos/resultados/` | Vacío a propósito | Los resultados de prueba se borran hasta congelar |

**Nada está commiteado todavía** (ni `experimentos/` ni los cambios de `src/`).

### 15.4 Qué falta, en orden

> A 2026-09-28 están hechos los pasos 1-5 (ver 15.7). Lo siguiente es el paso 6.

**Autor**

1. Anotar las 100 filas de `anotaciones/ANOTAR_AQUI.xlsx` (unas 2-3 h). Anotación ciega: no abrir
   `segmentacion_*.png` ni `resultados/` mientras tanto.
2. Al terminar: decir si la persona de la caja roja de **imagen48** (`segmentacion_02.png`) es la
   protagonista (`si` / `no` / `dudoso`); se escribe en `A2_aplicabilidad.csv`.
3. Copiar en `config_experimentos.yaml` (`precios:`) los precios vigentes de Gemini 2.5 Flash y Pro
   (USD por millón de tokens, entrada y salida), la fecha de consulta y el tipo de cambio USD/EUR.

**Implementación (con el asistente)**

4. `importar_anotaciones.py` → ejecutar 00 (valida A1/A2 y propone R) → pegar `listas.R`.
5. **Piloto de coste:** `ejecutar_lote.py piloto` (imagen1 con saliencia, imagen6 con YOLO; cuesta
   menos de 1 €; no necesita congelación). Con sus tokens y los precios, decidir T100 = 100 o 50
   (15.1). Si las 115 ejecuciones superan ~20 € (80 % del saldo), se reduce.
6. Commit de todo → congelar (00 §7, `CONGELAR = True`) → commit de `congelacion.json`.
7. E1 y E2 definitivos (sin API).
8. `ejecutar_lote.py T100 --tope-eur 20` (una noche; reanudable) y después `ejecutar_lote.py R --tope-eur 20`.
9. Verificador de E3 en `comun.py` (reglas de §6.3) y un único notebook E3/E4 con sus tablas y figuras.
10. Elegir 2-3 casos para §9.5 y redactar §9.6 a mano.

### 15.5 Lanzador (`experimentos/ejecutar_lote.py`)

- **Uso:** `.venv/Scripts/python.exe experimentos/ejecutar_lote.py {piloto|T100|R} [--tope-eur X] [--estado] [--forzar]`.
- **Salida:** `outputs/experimentos/{tanda}/rep{r}/{stem}/`, con las 6 `salida_*.json`, la lectura
  visual y su uso de tokens, los PNG de verificación de los especialistas, `log.txt` (la salida
  verbosa de CrewAI) y `registro.json`.
- **`registro.json`:** estado (`ok` / `error` / `incompleta`), intento, segundos, marca de fin de
  cada tarea (aproximada, porque el Nivel 2 va en paralelo), `primera_del_proceso` (la primera
  imagen incluye la carga de YOLO y la ResNet: excluirla de la latencia), tokens por instancia de
  LLM (`prompt`, `completion` con el razonamiento incluido, `reasoning`, `cached`, `requests`),
  tokens de la lectura visual, VRAM pico, error y datos del sistema (commit, congelado, modelos,
  flag, tracing, versión del prompt visual) y `coste_usd_estimado`.
- **Coste:** `comun.coste_usd` no descuenta la caché implícita de Gemini; es una cota superior.
- **Guardas:** las tandas T100 y R exigen congelación (salvo `--forzar`); `--tope-eur` exige los
  precios; se detiene ante errores de cuota o facturación y tras 3 errores seguidos; salta las
  imágenes ya `ok`, así que basta con relanzar el mismo comando para reanudar.
- **E1e** lee `outputs/experimentos/T100/*/imagen*/salida_orquestador.json`: la estructura
  `rep1/imagenN` del lanzador encaja.

### 15.7 Avance del 2026-09-28

- **Anotación terminada:** A1 100/100 y las 4 columnas ciegas de A2 100/100, importadas desde la hoja
  sin valores inválidos; `protagonista` de imagen48 = `si`. 00 da A1 y A2 completas y 0 revisiones
  obsoletas.
- **Precios fijados** (consulta del 2026-09-28, nivel de pago estándar, USD por millón de tokens):
  Flash 0,30 entrada / 2,50 salida; Pro 1,25 / 10,00 (tramo ≤ 200.000). `usd_por_eur: 1.0` como cota
  conservadora.
- **R fijada:** `imagen62, imagen9, imagen35, imagen86, imagen50` (una por contexto; 4 `yolo` y 1
  `saliencia`).
- **Presupuesto real:** el panel de AI Studio tiene un límite mensual de 30 € con 7,95 € gastados a
  28-09 antes del piloto (quedan ~22 € hasta el 1 de octubre, cuando se reinicia). El panel tarda
  hasta 24 h en actualizarse.
- **Piloto (sistema sin congelar; no vale como resultado):** imagen1 (saliencia) e imagen6 (yolo), 2/2
  `ok`, con las 6 salidas.

  | | imagen1 | imagen6 |
  |---|---|---|
  | Tiempo total | 129 s (incluye la carga de modelos) | 91 s |
  | Tokens LLM (entrada / salida con razonamiento) | ~97k / ~23k | ~97k / ~19k |
  | Crítico (Pro), entrada / salida | 23.100 / 5.267 | 23.337 / 4.437 |
  | Lectura visual (Flash), entrada / salida | 768 / 699 | 768 / 829 |
  | Coste estimado (cota superior) | 0,149 USD | 0,134 USD |
  | VRAM pico | 356 MB | 346 MB |

  Cada agente hace 2 llamadas al LLM. El crítico es ~60 % del coste. **Estimación para T100 + R (115
  ejecuciones): ~16 USD (~14 €)**, dentro de los ~22 € disponibles: se mantiene **T100 = 100** y R = 5 × 3.
  Tope del lanzador: `--tope-eur 20`.
- Detalle cosmético: el panel final de CrewAI («Crew Completion») sale por la consola aunque el resto
  va a `log.txt`, porque Rich guarda la salida original. No afecta a los datos.

### 15.6 Amenazas nuevas para §8.6

- Anotación reducida: sin `horizontal_estructural`, el horizonte solo se evalúa con la definición
  estricta; sin `vegetacion_dominante`, ese caso límite no se describe.
- R con 5 imágenes: la reproducibilidad es indicativa.
- Sin V20, no se mide el efecto de la lectura visual sobre las citas del crítico.
- Si T100 se reduce a 50 por presupuesto, E3 y E4b se calculan sobre 50.
