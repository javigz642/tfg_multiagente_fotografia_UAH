# Estado de la validación experimental

Resumen para retomar el trabajo en una conversación nueva: qué está hecho, qué decisiones se
tomaron y qué toca después. El plan completo y las justificaciones están en
[`../experimentos.md`](../experimentos.md); este fichero solo dice **dónde estamos**.

**Última actualización:** 2026-11-07 · **Rama:** `agente3-sam`

---

## 1. Dónde estamos

```
Fase 1  Anotar el conjunto de prueba ........... HECHO
Fase 2  Calibrar umbrales sobre data/ .......... HECHA (medición final e55d0dc, con el código de e3d4bfe)
Fase 3  Instrumentos + congelación ............. SIGUIENTE  → aquí queda el sistema final
Fase 4  Ejecutar E1–E4 y escribir resultados ... pendiente
```

**BLOQUEO (2026-11-07): 4 fotos de `data/` son ficheros distintos en el Mac y en Windows**
(imagen48, 61, 63 y 81: otras dimensiones y otro MD5). La anotación de aplicabilidad se hizo sobre
las versiones del Mac (las miniaturas de `plantilla_revision_humana.xlsx` lo confirman), que son
además las descargas originales de Unsplash; la calibración midió las de Windows. Por tanto, esas
4 filas cruzan medidas de una foto con la anotación de otra. También explica la diferencia de YOLO
61/60 entre máquinas del §2.3 del plan (es imagen61, no el hardware). Estimación con las versiones
del Mac: color, convergencia y nitidez no cambian; la regla del horizonte pasaría a dar (0.1517; 14°).
**Decisión del usuario:** igualar las dos máquinas combinando versiones: imagen61 e imagen63 con la
versión del Mac (la anotada) e imagen48 e imagen81 con la de Windows (ya medidas así). El
manifiesto recoge esa combinación. La anotación de 48 y 81 parece seguir valiendo para la versión
de Windows (el nombre del protagonista de imagen48, «chica con vestido», ya no la describe); el
usuario lo confirma.

**Siguiente acción (usuario, Windows): paso 2.10.** `git pull`; `calibracion.py medir --rehacer`;
`calibracion.py plantilla --sobrescribir --conservar --revisar imagen63` (imagen61 sale en blanco
sola, porque antes no era yolo); revisar las filas en blanco; `calibracion.py exportar`; commit y
push solo de `experimentos/resultados/calibracion/` y `experimentos/anotaciones/segmentacion_calibracion.csv`.
Después el asistente vuelve a aplicar la regla (2.11): el horizonte probablemente cambie.

Hecho en la fase 3: `manifiesto_calibracion.py` y su CSV (88 fotos idénticas a su descarga de
Unsplash, imagen54 derivada a B/N, 11 sin original localizado) y `calibracion.py --conjunto prueba`
(comprobación de MD5 contra el manifiesto, caché ligada al código de `src/`, `exportar` más
estricto y `plantilla --conservar --revisar`).

---

## 2. Convenciones (no cambiarlas sin hablarlo)

- **Nombres de los conjuntos** (no usar C2, C3, T40, R, A1/A2 como nombres de conjunto o anotación):
  - **conjunto de calibración** = `data/imagen{1..100}.jpg` → solo para ajustar umbrales;
  - **conjunto de prueba** = `data_prueba/test{1..40}.jpg` → todos los resultados;
  - **EVA-test / EVA-val / EVA-other** = particiones de EVA (fuera del repo) para E1;
  - **fotos de repetición** = imagen62, imagen9, imagen35, imagen86, imagen50 (de `data/`);
  - **tanda de prueba** = las 40 ejecuciones con API; **anotación de contexto / de aplicabilidad**.
  - Los códigos A1…O5 solo son las filas del diseño del conjunto de prueba (§5.1 del plan).
- **Una sola máquina** para todo lo que produce resultados o umbrales: el equipo **Windows con
  RTX 4060**. El Mac sirve para desarrollar y probar con pocas fotos.
- **No se versionan**: `data/`, `data_prueba/`, `outputs/`, `experimentos/anotaciones/*.xlsx`.
  Sí se versionan los CSV exportados y los resultados en `experimentos/resultados/`.
- **Commits:** solo cuando el usuario lo pide; se suben a `agente3-sam`.
- El usuario prefiere explicaciones en castellano, paso a paso, y diciendo qué hace él y qué hace
  el asistente.

---

## 3. Lo que está hecho

| Qué | Dónde | Commit |
|---|---|---|
| Selección y descarga del conjunto de prueba (40 fotos de Unsplash, P2 girada 15° a propósito) | `descargar_prueba.py`, `manifiesto_prueba.csv`, `seleccion_prueba.json` | `4d39696` |
| Revisión del plan: criterios de éxito con tolerancias, horizonte amplio como principal, nombres descriptivos | `experimentos.md` §6.0 y §6.2 | `06c6a60` |
| Anotación ciega de contexto y aplicabilidad del conjunto de prueba | `anotaciones/anotacion_prueba.csv` (generada con `plantilla_anotacion_prueba.py`) | `a9f65ec` |
| Script de calibración (medir + revisión ciega de la segmentación) | `calibracion.py` | `e6d8d06` |
| Regla para elegir cada umbral, fijada antes de medir | `experimentos.md` §4.1 | `25998ad` |
| Medición en Windows y revisión de la segmentación (pasos 2.1–2.3) | `resultados/calibracion/`, `anotaciones/segmentacion_calibracion.csv` | `575306d` |

### 3.1 Qué salió de la medición (2026-11-07, Windows)

- **Entorno:** ASUS-JAVIER, Windows 11, RTX 4060 Laptop, Python 3.12.10, torch 2.6.0+cu124, CUDA
  12.4, ultralytics 8.4.102. Se midió con el commit `38845ad`, sin cambios en `src/`
  (`entorno_calibracion.json`).
- **Percepción:** 100 fotos medidas; **60 con `fuente = yolo`** y 40 con saliencia (coincide con lo
  ya visto en Windows, §2.3 del plan).
- **Revisión de la segmentación (60 fotos `yolo`):** **59 `bien`, 1 `parcial` (imagen95, coche) y
  0 `mal`.**
  - Score de SAM entre 0,921 y 0,998; relleno entre 0,240 y 0,895. Ninguna foto queda fuera de los
    valores provisionales (0,90 y [0,20; 0,95]), así que la condición de segmentación fiable
    **nunca cierra** en el conjunto de calibración.
  - Consecuencia prevista: con 0 negativas, a `SAM_SCORE_MIN`, `RELLENO_MIN` y `RELLENO_MAX` les toca
    la **regla 3 del §4.1** («comprobado, no calibrado»): se mantienen los valores actuales y se
    declara que el conjunto de calibración no tiene segmentaciones `mal` con las que ajustarlos. Hay
    que confirmarlo en el 2.4 y decirlo así en el cap. 6 y en los límites (§8 del plan).
- **¿Caja = protagonista?:** 49 `si` y **11 `no`**. Las 11 son casos en los que YOLO elige un objeto
  que no es el protagonista (imagen77 libro, imagen90 jarrón, imagen93 persona con caballo, imagen99
  vela con humo…; ver los comentarios del CSV). Importa para la condición de composición, que abre
  con `fuente = yolo` sin comprobar si la caja es el protagonista: en el 2.4 se cuentan como falsos
  positivos de composición y se usan en el barrido de `CONF_YOLO_THRESHOLD`.

### 3.2 Qué salió del análisis de huecos (paso 2.4, 2026-11-07)

Script `analisis_calibracion.py` (sin GPU; lee los tres CSV). Salidas en `resultados/calibracion/`:
`propuesta_umbrales.csv`, `barrido_yolo.csv`, `escalas_composicion.csv` y `huecos_*.png`.

| Umbral | Actual | Regla §4.1 | VP/FP/FN/VN | Tasa FP (actual) | Sensib. (actual) |
|---|---|---|---|---|---|
| `RATIO_CROMATICO_MIN` | 0.09 | **0.17** (regla 2) | 76/2/0/21 | 9 % (13 %) | 100 % (100 %) |
| Horizonte (L, A) | 0.13, 10° | **0.09, 0°** (4→2) **degenerado** | 18/4/14/62 | 6 % (12 %) | 56 % (53 %) |
| ↳ alternativa A (A fijo en 10°) | | 0.1517, 10° | 17/5/15/61 | 8 % | 53 % |
| ↳ alternativa B (A ≥ 10°) | | 0.1517, 14° | 18/6/14/60 | 9 % | 56 % |
| `APERTURA_MIN_GRADOS` | 35 | **89** (regla 2) | 4/3/10/32 | 9 % (40 %) | 29 % (64 %) |
| `SAM_SCORE_MIN`, `RELLENO_MIN/MAX` | 0.90; 0.20–0.95 | sin cambios (regla 3) | 60/0/0/0 | — | 100 % |
| `VAR_LAPLACIANO_MIN` | 50 | **2594** (regla 2) | 10/1/38/11 | 8 % (92 %) | 21 % (94 %) |
| ↳ alternativa (solo caja = protagonista) | | 1100 | 18/0/26/5 | 0 % (80 %) | 41 % |

- Con los valores actuales, cromático, horizonte, convergencia y nitidez **superan el 10 % de FP**
  de la regla, así que ninguno se puede mantener tal cual sin apartarse de ella.
- **Horizonte:** la regla elige ángulo máximo 0° (28 de 86 candidatos miden exactamente 0°). Con eso
  `angulo_horizonte` valdría siempre 0 al abrir: la métrica ya no podría decir que una foto está
  torcida. Las alternativas A y B no son la regla y habría que declararlas.
- **Convergencia:** la apertura no separa perspectiva de no perspectiva (AUC 0.64). Una perspectiva
  con líneas de un solo lado da un abanico estrecho (imagen4: 48.9°) y unas pocas diagonales de una
  fachada frontal que se cruzan por azar dan 80–90° (imagen8, imagen6). Con 89° la condición casi
  no abre (precisión 4/7 en calibración). `score / score_nulo` separaría mejor (AUC 0.91): queda
  como trabajo futuro, no se rediseña tras ver los datos. El mensaje de cierre por apertura ya no
  dice «líneas casi paralelas» (corregido antes de congelar).
- **Nitidez:** 7 de las 12 negativas lo son porque la caja de YOLO no es el protagonista
  (imagen6, 8, 15, 77, 85, 86, 99); este umbral no puede cerrarlas sin cerrar casi todo.
- **SAM:** confirmado «comprobado, no calibrado» (0 segmentaciones `mal`).
- **YOLO (solo documentar):** subir el umbral no separa las cajas equivocadas: con 0.90 quedan
  44 si / 6 no (pierde 5 buenas para quitar 5 malas). Por debajo de 0.75 entrarían 14 fotos sin revisar.
- **Composición:** las 11 fotos con caja ≠ protagonista abren la condición → 11 FP en calibración.
- **Escalas de composición:** `margen_patron` (fotos yolo) P10 = 0.012, P90 = 0.133 (0.15 actual es
  el P95); `d_equilibrio` (las 100) P10 = 0.019, P90 = 0.142 (0.03 / 0.12 actuales son P27 / P82). El
  plan no fija qué percentiles usar: lo decide el usuario.
- **`BLOB_*`:** no se activan (ratio entre 0.010 y 0.405; ninguna foto sin sujeto).

### Decisiones tomadas con el usuario

1. **Criterios de éxito revisados antes de abrir el conjunto de prueba** (tabla en §6.0 del plan).
   Lo heurístico y lo que hace el LLM se exigen con tolerancia (1 de cada 10 o de cada 20); solo la
   capa Python determinista se exige al 100 %. Motivo: con 40 fotos, «0 errores» no es medible, y
   un LLM no puede dar el 100 %. **No se relajan después de ver resultados.**
2. **E2 se juzga con la precisión de apertura** (VP / (VP + FP)): global ≥ 90 %, ≥ 80 % por
   condición y sensibilidad global ≥ 60 %.
3. **Horizonte:** la definición **amplia** (natural **o** horizontal estructural) es la principal
   y la estrecha (solo natural) es un análisis secundario.
4. **E1 (c):** el % enviado a «otro» se calcula sobre las 35 fotos de las cinco clases, no sobre 40.
5. **La revisión de la segmentación es ciega a las cifras:** sus imágenes muestran la caja y la
   máscara, pero no el score de SAM, el relleno ni si el sistema la consideró fiable.
6. **Regla para elegir cada umbral fijada antes de medir** (§4.1 del plan, 2026-11-07). Con banda
   vacía, el corte va en su extremo más exigente. Sin banda vacía, el corte de mayor sensibilidad entre
   los que dejan abrir como mucho el 10 % de las negativas. Con menos de 3 positivas o negativas,
   «comprobado, no calibrado».
7. **La segmentación del conjunto de prueba se anota igual que la de calibración** (mismo script,
   imágenes sin cifras) y **antes de mirar las decisiones del sistema** (§5.3 y §6.2 del plan). La
   convergencia sin veredicto propio y la segmentación no calibrada se declaran en los límites (§8).

### Estado de la anotación del conjunto de prueba

- Hecha a ciegas: sin haber abierto el manifiesto y sin pasar ninguna foto por el sistema. **Ahora
  el manifiesto ya se puede abrir.**
- **No modificar `anotacion_prueba.csv`** salvo errores evidentes de transcripción, y declarándolo.
- Detalles menores aceptados: test9 y test33 tienen `protagonista = no` pero conservan el nombre;
  el comentario de test38 dice «dudoso» aunque el valor es `no`. No afectan a ninguna métrica.
- Cobertura por condición (si / no / dudoso): color 30/9/1 · horizonte amplio 14/24/2 · horizonte
  natural 6/33/1 · perspectiva **4**/36/0 · espacio 23/15/2 · nitidez 25/14/1 · anclaje 23/17/0.
  La perspectiva está en el mínimo de 4 positivos, y eso se declara en los límites.

---

## 4. Lo que hay que hacer (en orden)

### Fase 2 · Calibración sobre `data/`

| # | Quién | Qué |
|---|---|---|
| 2.1 | Usuario (Windows) | ~~Medir~~ HECHO (2026-11-07) |
| 2.2 | Usuario | ~~Revisar la segmentación y el protagonista de las 60 fotos `yolo` y exportar~~ HECHO |
| 2.3 | Usuario | ~~Commit y push de los tres ficheros~~ HECHO (`575306d`) |
| 2.4 | Asistente | ~~Análisis de huecos con la regla del §4.1 del plan~~ HECHO (§3.2) |
| 2.5 | Usuario | ~~Aceptar o ajustar los valores propuestos~~ HECHO: acepta la recomendación (abajo) |
| 2.6 | Asistente | ~~Aplicarlos y ejecutar `tests/`~~ HECHO |
| 2.7 | Usuario (Windows) | ~~`calibracion.py medir --rehacer`~~ HECHO (`fa5afc3`, medido con `8678fa4`) |
| 2.8 | Asistente | ~~Comprobar las decisiones~~ HECHO: variables idénticas a la medición anterior; las 6 condiciones coinciden foto a foto con la propuesta (color 78, horizonte 22, convergencia 7, espacio 60, nitidez 11, composición 60) |

| 2.9 | Usuario (Windows) | ~~Re-medir por el mensaje de cierre corregido~~ HECHO (`e55d0dc`): solo cambian los 44 textos de `motivo_convergencia` que cierran por apertura; variables y decisiones idénticas |

**Ficheros de la primera campaña:** el commit `fa5afc3` subió además 67 ficheros de la primera
campaña (notebooks, `comun.py`, `config_experimentos.yaml`, `ejecutar_lote.py`, `congelacion.json`,
`resultados/E1`–`E5`, anotaciones `A1`/`A2`…). Se movieron con `git mv` a
`experimentos/primera_campana/` (ver su `LEEME.md`), sin borrarlos.

**Mensaje de cierre de la convergencia corregido:** decía «líneas casi paralelas», que es falso con
el corte en 89° (imagen4, una perspectiva real de 48.9°, cerraba con ese texto). Ahora dice que con
un abanico menor no se distingue una perspectiva de paralelas o de diagonales que se cruzan por
azar. También se ajustaron el comentario de `APERTURA_MIN_GRADOS`, el docstring del gate, la
descripción de `apertura_haz` en el esquema y el prompt de `agents.yaml`.

**Valores aceptados (2.5, 2026-11-07):** `RATIO_CROMATICO_MIN` 0.17; horizonte con la alternativa A
(`LONGITUD_MIN_HORIZONTE` 0.1517, `ANGULO_MAX_HORIZONTE` 10, desviación declarada); `APERTURA_MIN_GRADOS`
89; `VAR_LAPLACIANO_MIN` 2594; SAM sin cambios; escalas de composición P10/P90 (margen 0.01/0.13,
`d_equilibrio` 0.02/0.14); `CONF_YOLO_THRESHOLD` y `BLOB_*` sin cambios. Resumen en el §4.2 del plan.

**Cómo hacer el 2.4:**

- **Entradas (todas en el repo):**
  - `resultados/calibracion/medidas_calibracion.csv`: una fila por foto con todas las variables de
    decisión y si cada condición abre (`gate_*`);
  - `anotaciones/segmentacion_calibracion.csv`: `segmentacion` y `caja_protagonista` de las 60
    fotos `yolo`;
  - `resultados/aplicabilidad_vlm/anotacion_humana_A2.csv`: aplicabilidad de las 100 fotos
    (`color`, `horizonte_natural`, `horizontal_estructural`, `perspectiva`, `protagonista`,
    `espacio`, `nitidez`, `anclaje`). Mezcla codificación y separador: leerla con `leer_csv()` de
    `calibracion.py`. El horizonte amplio es `si` si lo es el natural **o** el estructural.
- **Qué producir:** un script `experimentos/analisis_calibracion.py` (versionado), que escriba en
  `resultados/calibracion/`:
  - una gráfica por umbral (fotos ordenadas por su variable, positivas y negativas en colores
    distintos, el corte actual y el propuesto);
  - `propuesta_umbrales.csv`: umbral, valor actual, valor propuesto, regla del §4.1 aplicada
    (1, 2, 3 o 4), VP/FP/FN/VN en calibración y las fotos que fallan.
- **Umbrales por huecos:** los de la tabla del §4.1 del plan (`RATIO_CROMATICO_MIN`, horizonte con la
  definición amplia, `APERTURA_MIN_GRADOS`, `SAM_SCORE_MIN` + `RELLENO_MIN/MAX`,
  `VAR_LAPLACIANO_MIN`).
- **Otros (no por huecos):** barrido de `CONF_YOLO_THRESHOLD` (usar `conf_yolo_max` y
  `caja_protagonista`; ojo, YOLO no da cajas por debajo de 0,25), percentiles de `margen_patron` y
  `d_equilibrio` para las escalas de `agents.yaml`, y `BLOB_*` (comprobar que no se activan).
- **Además, informar** de la condición de composición: cuántas de las 11 fotos con
  `caja_protagonista = no` abren composición (son falsos positivos de esa condición).
- **No tocar todavía el código:** primero se presentan los valores al usuario (2.5).

**Cómo hacer el 2.6:** quitar «PROVISIONAL» del código y actualizar los comentarios que citan cifras
(`espacio_aislamiento_tool.py` líneas ~55–85, y las demás herramientas si cambia algún valor),
`agents.yaml` y la Figura 28 (`docs/diagramas/espacio_aislamiento_pipeline.html`); ejecutar
`tests/`. Si algún umbral cambia, volver a medir en Windows (`calibracion.py medir --rehacer`) para
que `medidas_calibracion.csv` refleje el sistema final.

Mejoras menores de `calibracion.py`, solo si se vuelve a usar: `exportar` debería comprobar que
están todas las fotos `yolo` de `medidas_calibracion.csv` y no fallar con la hoja vacía; la caché no
guarda el commit (tras cambiar código hay que usar `--rehacer`).
**Alcance: todos los agentes, no solo SAM.** (a) Nuevos por SAM: `SAM_SCORE_MIN`,
`RELLENO_MIN/MAX` y `VAR_LAPLACIANO_MIN` (depende de la máscara); son los únicos que necesitan la
revisión manual de la segmentación. (b) Ya calibrados antes, pero sobre un `data/` más pequeño o
con el horizonte estrecho, así que se vuelven a comprobar con las 100 fotos y la anotación que ya
existe: cromático (hay casos a 0,086/0,093/0,099 del corte), horizonte (ahora con la definición
amplia), apertura de la convergencia (la condición más débil), barrido de YOLO y escalas de
composición. (c) Solo se documentan: modelo nulo, fusión de matices, margen de frontera, `BLOB_*`
(no ejercitados), `umbral_otro` y W. La EVALUACIÓN (E1–E4) cubre el sistema entero sobre el
conjunto de prueba; los resultados de la primera campaña no cuentan como finales.

Regla de calibración: método de huecos con la regla del §4.1 del plan (qué hacer con banda vacía,
sin ella y con pocos casos), fijada antes de medir. Un falso positivo rompe la verificabilidad; un
falso negativo solo hace callar. **El conjunto de prueba no se usa para calibrar.**

### Fase 3 · Instrumentos y congelación (el sistema final)

- `manifiesto_calibracion.csv` (nombre, MD5, dimensiones, URL, autor de las 100 fotos de `data/`).
- `config_experimentos.yaml`, `comun.py` (verificador de citas, IC de Wilson…), `ejecutar_lote.py`
  (lanzador reanudable con tokens, tiempos, VRAM/RAM y copia por tanda). Detalle en el §7 del plan.
- Añadir a `calibracion.py` la opción `--conjunto prueba`: leer `data_prueba/` y escribir en
  `resultados/prueba/`, `outputs/prueba/` y `anotaciones/segmentacion_prueba.{xlsx,csv}`, sin pisar
  los ficheros de calibración. Solo cambian las rutas; el resto del script es igual (§5.3 y §7 del
  plan). Se aprovecha para hacer las mejoras menores de la fase 2.
- El usuario consulta y apunta las tarifas vigentes de Gemini.
- `congelacion.json` (checklist del §9 del plan) y commit con el árbol limpio. **A partir de ese
  commit no se toca el código.**

### Fase 4 · Experimentos (Windows)

1. Sin API. **Primero**, el usuario anota la segmentación y el protagonista de las fotos `yolo` del
   conjunto de prueba con `calibracion.py --conjunto prueba` (`medir` → `plantilla` → `exportar`),
   **sin abrir** `resultados/prueba/medidas_*.csv`: así la anotación no se contamina con las
   decisiones del sistema. **Después**, E1, E2 y el determinismo de la capa Python.
2. Con API: tanda de prueba (40) y repeticiones (5 × 3), unos 7,4 USD.
3. E3, E4, casos ilustrativos, tabla resumen (§9.6) y cambios en la memoria (§11 del plan).

---

## 5. Referencia: la primera campaña (antes de SAM, sobre `data/`)

Sirve para anticipar resultados; **no son resultados de esta campaña**.

- E1: F1 mínima 0,769 (producto); 90/90 aceptadas correctas; 10 % a «otro». Cumplía.
- E2 (horizonte estrecho): precisión de apertura global ≈ 84 %. Por condición: cromática 77/78,
  horizonte 9/26, convergencia 12/25 (48 %), espacio 54/59, nitidez 49/54, composición 51/59.
  La mayoría de los FP de horizonte eran horizontales estructurales (cornisas, fachadas).
- E3: 99,92 % de citas exactas; 2 de 3.580 citas a métricas cerradas (ninguna del crítico).
- E4: 59/60 informes fieles; arbitraje igual en 4 de 5 fotos; ~0,13 USD y ~98 s por análisis.

---

## 6. Ficheros clave

| Fichero | Para qué |
|---|---|
| `../experimentos.md` | Plan completo (qué se mide, criterios, guía de anotación, congelación) |
| `../TFG-EdisonJavierTayanGualoto_Aux.docx` | Memoria auxiliar (versionada desde `38845ad`); el cap. 9 tiene los resultados de la primera campaña |
| `calibracion.py` | Fase 2 |
| `plantilla_anotacion_prueba.py` | Plantilla y exportación de la anotación del conjunto de prueba |
| `anotaciones/anotacion_prueba.csv` | Verdad de referencia para E1 (contexto) y E2 (aplicabilidad) |
| `resultados/aplicabilidad_vlm/anotacion_humana_A2.csv` | Aplicabilidad anotada de las 100 fotos de calibración |
| `aplicabilidad_vlm.py`, `evaluar_decisores.py`, `plantilla_revision.py` | Línea exploratoria del VLM (trabajo futuro); `leer_anotaciones` se reutiliza |
| `src/tfg_multiagente_fotografia/tools/espacio_aislamiento_tool.py` | Umbrales provisionales de SAM (líneas ~55–85) |
