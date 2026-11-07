# Estado de la validación experimental

Resumen para retomar el trabajo en una conversación nueva: qué está hecho, qué decisiones se
tomaron y qué toca después. El plan completo y las justificaciones están en
[`../experimentos.md`](../experimentos.md); este fichero solo dice **dónde estamos**.

**Última actualización:** 2026-11-07 · **Rama:** `agente3-sam`

---

## 1. Dónde estamos

```
Fase 1  Anotar el conjunto de prueba ........... HECHO
Fase 2  Calibrar umbrales sobre data/ .......... EN CURSO (script listo, falta ejecutarlo en Windows)
Fase 3  Instrumentos + congelación ............. pendiente  → aquí queda el sistema final
Fase 4  Ejecutar E1–E4 y escribir resultados ... pendiente
```

**Siguiente acción concreta:** en el equipo Windows, ejecutar `experimentos/calibracion.py`
(`medir` → `plantilla` → revisar → `exportar`) y subir los tres CSV/JSON resultantes (§4).

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
| Script de calibración (medir + revisión ciega de la segmentación) | `calibracion.py` | «Script de calibración…» (2026-11-06) |

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
| 2.1 | Usuario (Windows) | `git pull`, `uv sync`, `uv run python experimentos/calibracion.py medir` (~4 s/foto) |
| 2.2 | Usuario | `... calibracion.py plantilla`; revisar en `anotaciones/segmentacion_calibracion.xlsx` la segmentación (`bien`/`parcial`/`mal`) y si la caja es el protagonista (`si`/`no`) de las ~60 fotos `yolo`; `... calibracion.py exportar` |
| 2.3 | Usuario | Commit y push de `resultados/calibracion/medidas_calibracion.csv`, `resultados/calibracion/entorno_calibracion.json` y `anotaciones/segmentacion_calibracion.csv` |
| 2.4 | Asistente | Análisis de huecos por umbral, con gráficas: cruzar `medidas_calibracion.csv` con `resultados/aplicabilidad_vlm/anotacion_humana_A2.csv` (aplicabilidad de las 100 fotos, ya anotada) y con la revisión de la segmentación. Umbrales: `SAM_SCORE_MIN`, `RELLENO_MIN/MAX`, `VAR_LAPLACIANO_MIN`, `RATIO_CROMATICO_MIN`, `LONGITUD_MIN_HORIZONTE`/`ANGULO_MAX_HORIZONTE` (con el horizonte amplio), `APERTURA_MIN_GRADOS`, barrido de `CONF_YOLO_THRESHOLD`, percentiles de `margen_patron` y `d_equilibrio` para `agents.yaml` |
| 2.5 | Usuario | Aceptar o ajustar los valores propuestos |
| 2.6 | Asistente | Aplicarlos: quitar «PROVISIONAL» del código y actualizar los comentarios, `agents.yaml` y la Figura 28 (`docs/diagramas/espacio_aislamiento_pipeline.html`); ejecutar `tests/` |

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
- El usuario consulta y apunta las tarifas vigentes de Gemini.
- `congelacion.json` (checklist del §9 del plan) y commit con el árbol limpio. **A partir de ese
  commit no se toca el código.**

### Fase 4 · Experimentos (Windows)

1. Sin API: E1, E2 y el determinismo de la capa Python. Después, anotación de segmentación y
   protagonista de las fotos `yolo` del conjunto de prueba (no puede ser ciega).
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
| `../TFG-EdisonJavierTayanGualoto_Aux.docx` | Memoria auxiliar (sin versionar); el cap. 9 tiene los resultados de la primera campaña |
| `calibracion.py` | Fase 2 |
| `plantilla_anotacion_prueba.py` | Plantilla y exportación de la anotación del conjunto de prueba |
| `anotaciones/anotacion_prueba.csv` | Verdad de referencia para E1 (contexto) y E2 (aplicabilidad) |
| `resultados/aplicabilidad_vlm/anotacion_humana_A2.csv` | Aplicabilidad anotada de las 100 fotos de calibración |
| `aplicabilidad_vlm.py`, `evaluar_decisores.py`, `plantilla_revision.py` | Línea exploratoria del VLM (trabajo futuro); `leer_anotaciones` se reutiliza |
| `src/tfg_multiagente_fotografia/tools/espacio_aislamiento_tool.py` | Umbrales provisionales de SAM (líneas ~55–85) |
