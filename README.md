# Sistema Multiagente para Análisis de Composición Fotográfica

Trabajo de Fin de Grado — Grado en Ingeniería en Computadores, Universidad de Alcalá.
**Autor:** Edison Javier Tayan Gualoto · **Tutora:** Sira Elena Palazuelos Cagigas.

Sistema **multiagente jerárquico** construido sobre [CrewAI](https://crewai.com) que analiza
la composición fotográfica de una imagen y produce una crítica estructurada, razonada y
verificable. En lugar de un LLM generalista de "caja negra", cuatro especialistas emiten
**métricas cuantitativas verificables** (regla de los tercios, líneas y punto de fuga, espacio
y aislamiento del sujeto, luz y tono) calculadas con visión por computador clásica
(OpenCV/NumPy/scikit-learn), y un agente crítico (LLM) se limita a **verbalizarlas y
arbitrarlas**, citando cada afirmación con su métrica: `afirmación [métrica = valor]`.

La visión por computador corre **local en GPU**; la síntesis del crítico usa la API de
**Google Gemini** en la nube.

## Arquitectura

- **Nivel 1 — Orquestador**: clasifica el contexto de la escena (ResNet18 fine-tuneado, 5
  clases + "otro") y calcula una percepción compartida (YOLO + mapa de saliencia espectral)
  que localiza el sujeto principal, si lo hay.
- **Nivel 2 — Cuatro especialistas**, ortogonales entre sí: composición espacial, líneas y
  dirección, espacio y aislamiento del sujeto, luz y tono. Cada uno emite un `informe`
  recomputable (autoría del engine) y un `diagnostico` en prosa (autoría del LLM, citando el
  informe).
- **Nivel 3 — Crítico**: recibe los cuatro diagnósticos + un vector de pesos contextuales W
  (qué dimensión importa más según el tipo de escena) y sintetiza una crítica final,
  arbitrando tensiones y citando siempre su fuente.


## Requisitos

- Python `>=3.10,<3.14`.
- [uv](https://docs.astral.sh/uv/) como gestor de dependencias.
- GPU con CUDA recomendada (el clasificador y YOLO corren en local); funciona también en CPU.
  El segmentador SAM del especialista de espacio corre siempre en CPU, a propósito (ver Modelos).
- Una clave de API de Google Gemini.

## Instalación

```bash
pip install uv
uv sync
```

Crear un fichero `.env` en la raíz con la clave de Gemini:

```
GEMINI_API_KEY=tu_clave_aqui
```

### Modelos

Hacen falta tres artefactos:

- `models/yolo/yolo11m.pt` — YOLO11-medium (Ultralytics), no incluido en el
  repositorio; debe descargarse y colocarse en esa ruta.
- `models/sam/sam2.1_b.pt` — SAM 2.1 base (Ultralytics), el segmentador del especialista de
  espacio y aislamiento. No se incluye en el repositorio, pero **se descarga solo** a esa ruta
  (154 MB) la primera vez que se analiza una imagen con sujeto.
- `models/clasificador_contexto/pesos_fine_tuning_ligero.pt` — ResNet18 fine-tuneado sobre
  el dataset EVA para clasificar el contexto de la escena; se incluye junto con
  `class_to_idx.json` y `umbral.json`.

Sin YOLO ni el clasificador el sistema no arranca. Contactar con el autor si se necesitan
para evaluación del TFG.

MD5 de referencia, para comprobar que se usan exactamente los mismos pesos:

| Fichero | MD5 |
|---|---|
| `models/yolo/yolo11m.pt` | `2c2bcbb54c3829b20d31a32524ee0f03` |
| `models/sam/sam2.1_b.pt` | `e57666d5a782119e02af8f0de6713e32` |

**Reproducibilidad de la segmentación.** La máscara de SAM no tiene azar, pero sí depende del
entorno: cambia ligeramente entre versiones de `ultralytics` (del 0,1 al 0,3 % de los píxeles
entre 8.4.102 y 8.4.202) y, en algún píxel suelto, entre CPU y GPU (medido con MPS; con CUDA
no se ha comprobado). Por eso el especialista
fija `DEVICE_SAM = "cpu"` y las versiones están congeladas en `uv.lock`: **no ejecutar
`uv lock --upgrade`** sin recalibrar los umbrales de `espacio_aislamiento_tool.py`.

## Ejecución

**Interfaz gráfica (vía normal, y la de la demo):**

```bash
uv run streamlit run app.py
```

Se sube una foto, se pulsa "Analizar" y, tras unos minutos, aparece la crítica junto a los
seis paneles de verificación visual de cada etapa del análisis.

**Desde terminal, sobre una imagen fija** (pensado para desarrollo; la ruta se edita dentro
de `main.py`):

```bash
crewai run
```

## Estructura del repositorio

```
app.py                          # Interfaz Streamlit
src/tfg_multiagente_fotografia/
  crew.py, main.py, rutas.py    # Orquestación del crew y layout de outputs
  config/                       # agents.yaml + tasks.yaml (prompts de los 6 agentes)
  parametros/                   # W contextual y modelo nulo del punto de fuga, versionados
  schemas/                      # Contratos Pydantic de cada nivel
  tools/                        # Los engines de visión por computador de cada agente
models/                         # Artefactos del clasificador de contexto, YOLO y SAM (ver arriba)
data/                           # Corpus de imágenes de prueba (no versionado)
outputs/                        # Salidas regenerables (ignoradas por Git)
tests/                          # Pruebas manuales de los componentes
docs/                           # Diagramas, evidencia y documentos aprobados del TFG
```


## Tests

No hay un runner único: cada script de `tests/` se ejecuta a mano y sale por consola
(código 1 si falla algo). Por ejemplo:

```bash
uv run python tests/test_composicion_espacial_tool.py
uv run python tests/test_lineas_direccion_tool.py
uv run python tests/test_espacio_aislamiento_tool.py
uv run python tests/test_luz_tono_tool.py
```

El de espacio y aislamiento tarda unos 4-5 minutos: pasa SAM dos veces por cada imagen de
`data/` para comprobar que el análisis es determinista.
