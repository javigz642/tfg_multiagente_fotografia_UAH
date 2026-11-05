# Campaña 2: validación del sistema con SAM y un conjunto de prueba nuevo

Plan de la segunda campaña experimental, con el sistema que segmenta con SAM 2.1. **No contiene
resultados**: fija qué se mide, sobre qué y con qué criterio **antes** de ejecutar, como el capítulo 8.

- **Fecha del plan:** 2026-11-05. **Rama:** `agente3-sam`.
- **Relación con lo anterior:** la campaña 1 (2026-09-28, capítulos 8 y 9, plan en `experimentos.md`)
  evaluó el sistema con GrabCut usando las mismas 100 fotos para calibrar y para medir. Este documento
  solo recoge lo que cambia; el resto de los protocolos, instrumentos y criterios se hereda.
- **Convención:** **[verificado]** comprobado en el código o ejecutando algo; **[pendiente]** falta
  hacerlo.

---

## 1. Decisiones de la campaña 2

1. **El sistema que se valida es el que segmenta con SAM.** No se hace un experimento comparativo entre
   GrabCut y SAM: el cambio se menciona brevemente en la memoria (§8), no se investiga.
2. **C2 (las 100 fotos de Unsplash) pasa a ser el conjunto de calibración y desarrollo.** Todos los
   umbrales se ajustan con él.
3. **Un conjunto nuevo, C3, de 40 fotos que el autor no ha visto, es el conjunto de prueba.** Los
   resultados del capítulo 9 se miden sobre C3. Esto elimina la amenaza «umbrales ajustados sobre el
   mismo corpus» de §8.6.
4. **Toda la campaña se ejecuta en una sola máquina**, el equipo Windows con CUDA de la campaña 1.
   Además, **las versiones están unificadas en todas las plataformas**: `pyproject.toml` fija
   `torch==2.6.0` y `torchvision==0.21.0` (antes el Mac instalaba torch 2.14) [verificado: con 2.6.0
   en el Mac pasan los tests de los 4 especialistas, la percepción, el clasificador y la parte
   determinista del crítico].
   - Unificar versiones no basta para que dos máquinas den exactamente lo mismo: con torch 2.6.0 en las
     dos, YOLO cuenta 61 sujetos en C2 en el Mac y 60 en la campaña 1 en Windows [verificado], porque
     el cálculo en GPU y en CPU (y en ARM frente a x86) no es idéntico bit a bit. Por eso la regla es
     una sola máquina, y la versión única es una garantía adicional.

## 2. Qué se mide y sobre qué conjunto

| Experimento | Campaña 1 | Campaña 2 |
|---|---|---|
| E1 Clasificador | EVA (prueba, validación, other) + C2 | **EVA: se heredan los resultados** (el clasificador no cambia). Lo que dependía de C2 se mide sobre **C3** |
| E2 Aplicabilidad | 6 condiciones sobre C2 | **6 condiciones sobre C3**, mismo criterio (0 falsos positivos) y misma Tabla 15 |
| E3 Citas | T100 sobre C2 | **T40 sobre C3**, mismos criterios |
| E4 Coste y latencia | T100 | **T40 sobre C3** |
| E4 Reproducibilidad | 5 fotos de C2 × 3 | **Las mismas 5 fotos de C2 × 3** (`imagen62`, `9`, `35`, `86`, `50`): la reproducibilidad no depende de la calibración y así se compara con la campaña 1 |

- **Coste estimado con API:** (40 + 15) análisis × 0,134 USD ≈ **7,4 USD**, con la tarifa de la campaña
  1 y sin descontar la caché. Hay que volver a consultar las tarifas antes de lanzar [pendiente].
- **Cambios en los instrumentos:**
  - El verificador de citas trata `score_segmentacion`, `relleno_bbox` y `segmentacion_fiable` como
    evidencia del gate, igual que `confianza`: citarlos no es citar una métrica cerrada.
  - La fidelidad de transcripción compara 10 campos en el Agente 3, uno booleano y dos que pueden ser
    `null`.
  - E4 añade la **RAM máxima** del proceso, porque SAM corre en CPU y no aparece en la VRAM.
- **Con 40 imágenes los intervalos de confianza son más anchos** que con 100. Es el precio de que el
  resultado sea honesto, y se dice en §8.6.

## 3. Calibración sobre C2

Se hace **antes** de abrir C3. Para el Agente 3:

| Parámetro | Valor provisional | Cómo se fija |
|---|---|---|
| `SAM_SCORE_MIN` | 0,90 | Método de huecos sobre la distribución del score en las fotos `yolo` (escala log), como `VAR_LAPLACIANO_MIN` |
| `RELLENO_MIN` / `RELLENO_MAX` | 0,20 / 0,95 | Ídem sobre el relleno; revisar los objetos rectangulares que rozan el techo |
| `VAR_LAPLACIANO_MIN` | 50 | Recalibrar: se fijó con máscaras de GrabCut |
| `MARGEN_FRONTERA_RATIO` | 0,01 | Repetir el análisis de sensibilidad con SAM |

Si se recalibra algún umbral de otro especialista, se hace también aquí y sobre C2. Después:
1. quitar la etiqueta «PROVISIONAL» del código;
2. actualizar la Figura 28, que muestra 0,90 / 0,20 / 0,95 (`docs/diagramas/espacio_aislamiento_pipeline.html`);
3. ejecutar `tests/`;
4. congelar (§6).

## 4. C3: el conjunto de prueba

### 4.1 Composición (40 fotos)

Misma lógica que la Tabla 13: proporciones parecidas a C2 (7 por categoría y 5 para «otro»), y en cada
categoría casos en los que cada condición **debería abrirse** y casos en los que **debería cerrarse**.
Además se refuerzan los dos puntos débiles que mostró la campaña 1: la elección del sujeto y la
segmentación difícil.

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
| O4 | Textura que llena todo el encuadre (hojas, arena, un patrón) | Rama `sin_sujeto_claro`, que **no se dio nunca en C2** |
| O5 | Escena sin dominante (persona y animal a la par, un mercado) | Contexto ambiguo |

**Comprobación de cobertura.** Antes de cerrar la lista, contar cuántas fotos debería abrir y cuántas
debería cerrar cada condición. Se busca **al menos 4 de cada lado** en cromática, horizonte,
convergencia, identidad y textura. Si una foto no cumple lo que se esperaba de ella, **se queda igual**:
el subtipo es un objetivo de selección, no una etiqueta. La etiqueta real la da la anotación ciega.

### 4.2 Reglas de selección

- **Fuente:** Unsplash, con la misma licencia que C2 y, si es posible, de fotógrafos distintos.
- **Una búsqueda por fila de la tabla:** se coge la **primera foto que cumpla claramente la
  descripción**. No se elige la que «parece que va a salir bien».
- **Ninguna foto de C3 pasa por el sistema, la app ni los tests** hasta congelar. Se guardan en
  `data_c3/`, no en `data/`: los tests recorren `data/` y la contaminarían. `data_c3/` está en
  `.gitignore`.
- **No se sustituye ninguna foto después de ver resultados.** Si una resulta inválida (corrupta o
  duplicada de C2), se sustituye con la misma regla y se declara.
- **Variedad** de orientación y resolución parecida a C2 (de 2 a 26 MP).
- **Manifiesto** `experimentos/manifiesto_C3.csv`: nombre (`test1`…`test40`), fila de la tabla, URL,
  autor, MD5, dimensiones y fecha de descarga.

### 4.3 Anotación de C3

Con las mismas guías que la campaña 1:

1. **A1 (contexto) y A2 (aplicabilidad), a ciegas**, antes de ejecutar nada.
2. **Segmentación** (`bien` / `parcial` / `mal`) de las fotos con `fuente = yolo`, sobre las imágenes de
   verificación del Agente 3, **después** de A2. Ninguna anotación puede ser ciega a la máscara, pero A2
   sí lo es.

## 5. Criterios de éxito

Los de la Tabla 14, sin cambios, aplicados a C3. No se relajan: cambiar un criterio después de ver los
resultados de la campaña 1 invalidaría la comparación.

## 6. Congelación (añadir a la lista de `experimentos.md`, §11)

- [ ] Commit con el árbol limpio, con los umbrales ya calibrados y sin la etiqueta «PROVISIONAL».
- [ ] MD5 de `models/sam/sam2.1_b.pt` (`e57666d5a782119e02af8f0de6713e32`), de `models/yolo/yolo11m.pt`
      y del manifiesto de C3.
- [ ] Máquina usada y versiones (`torch 2.6.0+cu124`, `ultralytics 8.4.102`).
- [ ] Lista de C3 cerrada y anotaciones A1 y A2 de C3 hechas.

## 7. Orden de trabajo

1. Elegir y descargar C3 en `data_c3/` y crear el manifiesto. No abrir las fotos en el sistema.
2. Anotar A1 y A2 de C3 a ciegas.
3. Calibrar sobre C2 (§3), actualizar el código y la Figura 28, y ejecutar `tests/`.
4. Congelar (§6).
5. Sin API: E1 y E2 sobre C3. Anotar la segmentación de las fotos `yolo` de C3.
6. Con API, en la máquina Windows: T40 sobre C3 y R × 3.
7. E3 y E4. Actualizar la memoria (§8).

## 8. Cambios en la memoria

| Dónde | Qué cambia |
|---|---|
| 6.x Agente 3 | Cálculo con SAM y relleno; tercera condición de aplicabilidad; número de campos. **Figura 28**: sustituirla por `docs/diagramas/espacio_aislamiento_pipeline.png` |
| 4.x Tecnologías | SAM 2.1 base en CPU, con la justificación de reproducibilidad y la cita (Ravi et al., 2024 [X]) |
| 8.3 Conjuntos (Tablas 12 y 13) | C2 pasa a calibración; C3 como conjunto de prueba, con su composición (§4.1) |
| 8.6 Límites | Desaparece «umbrales ajustados sobre el mismo corpus»; se añaden «C3 tiene 40 imágenes» y «una sola máquina» |
| 9 | Todos los resultados que dependían de C2 pasan a C3; Figuras 35 y 36 regeneradas |
| 9.7 y 10 | «Condición de relleno de la máscara» y «segmentación de propósito general» dejan de ser trabajo futuro |

**Mención breve de SAM.** Un párrafo en el apartado del Agente 3 basta. Propuesta para adaptar:

> En la primera versión, la máscara del sujeto se obtenía con GrabCut inicializado con la caja de YOLO.
> Durante la validación se comprobó que este método puede fallar sin dar ninguna señal: en una imagen
> devolvió una máscara vacía y, aun así, las dos métricas se publicaron con confianza 1. Por ello se
> sustituyó por SAM 2.1 [X], un modelo de segmentación que recibe la misma caja como indicación y que,
> además de la máscara, estima su propia calidad. Esa estimación, junto con la fracción de la caja que
> ocupa la máscara, da lugar a una tercera condición de aplicabilidad, que cierra las dos métricas
> cuando la segmentación no es fiable. No se ha hecho una comparación sistemática entre los dos métodos.

La última frase es necesaria: sin un experimento comparativo, la memoria no puede afirmar que SAM
segmenta mejor en general, solo que corrige el fallo observado y añade una señal de calidad.
