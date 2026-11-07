# Guía de anotación de C2 (A1 y A2)

> **Estado: FIJADA** en la versión acotada (experimentos.md §15). Las definiciones no se cambian
> después de ver resultados: si se cambiaran, las tasas de E1 y E2 dejarían de ser una medida y
> pasarían a ser un ajuste.
>
> Fecha de fijación: 2026-09-27

Las anotaciones son la verdad de referencia de E1 (contexto) y E2 (condiciones de aplicabilidad).
Se anota **la foto**, no lo que el sistema dice de ella. Por eso el orden importa:

1. **Primero, las columnas ciegas**, en `anotaciones/ANOTAR_AQUI.xlsx` (pestaña «Anotar»), mirando
   solo la foto (`outputs/experimentos/hojas_revision/ciega_*.png` o la foto a tamaño completo).
   No mires antes las salidas del sistema ni las hojas `segmentacion_*.png`.
2. **Después**, `experimentos/importar_anotaciones.py` pasa la hoja a `A1_contexto.csv` y
   `A2_aplicabilidad.csv`, que son los que leen los notebooks.
3. Las columnas de segmentación (`segmentacion`, `protagonista`) ya estaban revisadas con la hoja
   de segmentación; solo falta `protagonista` de imagen48, que se revisa al final.

**Valores.** `si` / `no` / `dudoso` (sin tilde; `sí` también se acepta). Usa `dudoso` poco: en
E2, las filas `dudoso` se excluyen de la matriz 2×2 y se cuentan aparte.

**Versión acotada.** Para reducir la anotación se retiraron `horizontal_estructural` y
`vegetacion_dominante` (y con ellas la definición amplia del horizonte y el caso límite
«vegetación»). Se declara en §8.6.

---

## A1 · Contexto verdadero (columna `contexto`)

Una etiqueta por foto, según el **contenido protagonista** y no según el estilo:

| Etiqueta | Cuándo |
|---|---|
| `animal` | El protagonista es uno o varios animales |
| `arquitectura` | El protagonista es un edificio, una estructura o un espacio construido (interior o exterior) |
| `paisaje` | El protagonista es un entorno natural o rural abierto (montaña, mar, campo, cielo) |
| `producto-still_life` | El protagonista es un objeto o una composición de objetos (bodegón, producto, comida) |
| `retrato-humano` | El protagonista es una o varias personas |
| `otro` | No hay un protagonista de las cinco clases, o hay varios sin que ninguno predomine |

Casos mixtos: gana el elemento que la foto convierte en protagonista (tamaño, foco, posición). Una
persona pequeña en un paisaje es `paisaje`; un edificio en un paisaje donde el edificio manda es
`arquitectura`. Si no puedes decidirlo, `otro` (y anótalo en `notas`).

Nota: el `other` de EVA no es exactamente este `otro` (§5.1.2 de la memoria).

---

## A2 · Aplicabilidad de las reglas

### Columnas ciegas (solo con la foto)

| Columna | Pregunta | `si` | `no` |
|---|---|---|---|
| `color_analizable` | ¿La foto tiene color suficiente para hablar de un esquema cromático? | Hay matices reconocibles, aunque sean pocos | Blanco y negro o virado monocromo, nocturna casi sin color o muy desaturada |
| `horizonte_natural` | ¿Hay una línea de horizonte (cielo con tierra o mar) legible? | Se ve y cruza al menos un tercio del ancho, aunque esté parcialmente tapada | No hay, o solo se intuye. Una mesa o una fachada no cuentan |
| `perspectiva` | ¿La profundidad se organiza con líneas que convergen hacia un punto, visible o no? | Calles, pasillos, vías, fachadas en escorzo | Escena plana, líneas paralelas al plano de la foto o profundidad sugerida solo por tamaños o por la atmósfera |
| `sin_sujeto` | ¿La foto **no** tiene un sujeto discreto protagonista? | Paisaje abierto, textura, patrón o escena sin un elemento principal | Hay un elemento principal que se podría recortar |

### Columnas de segmentación (con la hoja de segmentación)

Son obligatorias solo en las imágenes con **fuente = yolo** en la percepción actual, que son las
únicas en las que los gates de espacio y de composición pueden abrirse.

| Columna | Pregunta | Valores |
|---|---|---|
| `segmentacion` | ¿La máscara (cian) rodea al sujeto de la caja (roja)? | `ok`: lo rodea; `parcial`: le falta o le sobra una parte importante; `mal`: vacía, en otro sitio o casi todo el encuadre |
| `protagonista` | ¿El objeto de la caja roja es el protagonista de la foto? | `si` / `no` / `dudoso` |

`fuente_revisada` y `clase_revisada` guardan la fuente y la clase de YOLO que viste al revisar. Si
la percepción cambia (otro umbral, otra imagen), el notebook detecta la diferencia y pide revisar
de nuevo esa fila. **No las edites a mano**, salvo para actualizarlas cuando revises una fila.

### Revisiones que ya existían

Las 70 revisiones de segmentación de `notebooks/revision_espacio_aislamiento.csv` se han migrado
tal cual, con sus notas. La fila de `imagen61` se ha vaciado porque la imagen se sustituyó el
2026-09-26 y la revisión anterior era de la imagen duplicada (con la percepción actual va por
saliencia, así que no necesita revisión). `patron_percibido` (de
`revision_composicion_espacial.csv`) va en `patron_percibido.csv`. No es una anotación de
aplicabilidad y queda fuera de la versión acotada.

---

## Cómo se usan en E2: «la regla aplica» por gate (§8.5.2)

En todos los gates:

- **falso positivo** (grave): la métrica está abierta y la regla no aplica;
- **falso negativo** (tolerable): la métrica está cerrada y la regla aplica.

Lo que cambia de un gate a otro es cuándo **aplica** la regla. Hay dos definiciones, estricta y
amplia; `config_experimentos.yaml` (`E2.definicion_criterio: estricta`) fija cuál decide el
criterio y la otra solo se informa.

| Gate | Aplica (estricta) | Aplica (amplia) |
|---|---|---|
| `esquema_cromatico` | `color_analizable = si` | igual |
| `angulo_horizonte` | `horizonte_natural = si` | igual (versión acotada) |
| `score_convergencia` | `perspectiva = si` | igual |
| `ratio_espacio_negativo`, `ratio_nitidez` | `sin_sujeto = no` y, si la fuente es `yolo`, `segmentacion ≠ mal` | `sin_sujeto = no` y, si la fuente es `yolo`, `segmentacion = ok` |
| composición (`d_tercios`, `d_centro`, `patron_dominante`) | `sin_sujeto = no` y, si la fuente es `yolo`, `protagonista = si` (el anclaje es el centro de la caja, no la máscara) | igual |

Con fuente `saliencia` la máscara y la caja no las ha fijado YOLO, así que `segmentacion` y
`protagonista` no intervienen. La regla aplica si hay un sujeto discreto (`sin_sujeto = no`), y en
ese caso el cierre es un falso negativo: típicamente, un sujeto que COCO no reconoce (un edificio,
un árbol, una joya).

Si el resultado depende de algún `dudoso`, la imagen se excluye de la matriz 2×2 de ese gate y se
cuenta aparte. `ratio_nitidez` usa la misma definición que `ratio_espacio_negativo`, porque no se
anota si la escena tiene textura (limitación declarada en E2).

### Casos límite de C2 (§8.3)

Se derivan de A2, sin columnas nuevas:

- **B/N o sin color:** `color_analizable = no`.
- **Sin sujeto:** `sin_sujeto = si`.
- **Horizonte:** `horizonte_natural = si`.
- **Perspectiva:** `perspectiva = si`.
