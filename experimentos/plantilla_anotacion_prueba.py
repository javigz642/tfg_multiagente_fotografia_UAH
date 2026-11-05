"""Anotación ciega del conjunto de prueba: contexto y aplicabilidad (experimentos.md, §5.3).

Dos subcomandos:

    plantilla   Genera el Excel de anotación sobre `data_prueba/`: una fila por foto, con su
                miniatura, un enlace a la imagen completa, la etiqueta de contexto y las ocho
                preguntas de aplicabilidad (desplegables). No pisa una plantilla ya rellenada.
    exportar    Lee el Excel rellenado, comprueba que está completo y coherente, deriva el
                horizonte amplio y escribe `anotacion_prueba.csv` (utf-8, separador ','),
                que es lo que se congela (§9) y lo que leen los experimentos.

La plantilla es CIEGA a propósito. Solo lee las fotos `data_prueba/test*.jpg`: no abre el
manifiesto, `seleccion_prueba.json` ni `data_prueba/_seleccion/`, que revelan la fila del
diseño (y por tanto la categoría y lo que se esperaba de cada foto). Tampoco muestra
ninguna decisión del sistema: ninguna foto del conjunto de prueba ha pasado todavía por él.

Las preguntas tienen las mismas claves que `plantilla_revision.py` (perspectiva, anclaje…),
así que el CSV también lo lee `aplicabilidad_vlm.leer_anotaciones`. La anotación de
segmentación y protagonista (fotos con fuente = yolo) va aparte y se hace DESPUÉS, porque
necesita las imágenes de verificación del sistema.

Uso (desde la raíz del proyecto):
    uv run python experimentos/plantilla_anotacion_prueba.py plantilla
    uv run python experimentos/plantilla_anotacion_prueba.py exportar
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image as ImagenExcel
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plantilla_revision import (  # noqa: E402
    ALTO_FILA_PT, AMBAR, AZUL, BLANCA, BORDE, ROJO, VACIO, VERDE, miniatura, numero_imagen,
)

RAIZ = Path(__file__).resolve().parents[1]
DIR_PRUEBA = RAIZ / "data_prueba"
DIR_ANOTACIONES = RAIZ / "experimentos" / "anotaciones"
PLANTILLA = DIR_ANOTACIONES / "anotacion_prueba.xlsx"
CSV_SALIDA = DIR_ANOTACIONES / "anotacion_prueba.csv"
N_ESPERADO = 40

# Etiquetas literales del sistema (class_to_idx.json + rechazo), para cruzar sin traducir.
CONTEXTOS = [
    ("animal", "El tema de la foto es uno o varios animales."),
    ("arquitectura", "El tema es un edificio, un interior o una escena urbana construida "
                     "(EVA: «architecture and city scenes»)."),
    ("paisaje", "El tema es una escena natural o rural (EVA: «natural and rural scenes»)."),
    ("producto-still_life", "El tema es uno o varios objetos inanimados: producto o bodegón "
                            "(EVA: «still life»)."),
    ("retrato-humano", "El tema es una o varias personas (EVA: «human»)."),
    ("otro", "La foto no pertenece claramente a ninguna de las cinco clases, o contenidos de "
             "clases distintas tienen el mismo peso y ninguno domina."),
]
ETIQUETAS_CONTEXTO = [c for c, _ in CONTEXTOS]

# (clave, título corto, métrica que decide, SÍ aplica cuando…, NO aplica cuando…)
# Definiciones operativas de la guía de aplicabilidad (experimentos.md, §6.2), en la
# redacción afinada sobre el conjunto de calibración (PROMPT_V2 de aplicabilidad_vlm.py).
PREGUNTAS = [
    ("color", "¿Color analizable?", "esquema_cromatico",
     "La imagen ENTERA tiene color: el color saturado ocupa una parte sustancial del "
     "encuadre, aunque domine un solo matiz (una foto azulada o anaranjada SÍ tiene esquema).",
     "Blanco y negro; escena dominada por blancos, grises, negros o tonos apagados con el "
     "color reducido a acentos pequeños; dominante tenue tipo sepia o virado desaturado."),
    ("horizonte_natural", "¿Horizonte natural?", "angulo_horizonte (definición estrecha)",
     "Hay un límite lejano casi recto donde termina la tierra o el agua contra el cielo y "
     "cruza buena parte del encuadre. Puede estar inclinado o cortado por el sujeto.",
     "Lo que toca el cielo es un perfil irregular de montañas, una arboleda cercana o "
     "edificios; no se ve cielo; o la única horizontal es de un objeto."),
    ("horizontal_estructural", "¿Horizontal estructural?", "angulo_horizonte (parte de la amplia)",
     "Una línea REALMENTE horizontal de la escena (cornisa, borde de mesa, peldaño, línea "
     "entre pared y mesa, pretil, orilla) se ve de frente, es continua y cruza MÁS DE LA "
     "MITAD del ancho. Responde con independencia de la pregunta anterior.",
     "Solo hay líneas que se alejan en perspectiva (barandillas, bordes de un pasillo, "
     "aceras que convergen), bordes cortos de objetos o líneas curvas o diagonales."),
    ("perspectiva", "¿Perspectiva que organiza?", "score_convergencia",
     "Líneas RECTAS reales (calles, raíles, pasillos, aristas de edificios) convergen hacia "
     "un punto de fuga y estructuran toda la imagen.",
     "Las líneas son curvas (carretera que gira, sendero); la foto la organiza un sujeto y "
     "la convergencia queda en el fondo; o solo hay paralelas, texturas o vegetación."),
    ("protagonista", "¿Hay protagonista?", "(apoyo para las tres siguientes)",
     "Hay un sujeto concreto y delimitado que es claramente el centro de atención y la "
     "razón de la foto (persona, animal, objeto, edificio, árbol aislado). Nómbralo en la "
     "columna siguiente.",
     "El tema es el paisaje o el lugar y el elemento es pequeño dentro de él; bodegón o "
     "mesa con varios objetos de peso parecido; patrón o masa."),
    ("espacio", "¿Medir espacio negativo?", "ratio_espacio_negativo",
     "Hay protagonista con un BORDE reconocible contra el fondo. También si es grande (un "
     "retrato cercano): la métrica dirá que hay poco espacio libre, y esa medida es válida.",
     "No hay protagonista; es amorfo y sin borde (humo, fuegos artificiales); se confunde "
     "con el fondo (camuflaje); o es un grupo de objetos."),
    ("nitidez", "¿Comparar nitidez?", "ratio_nitidez",
     "Hay protagonista y AL MENOS UNA de las dos regiones (sujeto o fondo) tiene textura "
     "fina visible: pelo, piel, plumas, tejido, hojas, grano, letras. Basta con una.",
     "No hay protagonista; es amorfo; o sujeto y fondo son ambos lisos (cerámica, "
     "porcelana, plástico, cielo, pared: una superficie lisa enfocada NO es detalle)."),
    ("anclaje", "¿Evaluar su posición?", "d_tercios, d_centro, patron_dominante",
     "El protagonista es UN elemento con un centro definido cuya posición en el encuadre "
     "(tercios o centro) tiene sentido evaluar.",
     "No hay protagonista; es un grupo de objetos; o es amorfo o está repartido por el "
     "encuadre."),
]
DECISIONES = [p[0] for p in PREGUNTAS]
DEPENDIENTES = ("espacio", "nitidez", "anclaje")
VALORES = {"si", "no", "dudoso"}

COLUMNAS = (
    [("imagen", "Imagen", 9), ("miniatura", "Miniatura", 23), ("abrir", "Abrir", 8),
     ("contexto", "Contexto", 19)]
    + [(c, titulo, 14) for c, titulo, *_ in PREGUNTAS[:5]]
    + [("protagonista_nombre", "Nombre del protagonista", 20)]
    + [(c, titulo, 14) for c, titulo, *_ in PREGUNTAS[5:]]
    + [("comentario", "Comentario (opcional)", 40), ("estado", "Estado", 12)]
)
LETRA = {c: get_column_letter(j) for j, (c, _, _) in enumerate(COLUMNAS, 1)}


def fotos_prueba(carpeta: Path) -> list[Path]:
    """Solo `test*.jpg` de la carpeta, sin entrar en subcarpetas (`_seleccion/`)."""
    rutas = sorted(
        (p for p in carpeta.iterdir()
         if p.is_file() and p.stem.startswith("test") and p.suffix.lower() in {".jpg", ".jpeg"}),
        key=lambda p: (numero_imagen(p), p.name),
    )
    if len(rutas) != N_ESPERADO:
        print(f"AVISO: {len(rutas)} fotos en {carpeta}, se esperaban {N_ESPERADO}.")
    return rutas


def desplegable(ws, opciones: list[str], titulo: str, ayuda: str) -> DataValidation:
    dv = DataValidation(
        type="list", formula1='"' + ",".join(opciones) + '"', allow_blank=True,
        showDropDown=False, errorTitle="Valor no válido",
        error="Elige un valor del desplegable.", promptTitle=titulo, prompt=ayuda,
    )
    ws.add_data_validation(dv)
    return dv


def hoja_anotacion(ws, rutas: list[Path], destino: Path) -> int:
    definicion = {c: (si, no) for c, _, _, si, no in PREGUNTAS}
    guia_contexto = "\n".join(f"{c}: {d}" for c, d in CONTEXTOS)

    # Fila 1: nombre de máquina (lo que lee `exportar`). Fila 2: título legible.
    for j, (clave, titulo, ancho) in enumerate(COLUMNAS, 1):
        ws.cell(row=1, column=j, value=clave).font = Font(color="808080", size=9)
        cab = ws.cell(row=2, column=j, value=titulo)
        cab.fill, cab.font, cab.border = AZUL, BLANCA, BORDE
        cab.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = ancho
        if clave in definicion:
            si, no = definicion[clave]
            cab.comment = Comment(f"SÍ: {si}\n\nNO: {no}", "Guía", width=340, height=240)
        elif clave == "contexto":
            cab.comment = Comment(guia_contexto, "Guía", width=380, height=260)
    ws.row_dimensions[2].height = 42

    dv_decision = desplegable(ws, ["si", "no", "dudoso"], "Respuesta",
                              "si / no / dudoso (la definición está en la cabecera)")
    dv_contexto = desplegable(ws, ETIQUETAS_CONTEXTO, "Contexto",
                              "Una de las seis etiquetas (definiciones en la cabecera)")

    for i, ruta in enumerate(rutas, 3):
        ws.row_dimensions[i].height = ALTO_FILA_PT
        ws.cell(row=i, column=1, value=ruta.stem)
        enlace = ws.cell(row=i, column=3, value="abrir")
        # Ruta relativa al Excel: el enlace sobrevive si se mueve el proyecto entero.
        enlace.hyperlink = os.path.relpath(ruta, destino.parent)
        enlace.font = Font(color="0563C1", underline="single")
        ws.add_image(ImagenExcel(miniatura(ruta)), f"{LETRA['miniatura']}{i}")

        obligatorias = [f"{LETRA[c]}{i}" for c in ["contexto", *DECISIONES]]
        rango = ",".join(obligatorias)
        prot = f"{LETRA['protagonista']}{i}"
        incoherente = ",".join(f'{LETRA[c]}{i}="si"' for c in DEPENDIENTES)
        ws.cell(row=i, column=len(COLUMNAS), value=(
            f'=IF(COUNTA({rango})=0,"pendiente",'
            f'IF(COUNTA({rango})<{len(obligatorias)},"incompleta",'
            f'IF(AND({prot}="no",OR({incoherente})),"revisar","completa")))'
        ))
        for clave, _, _ in COLUMNAS:
            celda = ws[f"{LETRA[clave]}{i}"]
            celda.border = BORDE
            centrada = clave in DECISIONES or clave == "contexto"
            celda.alignment = Alignment(vertical="center", wrap_text=True,
                                        horizontal="center" if centrada else None)
            if clave in DECISIONES:
                dv_decision.add(celda)
            elif clave == "contexto":
                dv_contexto.add(celda)

    ultima = len(rutas) + 2
    zona = f"{LETRA['color']}3:{LETRA['anclaje']}{ultima}"
    ws.conditional_formatting.add(zona, CellIsRule(operator="equal", formula=['"si"'], fill=VERDE))
    ws.conditional_formatting.add(zona, CellIsRule(operator="equal", formula=['"no"'], fill=ROJO))
    ws.conditional_formatting.add(zona, CellIsRule(operator="equal", formula=['"dudoso"'], fill=AMBAR))
    ws.conditional_formatting.add(zona, FormulaRule(formula=[f"LEN({LETRA['color']}3)=0"], fill=VACIO))
    ctx = f"{LETRA['contexto']}3:{LETRA['contexto']}{ultima}"
    ws.conditional_formatting.add(ctx, FormulaRule(formula=[f"LEN({LETRA['contexto']}3)=0"], fill=VACIO))
    estado = f"{LETRA['estado']}3:{LETRA['estado']}{ultima}"
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"completa"'], fill=VERDE))
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"revisar"'], fill=ROJO))
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"incompleta"'], fill=AMBAR))

    ws.freeze_panes = f"{LETRA['contexto']}3"
    ws.auto_filter.ref = f"A2:{LETRA['estado']}{ultima}"
    return ultima


def hoja_guia(ws) -> None:
    ws["A1"] = "Guía de anotación del conjunto de prueba (contexto y aplicabilidad)"
    ws["A1"].font = Font(bold=True, size=14)
    instrucciones = [
        "1. Abre cada foto completa con «abrir» antes de decidir: la miniatura sirve para "
        "orientarse, no para juzgar líneas finas ni texturas.",
        "2. Contexto: elige la etiqueta del TEMA de la foto, no la de un elemento suelto. "
        "«otro» se usa si no encaja en ninguna de las cinco o si ninguna domina.",
        "3. Aplicabilidad: responde si / no / dudoso por lo que VES en la escena, no por lo "
        "que crees que mediría el sistema. Que una regla aplique no significa que la foto "
        "la cumpla bien (un horizonte torcido sigue siendo un horizonte).",
        "4. Usa «dudoso» solo si de verdad no puedes decidir: esas fotos quedan fuera de los "
        "falsos positivos y negativos y se cuentan aparte.",
        "5. Coherencia: si no hay protagonista, espacio, nitidez y anclaje deben ser «no». "
        "La columna Estado marca «revisar» si no se cumple.",
        "6. Horizonte: se anotan por separado el natural y el estructural. La definición "
        "amplia (la principal en E2) es «si» cuando cualquiera de los dos lo es; no se anota, "
        "se deriva al exportar.",
        "7. Anotación ciega: no abras el manifiesto, seleccion_prueba.json ni "
        "data_prueba/_seleccion/ hasta exportar. Si alguna vez lo ves, apúntalo en Comentario.",
        "8. No cambies la fila 1 de la hoja Anotación: son los nombres que lee el script.",
        "9. Al terminar: uv run python experimentos/plantilla_anotacion_prueba.py exportar",
    ]
    for k, texto in enumerate(instrucciones, 3):
        ws.cell(row=k, column=1, value=texto).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=k, start_column=1, end_row=k, end_column=4)
        ws.row_dimensions[k].height = 34

    fila = len(instrucciones) + 5
    for j, cab in enumerate(["Etiqueta de contexto", "Cuándo se usa"], 1):
        celda = ws.cell(row=fila, column=j, value=cab)
        celda.fill, celda.font, celda.border = AZUL, BLANCA, BORDE
    for k, (etiqueta, texto) in enumerate(CONTEXTOS, fila + 1):
        ws.cell(row=k, column=1, value=etiqueta).border = BORDE
        celda = ws.cell(row=k, column=2, value=texto)
        celda.alignment, celda.border = Alignment(wrap_text=True, vertical="top"), BORDE
        ws.merge_cells(start_row=k, start_column=2, end_row=k, end_column=4)
        ws.row_dimensions[k].height = 30

    fila = fila + len(CONTEXTOS) + 2
    for j, cab in enumerate(["Columna", "Métrica que decide", "SÍ aplica cuando…",
                             "NO aplica cuando…"], 1):
        celda = ws.cell(row=fila, column=j, value=cab)
        celda.fill, celda.font, celda.border = AZUL, BLANCA, BORDE
    for k, (clave, titulo, metrica, si, no) in enumerate(PREGUNTAS, fila + 1):
        for j, valor in enumerate([f"{clave}\n{titulo}", metrica, si, no], 1):
            celda = ws.cell(row=k, column=j, value=valor)
            celda.alignment, celda.border = Alignment(wrap_text=True, vertical="top"), BORDE
        ws.row_dimensions[k].height = 80
    for col, ancho in zip("ABCD", (26, 26, 60, 60)):
        ws.column_dimensions[col].width = ancho


def hoja_progreso(ws, ultima: int) -> None:
    ws["A1"] = "Progreso de la anotación"
    ws["A1"].font = Font(bold=True, size=14)
    estado = f"Anotación!${LETRA['estado']}$3:${LETRA['estado']}${ultima}"
    for k, valor in enumerate(["completa", "incompleta", "revisar", "pendiente"], 3):
        ws.cell(row=k, column=1, value=valor)
        ws.cell(row=k, column=2, value=f'=COUNTIF({estado},"{valor}")')

    # Recuento por pregunta: sirve para la comprobación de cobertura del §5.1
    # (al menos 4 fotos a cada lado en cada condición).
    fila = 9
    for j, cab in enumerate(["Pregunta", "si", "no", "dudoso", "sin responder"], 1):
        celda = ws.cell(row=fila, column=j, value=cab)
        celda.fill, celda.font = AZUL, BLANCA
    for k, clave in enumerate(DECISIONES, fila + 1):
        rango = f"Anotación!${LETRA[clave]}$3:${LETRA[clave]}${ultima}"
        ws.cell(row=k, column=1, value=clave)
        for j, valor in enumerate(["si", "no", "dudoso"], 2):
            ws.cell(row=k, column=j, value=f'=COUNTIF({rango},"{valor}")')
        ws.cell(row=k, column=5, value=f"=COUNTBLANK({rango})")

    fila = fila + len(DECISIONES) + 2
    for j, cab in enumerate(["Contexto", "fotos"], 1):
        celda = ws.cell(row=fila, column=j, value=cab)
        celda.fill, celda.font = AZUL, BLANCA
    rango = f"Anotación!${LETRA['contexto']}$3:${LETRA['contexto']}${ultima}"
    for k, etiqueta in enumerate(ETIQUETAS_CONTEXTO, fila + 1):
        ws.cell(row=k, column=1, value=etiqueta)
        ws.cell(row=k, column=2, value=f'=COUNTIF({rango},"{etiqueta}")')
    ws.column_dimensions["A"].width = 24


def generar_plantilla(args) -> None:
    destino = args.salida.resolve()
    if destino.exists() and not args.sobrescribir:
        raise SystemExit(f"{destino} ya existe y podría tener anotaciones. Usa --sobrescribir "
                         "para generarla de nuevo (perderías lo anotado) o --salida para otra ruta.")
    rutas = fotos_prueba(args.carpeta)
    wb = Workbook()
    ws = wb.active
    ws.title = "Anotación"
    ultima = hoja_anotacion(ws, rutas, destino)
    hoja_guia(wb.create_sheet("Guía"))
    hoja_progreso(wb.create_sheet("Progreso"), ultima)
    destino.parent.mkdir(parents=True, exist_ok=True)
    wb.save(destino)
    print(f"{len(rutas)} fotos · plantilla: {destino}")


def _texto(valor) -> str:
    return "" if valor is None else str(valor).strip()


def _decision(valor) -> str:
    texto = _texto(valor).lower()
    return "si" if texto == "sí" else texto


def horizonte_amplio(natural: str, estructural: str) -> str:
    """Misma regla que aplicabilidad_vlm.horizonte_amplio: «si» si cualquiera lo es."""
    if "si" in (natural, estructural):
        return "si"
    if "dudoso" in (natural, estructural):
        return "dudoso"
    return "no"


def exportar(args) -> None:
    origen = args.plantilla.resolve()
    if not origen.exists():
        raise SystemExit(f"No existe {origen}. Genera antes la plantilla.")
    ws = load_workbook(origen)["Anotación"]
    claves = [_texto(c.value) for c in ws[1]]
    faltan = {"imagen", "contexto", *DECISIONES} - set(claves)
    if faltan:
        raise SystemExit(f"La fila 1 de la hoja Anotación no tiene: {sorted(faltan)}")
    col = {c: j for j, c in enumerate(claves) if c}

    filas, errores = [], []
    for fila in ws.iter_rows(min_row=3, values_only=True):
        imagen = _texto(fila[col["imagen"]])
        if not imagen:
            continue
        registro = {"imagen": imagen, "contexto": _texto(fila[col["contexto"]])}
        invalidas = [] if registro["contexto"] in ETIQUETAS_CONTEXTO else ["contexto"]
        for clave in DECISIONES:
            registro[clave] = _decision(fila[col[clave]])
            if registro[clave] not in VALORES:
                invalidas.append(clave)
        if invalidas:
            errores.append(f"{imagen}: falta o no es válido: {', '.join(invalidas)}")
            continue
        if registro["protagonista"] == "no" and any(registro[c] == "si" for c in DEPENDIENTES):
            errores.append(f"{imagen}: sin protagonista, pero espacio/nitidez/anclaje = si")
        registro["horizonte_amplio"] = horizonte_amplio(
            registro["horizonte_natural"], registro["horizontal_estructural"])
        for extra in ("protagonista_nombre", "comentario"):
            registro[extra] = _texto(fila[col[extra]]) if extra in col else ""
        filas.append(registro)

    if not errores and len(filas) != N_ESPERADO:
        errores.append(f"hay {len(filas)} fotos anotadas; se esperaban {N_ESPERADO}")
    if errores:
        print("La anotación no está lista para exportar:")
        for e in errores:
            print(f"  - {e}")
        raise SystemExit(1)

    campos = ["imagen", "contexto", *DECISIONES[:3], "horizonte_amplio", *DECISIONES[3:5],
              "protagonista_nombre", *DECISIONES[5:], "comentario"]
    destino = args.salida.resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(sorted(filas, key=lambda r: numero_imagen(Path(r["imagen"]))))
    print(f"{len(filas)} fotos exportadas · {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="orden", required=True)

    p = sub.add_parser("plantilla", help="genera el Excel ciego de anotación")
    p.add_argument("--carpeta", type=Path, default=DIR_PRUEBA)
    p.add_argument("--salida", type=Path, default=PLANTILLA)
    p.add_argument("--sobrescribir", action="store_true",
                   help="reemplazar la plantilla aunque ya exista (perderías lo anotado)")
    p.set_defaults(funcion=generar_plantilla)

    e = sub.add_parser("exportar", help="valida el Excel rellenado y escribe el CSV")
    e.add_argument("--plantilla", type=Path, default=PLANTILLA)
    e.add_argument("--salida", type=Path, default=CSV_SALIDA)
    e.set_defaults(funcion=exportar)

    args = parser.parse_args()
    args.funcion(args)


if __name__ == "__main__":
    main()
