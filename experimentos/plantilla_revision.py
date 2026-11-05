"""Plantilla Excel para la revisión humana de la aplicabilidad (anotación A2).

Genera una hoja con una fila por fotografía, su miniatura, un enlace a la imagen
completa y una columna por pregunta con un desplegable si / no / dudoso. Las preguntas
son las mismas ocho que responden Gemini y el juez, con la misma definición operativa,
para que las tres fuentes sean comparables.

La plantilla es CIEGA a propósito: no muestra la decisión de la heurística, ni la de
Gemini, ni la de Claude. Ver lo que ha decidido una máquina antes de anotar sesga al
anotador hacia esa decisión, que es justo lo que la anotación tiene que poder corregir.

Una vez rellenada, se pasa tal cual al cruce:
    uv run python experimentos/aplicabilidad_vlm.py \\
        --anotaciones experimentos/resultados/aplicabilidad_vlm/plantilla_revision_humana.xlsx \\
        --juez experimentos/resultados/aplicabilidad_vlm/juicio_claude.csv

Uso:
    uv run python experimentos/plantilla_revision.py              # no pisa una ya rellenada
    uv run python experimentos/plantilla_revision.py --sobrescribir
"""
from __future__ import annotations

import argparse
import io
import os
import re
from pathlib import Path

import cv2
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image as ImagenExcel
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

RAIZ = Path(__file__).resolve().parents[1]
DATA_PATH = RAIZ / "data"
DESTINO = RAIZ / "experimentos" / "resultados" / "aplicabilidad_vlm" / "plantilla_revision_humana.xlsx"

# Miniatura: lado mayor en píxeles y alto de fila en puntos (1 pt ≈ 1,33 px).
LADO_MINIATURA = 150
ALTO_FILA_PT = 118

# (columna, título corto, métrica que decide, SÍ aplica cuando…, NO aplica cuando…)
PREGUNTAS = [
    ("color", "¿Color analizable?", "esquema_cromatico",
     "Hay color real con saturación apreciable. También si domina un solo matiz "
     "saturado: una foto azulada o anaranjada SÍ tiene esquema.",
     "Blanco y negro, casi acromática, muy desaturada o nocturna con solo un resto de "
     "color."),
    ("horizonte_natural", "¿Horizonte natural?", "angulo_horizonte (estricta)",
     "Hay una línea de horizonte legible (cielo contra tierra o mar) que cruza buena "
     "parte del encuadre y sobre la que tiene sentido juzgar si la foto está nivelada.",
     "Solo hay horizontales de objetos o construcciones, el horizonte está tapado "
     "(montañas, árboles, edificios) o no hay ninguno."),
    ("horizontal_estructural", "¿Horizontal estructural?", "angulo_horizonte (amplia)",
     "Una línea horizontal de la propia escena (cornisa, borde de mesa, línea del "
     "suelo, fachada, barandilla, orilla) cruza MÁS DE LA MITAD del ancho y sirve de "
     "referencia de nivelación. Se responde con independencia de la pregunta anterior.",
     "Las horizontales son cortas, curvas o diagonales por la perspectiva."),
    ("perspectiva", "¿Perspectiva que organiza?", "score_convergencia",
     "Líneas reales de la escena (calles, raíles, pasillos, aristas, filas de objetos) "
     "convergen hacia un punto de fuga y ESTRUCTURAN la composición.",
     "Solo hay paralelas, texturas, vegetación o un muro plano, o la convergencia es "
     "marginal y no organiza la foto."),
    ("protagonista", "¿Hay protagonista?", "(apoyo para las tres siguientes)",
     "Hay un sujeto concreto y delimitado (persona, animal, objeto, edificio o elemento "
     "aislado) que es claramente el protagonista. Nómbralo en la columna siguiente.",
     "Paisaje sin foco, patrón, masa vegetal o varios elementos con el mismo peso."),
    ("espacio", "¿Medir espacio negativo?", "ratio_espacio_negativo",
     "Tiene sentido medir qué parte del encuadre ocupa el protagonista frente al espacio "
     "libre: hay protagonista delimitado y separable del fondo.",
     "No hay protagonista, o lo hay pero llena el encuadre sin borde reconocible o se "
     "confunde con el fondo."),
    ("nitidez", "¿Comparar nitidez?", "ratio_nitidez",
     "Tiene sentido comparar la nitidez del protagonista con la del fondo: hay "
     "protagonista delimitado y alguna de las dos regiones tiene detalle o textura.",
     "No hay protagonista, o toda la imagen es lisa (niebla, cielo, pared uniforme)."),
    ("anclaje", "¿Evaluar su posición?", "d_tercios, d_centro, patron_dominante",
     "El protagonista es un elemento único y localizable cuya posición en el encuadre "
     "(cerca de los tercios o del centro) tiene sentido evaluar.",
     "No hay protagonista, hay varios con el mismo peso, o la composición la organiza "
     "una masa, un patrón o el paisaje entero."),
]
DECISIONES = [p[0] for p in PREGUNTAS]

AZUL = PatternFill("solid", start_color="1F3864")
VACIO = PatternFill("solid", start_color="FFF2CC")
VERDE = PatternFill("solid", start_color="C6EFCE")
ROJO = PatternFill("solid", start_color="FFC7CE")
AMBAR = PatternFill("solid", start_color="FFEB9C")
BLANCA = Font(bold=True, color="FFFFFF")
FINO = Side(style="thin", color="BFBFBF")
BORDE = Border(left=FINO, right=FINO, top=FINO, bottom=FINO)


def numero_imagen(ruta):
    m = re.search(r"(\d+)$", ruta.stem)
    return int(m.group(1)) if m else 10**9


def miniatura(ruta):
    img = cv2.imread(str(ruta))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    alto, ancho = img.shape[:2]
    escala = LADO_MINIATURA / max(alto, ancho)
    img = cv2.resize(img, (round(ancho * escala), round(alto * escala)),
                     interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"No se pudo codificar la miniatura: {ruta}")
    return io.BytesIO(buffer.tobytes())


def hoja_revision(ws, rutas, destino):
    columnas = (
        [("imagen", "Imagen", 11), ("miniatura", "Miniatura", 23), ("abrir", "Abrir", 9)]
        + [(c, titulo, 14) for c, titulo, *_ in PREGUNTAS[:5]]
        + [("protagonista_nombre", "Nombre del protagonista", 22)]
        + [(c, titulo, 14) for c, titulo, *_ in PREGUNTAS[5:]]
        + [("comentario", "Comentario (opcional)", 45), ("estado", "Estado", 13)]
    )
    letra = {c: get_column_letter(j) for j, (c, _, _) in enumerate(columnas, 1)}
    definicion = {c: (si, no) for c, _, _, si, no in PREGUNTAS}

    # Fila 1: nombre de máquina (lo que lee --anotaciones). Fila 2: título legible.
    for j, (clave, titulo, ancho) in enumerate(columnas, 1):
        maquina = ws.cell(row=1, column=j, value=clave)
        maquina.font = Font(color="808080", size=9)
        cab = ws.cell(row=2, column=j, value=titulo)
        cab.fill, cab.font, cab.border = AZUL, BLANCA, BORDE
        cab.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = ancho
        if clave in definicion:
            si, no = definicion[clave]
            cab.comment = Comment(f"SÍ: {si}\n\nNO: {no}", "Guía A2", width=320, height=220)
    ws.row_dimensions[2].height = 42

    validacion = DataValidation(
        type="list", formula1='"si,no,dudoso"', allow_blank=True, showDropDown=False,
        errorTitle="Valor no válido", error="Elige si, no o dudoso en el desplegable.",
        promptTitle="Respuesta", prompt="si / no / dudoso (pasa el ratón por la cabecera "
        "para ver la definición)",
    )
    ws.add_data_validation(validacion)

    for i, ruta in enumerate(rutas, 3):
        ws.row_dimensions[i].height = ALTO_FILA_PT
        ws.cell(row=i, column=1, value=ruta.stem)
        enlace = ws.cell(row=i, column=3, value="abrir")
        # Ruta relativa al propio Excel, para que el enlace siga funcionando si se mueve
        # el proyecto entero de carpeta.
        enlace.hyperlink = os.path.relpath(ruta, destino.parent)
        enlace.font = Font(color="0563C1", underline="single")
        imagen = ImagenExcel(miniatura(ruta))
        ws.add_image(imagen, f"{letra['miniatura']}{i}")

        decisiones = [f"{letra[c]}{i}" for c in DECISIONES]
        rango = ",".join(decisiones)
        ws.cell(row=i, column=len(columnas), value=(
            f'=IF(COUNTA({rango})=0,"pendiente",'
            f'IF(COUNTA({rango})<{len(DECISIONES)},"incompleta",'
            f'IF(AND({letra["protagonista"]}{i}="no",OR({letra["espacio"]}{i}="si",'
            f'{letra["nitidez"]}{i}="si",{letra["anclaje"]}{i}="si")),"revisar","completa")))'
        ))
        for clave, _, _ in columnas:
            celda = ws[f"{letra[clave]}{i}"]
            celda.border = BORDE
            celda.alignment = Alignment(vertical="center", wrap_text=True,
                                        horizontal="center" if clave in DECISIONES else None)
            if clave in DECISIONES:
                validacion.add(celda)

    ultima = len(rutas) + 2
    zona = f"{letra[DECISIONES[0]]}3:{letra['anclaje']}{ultima}"
    # Color por respuesta: se ve de un vistazo qué falta y si hay patrones raros.
    ws.conditional_formatting.add(zona, CellIsRule(operator="equal", formula=['"si"'], fill=VERDE))
    ws.conditional_formatting.add(zona, CellIsRule(operator="equal", formula=['"no"'], fill=ROJO))
    ws.conditional_formatting.add(zona, CellIsRule(operator="equal", formula=['"dudoso"'], fill=AMBAR))
    primera = f"{letra[DECISIONES[0]]}3"
    ws.conditional_formatting.add(zona, FormulaRule(formula=[f'LEN({primera})=0'], fill=VACIO))
    estado = f"{letra['estado']}3:{letra['estado']}{ultima}"
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"completa"'], fill=VERDE))
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"revisar"'], fill=ROJO))
    ws.conditional_formatting.add(estado, CellIsRule(operator="equal", formula=['"incompleta"'], fill=AMBAR))

    ws.freeze_panes = f"{letra['color']}3"
    ws.auto_filter.ref = f"A2:{letra['estado']}{ultima}"
    return letra, ultima


def hoja_guia(ws):
    ws["A1"] = "Guía de anotación de la aplicabilidad (A2)"
    ws["A1"].font = Font(bold=True, size=14)
    instrucciones = [
        "1. Mira la foto completa con «abrir» antes de decidir: la miniatura sirve para "
        "orientarse, no para juzgar líneas finas.",
        "2. Responde si / no / dudoso en cada columna con el desplegable. Usa «dudoso» solo "
        "si de verdad no puedes decidir; en el cruce cuenta como no abrir y queda fuera "
        "de los falsos positivos y negativos.",
        "3. Responde a la pregunta por lo que VES en la escena, no por lo que crees que "
        "mediría el sistema.",
        "4. Coherencia: si no hay protagonista, espacio, nitidez y anclaje deben ser «no». "
        "La columna Estado marca «revisar» si no se cumple.",
        "5. La plantilla no muestra lo que decidieron la heurística, Gemini ni Claude, a "
        "propósito: verlo antes de anotar sesgaría la anotación.",
        "6. No cambies la fila 1 de la hoja Revisión: son los nombres que lee el script.",
    ]
    for k, texto in enumerate(instrucciones, 3):
        ws.cell(row=k, column=1, value=texto).alignment = Alignment(wrap_text=True)
        ws.merge_cells(start_row=k, start_column=1, end_row=k, end_column=4)
        ws.row_dimensions[k].height = 32

    fila = len(instrucciones) + 5
    for j, cab in enumerate(["Columna", "Métrica que decide", "SÍ aplica cuando…",
                             "NO aplica cuando…"], 1):
        celda = ws.cell(row=fila, column=j, value=cab)
        celda.fill, celda.font, celda.border = AZUL, BLANCA, BORDE
    for k, (clave, titulo, metrica, si, no) in enumerate(PREGUNTAS, fila + 1):
        for j, valor in enumerate([f"{clave}\n{titulo}", metrica, si, no], 1):
            celda = ws.cell(row=k, column=j, value=valor)
            celda.alignment = Alignment(wrap_text=True, vertical="top")
            celda.border = BORDE
        ws.row_dimensions[k].height = 75
    for col, ancho in zip("ABCD", (26, 26, 60, 60)):
        ws.column_dimensions[col].width = ancho
    nota = fila + len(PREGUNTAS) + 2
    ws.cell(row=nota, column=1, value=(
        "horizonte_amplio no se anota: el script lo deriva como «si» cuando "
        "horizonte_natural o horizontal_estructural es «si»."
    )).font = Font(italic=True)


def hoja_progreso(ws, letra, ultima):
    ws["A1"] = "Progreso de la revisión"
    ws["A1"].font = Font(bold=True, size=14)
    estado = f"Revisión!${letra['estado']}$3:${letra['estado']}${ultima}"
    for k, valor in enumerate(["completa", "incompleta", "revisar", "pendiente"], 3):
        ws.cell(row=k, column=1, value=valor)
        ws.cell(row=k, column=2, value=f'=COUNTIF({estado},"{valor}")')

    fila = 9
    for j, cab in enumerate(["Pregunta", "si", "no", "dudoso", "sin responder"], 1):
        celda = ws.cell(row=fila, column=j, value=cab)
        celda.fill, celda.font = AZUL, BLANCA
    for k, clave in enumerate(DECISIONES, fila + 1):
        rango = f"Revisión!${letra[clave]}$3:${letra[clave]}${ultima}"
        ws.cell(row=k, column=1, value=clave)
        for j, valor in enumerate(["si", "no", "dudoso"], 2):
            ws.cell(row=k, column=j, value=f'=COUNTIF({rango},"{valor}")')
        ws.cell(row=k, column=5, value=f"=COUNTBLANK({rango})")
    ws.column_dimensions["A"].width = 24


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--carpeta", type=Path, default=DATA_PATH)
    parser.add_argument("--salida", type=Path, default=DESTINO)
    parser.add_argument("--sobrescribir", action="store_true",
                        help="reemplazar la plantilla aunque ya exista (perderías lo anotado)")
    args = parser.parse_args()

    destino = args.salida.resolve()
    if destino.exists() and not args.sobrescribir:
        raise SystemExit(
            f"{destino} ya existe y podría tener anotaciones. "
            "Usa --sobrescribir para generarla de nuevo, o --salida para otra ruta."
        )
    rutas = sorted(
        (p for p in args.carpeta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: (numero_imagen(p), p.name),
    )

    wb = Workbook()
    revision = wb.active
    revision.title = "Revisión"
    letra, ultima = hoja_revision(revision, rutas, destino)
    hoja_guia(wb.create_sheet("Guía"))
    hoja_progreso(wb.create_sheet("Progreso"), letra, ultima)

    destino.parent.mkdir(parents=True, exist_ok=True)
    wb.save(destino)
    print(f"{len(rutas)} imágenes · plantilla: {destino}")


if __name__ == "__main__":
    main()
