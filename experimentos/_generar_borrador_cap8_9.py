"""Genera el borrador en Word de los capítulos 8 y 9 del TFG (python-docx).

Pies de tabla y figura con campos SEQ (como en la memoria) y referencias cruzadas con campos REF a
marcadores, para que al pegarlo en la memoria y actualizar campos (Ctrl+A, F9) se renumere todo.
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RAIZ = Path(r"C:\Users\ediso\Documents\tfg_multiagente_fotografia")
RES = RAIZ / "experimentos" / "resultados"
DESTINO = RAIZ / "Capitulos_8_9_borrador.docx"

# Numeración provisional: la memoria va por la Tabla 10 y la Figura 32.
TABLAS = ["preguntas", "conjuntos", "diseno_c2", "protocolos", "aplica", "e1_clases", "e1_umbral",
          "e2_t1", "e2_t2", "e3_autores", "e3_critico", "e4_coste", "e4_repro", "resumen"]
FIGURAS = ["matriz_c1", "gradcam", "latencia", "casos"]
NUM = {("T", k): 11 + i for i, k in enumerate(TABLAS)} | {("F", k): 33 + i for i, k in enumerate(FIGURAS)}
ETIQ = {"T": "Tabla", "F": "Figura"}

doc = Document()
estilos = doc.styles
normal = estilos["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(12)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing = 1.15
rpr = normal.element.get_or_add_rPr()
lang = OxmlElement("w:lang")
lang.set(qn("w:val"), "es-ES")
rpr.append(lang)
for nivel, tam in ((1, 16), (2, 13), (3, 12)):
    h = estilos[f"Heading {nivel}"]
    h.font.name = "Calibri"
    h.font.size = Pt(tam)
    h.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
estilos["Caption"].font.size = Pt(10)
estilos["Caption"].font.color.rgb = RGBColor(0x40, 0x40, 0x40)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.5)


# ---------------------------------------------------------------------------------------------
def _campo(parrafo, instr, texto):
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), instr)
    r = OxmlElement("w:r")
    rp = OxmlElement("w:rPr")
    rp.append(OxmlElement("w:noProof"))
    r.append(rp)
    t = OxmlElement("w:t")
    t.text = texto
    t.set(qn("xml:space"), "preserve")
    r.append(t)
    fld.append(r)
    parrafo._p.append(fld)


_ID = [100]


def _marcador(parrafo, nombre, inicio):
    if inicio:
        el = OxmlElement("w:bookmarkStart")
        el.set(qn("w:name"), nombre)
    else:
        el = OxmlElement("w:bookmarkEnd")
    el.set(qn("w:id"), str(_ID[0] if inicio else _ID[0] - 1))
    if inicio:
        _ID[0] += 1
    parrafo._p.append(el)


TOKEN = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|\{[TF]:\w+\})")


def texto(parrafo, s):
    """Añade texto con **negrita**, *cursiva* y referencias {T:clave} / {F:clave}."""
    for trozo in TOKEN.split(s):
        if not trozo:
            continue
        if trozo.startswith("**"):
            parrafo.add_run(trozo[2:-2]).bold = True
        elif trozo.startswith("{"):
            tipo, clave = trozo[1:-1].split(":")
            _campo(parrafo, f" REF _Ref{tipo}_{clave} \\h ", f"{ETIQ[tipo]} {NUM[(tipo, clave)]}")
        elif trozo.startswith("*"):
            parrafo.add_run(trozo[1:-1]).italic = True
        else:
            parrafo.add_run(trozo)
    return parrafo


def P(s, estilo=None):
    p = doc.add_paragraph(style=estilo)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return texto(p, s)


def V(items):
    for s in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        texto(p, s)
    doc.paragraphs[-1].paragraph_format.space_after = Pt(8)


def H(s, nivel):
    doc.add_heading(s, level=nivel)


def pie(tipo, clave, descripcion):
    p = doc.add_paragraph(style="Caption")
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    nombre = f"_Ref{tipo}_{clave}"
    _marcador(p, nombre, True)
    p.add_run(f"{ETIQ[tipo]} ")
    _campo(p, f" SEQ {ETIQ[tipo]} \\* ARABIC ", str(NUM[(tipo, clave)]))
    _marcador(p, nombre, False)
    texto(p, f": {descripcion}")


def _sombra(celda, color):
    tcpr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcpr.append(shd)


def tabla(clave, descripcion, cabecera, filas, anchos, negrita_ultima=False):
    t = doc.add_table(rows=1 + len(filas), cols=len(cabecera))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, fila in enumerate([cabecera] + filas):
        for j, valor in enumerate(fila):
            c = t.cell(i, j)
            c.width = Cm(anchos[j])
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            texto(p, str(valor))
            for r in p.runs:
                r.font.size = Pt(9.5)
                if i == 0 or (negrita_ultima and i == len(filas)):
                    r.bold = True
            if i == 0:
                _sombra(c, "D9D9D9")
    pie("T", clave, descripcion)


def figura(clave, descripcion, ruta, ancho_cm):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(ruta), width=Cm(ancho_cm))
    pie("F", clave, descripcion)


# =============================================================================================
# CAPÍTULO 8
# =============================================================================================
H("Validación experimental", 1)
P("Este capítulo fija el diseño de la validación antes de presentar ningún resultado. Delimita qué se puede "
  "validar de un sistema que emite juicios compositivos, enuncia las preguntas que se quieren responder, "
  "describe los conjuntos de imágenes y los instrumentos de medida, y establece para cada experimento un "
  "protocolo con su criterio de éxito. Los criterios se fijaron antes de ejecutar ningún experimento y no se "
  "modificaron después de ver los resultados, que se presentan en el capítulo 9 en el mismo orden.")

H("Qué se puede validar y qué no", 2)
P("El sistema no emite un juicio estético (apartados 1.5 y 3.1). Por ello la validación no mide si una "
  "crítica es acertada desde el punto de vista artístico ni si coincide con el gusto de un fotógrafo. "
  "Tampoco pretende demostrar que el vector de pesos contextuales W sea óptimo: W es una hipótesis de "
  "diseño con justificación bibliográfica (apartado 4.6) y los experimentos no la ponen a prueba.")
P("Lo que sí puede validarse son las garantías de ingeniería sobre las que descansa la propuesta: que el "
  "contexto que ordena la crítica sea el correcto, que una métrica solo sea citable cuando su regla aplica, "
  "que toda cifra de la crítica proceda del informe de una herramienta sin alterarse, y que un análisis tenga "
  "un coste y un tiempo acotados y sea reproducible.")
P("Quedan fuera de la validación la comparación con un sistema de agente único y el cuarto objetivo "
  "específico, relativo a la interfaz y su usabilidad. Este último no se valida experimentalmente porque no "
  "se ha realizado un estudio con usuarios; su cumplimiento se argumenta de forma cualitativa en el capítulo "
  "7 y se valora en el capítulo 10.")

H("Preguntas de validación", 2)
P("La validación se organiza en cuatro preguntas, cada una asociada a un experimento. La {T:preguntas} "
  "relaciona cada pregunta con los objetivos específicos del apartado 1.4.2, citados por su orden como OE1 "
  "(arquitectura con orquestador), OE2 (métricas cuantitativas verificables), OE3 (informe basado en datos y "
  "priorización por tipo de fotografía) y OE4 (interfaz).")
tabla("preguntas", "Preguntas de validación, objetivo específico que validan y experimento asociado",
      ["Pregunta", "Enunciado", "Objetivo", "Exp."],
      [["P1", "¿El clasificador asigna el contexto correcto y rechaza lo que no reconoce?", "OE1, OE3", "E1"],
       ["P2", "¿Las condiciones de aplicabilidad cierran una métrica cuando su regla no aplica, sin dejarla abierta cuando no aplica?", "OE2", "E2"],
       ["P3", "¿Toda cifra del texto sale del informe, sin alterarse y sin citar métricas cerradas? ¿Respeta el crítico el arbitraje?", "OE3, OE2", "E3"],
       ["P4", "¿Cuánto cuesta y cuánto tarda un análisis, y es reproducible?", "OE1", "E4"],
       ["—", "Usabilidad de la interfaz", "OE4", "no se valida"]],
      [1.6, 10.4, 2.0, 2.0])

H("Conjuntos de imágenes", 2)
P("Intervienen tres conjuntos (ver {T:conjuntos}). El primero procede de EVA, el conjunto con el que se "
  "entrenó el clasificador de contexto (apartado 5.1). Su partición de prueba (C1, 725 imágenes) no se usó "
  "para seleccionar el modelo, aunque sí para elegir el umbral de rechazo de 0,85; por eso el umbral se "
  "controla además sobre la partición de validación (val). La categoría *other* de EVA (O, 273 imágenes) "
  "sirve como sustituto de imágenes fuera de distribución: es un conjunto *near-OOD*, del mismo concurso "
  "y con la misma estética, y no equivale exactamente a la clase «otro» del sistema (apartado 5.1.2).")
tabla("conjuntos", "Conjuntos de imágenes de la validación y experimentos en los que intervienen",
      ["Conjunto", "Descripción", "n", "Uso"],
      [["C1", "Partición de prueba de EVA (136 animal, 143 arquitectura, 152 paisaje, 139 producto, 155 retrato)", "725", "E1"],
       ["O", "Categoría *other* de EVA, sustituto *near-OOD*", "273", "E1"],
       ["val", "Partición de validación de EVA", "725", "E1"],
       ["C2", "Corpus de desarrollo: fotografías de Unsplash seleccionadas por el autor", "100", "E1-E4"],
       ["R", "Subconjunto de C2, una imagen por contexto, con tres repeticiones", "5 × 3", "E4"]],
      [1.8, 10.2, 1.6, 2.4])
P("El segundo conjunto, C2, está formado por 100 fotografías de Unsplash publicadas bajo la licencia de "
  "Unsplash. La selección es intencional y estratificada, no aleatoria: se fijaron 18 imágenes por cada una "
  "de las cinco categorías del sistema y 10 de «otro» (fotografías reales que no encajan en ninguna "
  "categoría, como vehículos, fuegos artificiales o escenas sin dominante) y, dentro de cada categoría, unos "
  "subtipos elegidos para ejercitar cada condición de aplicabilidad en los dos sentidos: casos en los que "
  "debería abrirse y casos en los que debería cerrarse ({T:diseno_c2}). C2 no comparte ninguna imagen con el "
  "corpus de 40 fotografías usado durante el desarrollo (comprobado por MD5) y se congela mediante un "
  "manifiesto con el nombre, el MD5, las dimensiones y el origen de cada imagen.")
tabla("diseno_c2", "Diseño de C2 por categoría y subtipo, y qué pone a prueba cada grupo",
      ["Categoría (n)", "Subtipos (n)", "Qué pone a prueba"],
      [["Arquitectura (18)", "un punto de fuga (5), fachada frontal (4), contrapicado o dos fugas (3), interiores (3), nocturna o B/N (3)", "convergencia abierta y cerrada; condición cromática"],
       ["Paisaje (18)", "horizonte nivelado (5) o inclinado (3), camino que se aleja (3), sin horizonte (3), foco que YOLO no detecta (2), contraluz (2)", "horizonte; convergencia fuera de la arquitectura; vía de saliencia"],
       ["Retrato (18)", "fondo desenfocado y sujeto en tercios (5), centrado con fondo nítido (4), sujeto pequeño (3), B/N (3), clave alta o baja (3)", "espacio y nitidez; centrado; condición cromática; exposición"],
       ["Animal (18)", "especie que YOLO conoce con fondo desenfocado (6) o en entorno amplio (4), especie que YOLO no conoce (3), centrado frente a descentrado (3), camuflaje (2)", "fuente YOLO frente a saliencia; composición; percepción"],
       ["Producto (18)", "objeto que YOLO conoce (5), blanco sobre blanco (3), clave baja (3), bodegón con varios objetos (3), objeto que YOLO no conoce (2), color marcado (2)", "nitidez sin textura; elección del sujeto; esquema cromático"],
       ["Otro (10)", "vehículo sin personas (4), sin objeto significativo (3), persona y animal a la par (1), animal y paisaje empatados (1), deporte (1)", "rechazo a «otro» y pesos uniformes"]],
      [2.8, 8.6, 4.6])
P("Las imágenes tienen entre 2,07 y 25,96 megapíxeles (mediana 4,79); 53 son verticales, 43 horizontales y "
  "4 cuadradas. La anotación de contexto realizada después, sin consultar el diseño, asigna 19 imágenes a "
  "animal, 18 a cada una de las otras cuatro categorías y 9 a «otro»: una imagen diseñada como empate entre "
  "animal y paisaje se anotó como animal. La anotación de casos límite identifica 21 imágenes sin color "
  "analizable, 33 sin un sujeto discreto, 19 con horizonte natural y 16 con perspectiva lineal.")
P("El tercer conjunto, R, reúne cinco imágenes de C2, una por categoría, elegidas por sorteo con semilla "
  "fija alternando imágenes cuyo sujeto localiza YOLO y otras en las que lo localiza el mapa de saliencia. "
  "Cada una se analiza tres veces para medir la reproducibilidad.")

H("Instrumentos", 2)
P("**Anotación de referencia.** La verdad de referencia la anota el autor como anotador único, siguiendo "
  "una guía fijada antes de anotar. La anotación es ciega: se anota la fotografía sin ver ninguna salida del "
  "sistema. Consta del contexto verdadero de cada imagen (A1, seis etiquetas) y de cuatro preguntas de "
  "aplicabilidad (A2) respondidas con «sí», «no» o «dudoso»: si la imagen tiene color analizable, si tiene "
  "un horizonte natural legible, si su profundidad se organiza por perspectiva lineal y si carece de un "
  "sujeto discreto. En las 60 imágenes cuyo sujeto detecta YOLO se revisa además si la máscara de "
  "segmentación rodea al sujeto (bien, parcial o mal) y si el objeto detectado es el protagonista.")
P("**Medición directa de las condiciones de aplicabilidad.** Las confianzas se obtienen ejecutando las "
  "herramientas deterministas de los cuatro especialistas sobre C2, no leyendo el texto generado por el "
  "modelo de lenguaje. Así, la medida no incluye los posibles errores de transcripción del LLM, que se "
  "evalúan aparte. Cada decisión se replica a partir de los campos del informe y se comprueba que "
  "coincide con la del código.")
P("**Verificador de citas.** Extrae cada cita con formato [campo = valor] de los diagnósticos y de la "
  "crítica, la resuelve contra el informe correspondiente y la clasifica como exacta (redondeo correcto a "
  "los decimales escritos), truncada (coincide al cortar dígitos, no al redondear), alterada o referida a un "
  "campo inexistente. Marca como violación citar el valor de una métrica cerrada o un campo de evidencia "
  "que depende de ella, recalcula el arbitraje con el mismo código del crítico y compara, campo a campo, el "
  "informe transcrito por el LLM con el que produce la herramienta.")
P("**Lanzador por lotes y registro.** Cada análisis se ejecuta de forma completa y reanudable, y para cada "
  "uno se registran el tiempo total y el instante en que termina cada tarea, los tokens de entrada y de "
  "salida de cada modelo (incluidos los de razonamiento), la memoria de GPU máxima y los errores. El coste "
  "se calcula multiplicando los tokens por las tarifas oficiales de la API de Gemini consultadas el 28 de "
  "septiembre de 2026: 0,30 y 2,50 USD por millón de tokens de entrada y salida para Gemini 2.5 Flash, y "
  "1,25 y 10,00 USD para Gemini 2.5 Pro. Los tokens de razonamiento se facturan como salida.")
P("**Congelación del sistema.** Antes de ejecutar se fijaron el código (commit), los modelos, las "
  "instrucciones de los agentes, W, las constantes de cada herramienta, las imágenes, las anotaciones, los "
  "criterios de éxito y las tarifas. Todas las ejecuciones se hicieron con ese estado.")

H("Protocolos", 2)
P("La {T:protocolos} resume los cuatro protocolos y sus criterios de éxito. T100 designa la ejecución "
  "completa del sistema, una vez, sobre cada imagen de C2. Un criterio no cumplido se informa tal cual en "
  "el capítulo 9.")
tabla("protocolos", "Protocolos experimentales y criterios de éxito fijados antes de ejecutar",
      ["Exp.", "Conjunto", "Qué se mide", "Criterio de éxito"],
      [["E1", "C1, O, val, C2", "Exactitud y F1 por clase; cobertura y exactitud con el umbral 0,85; AUROC frente a O; % enviado a «otro»", "F1 ≥ 0,75 en cada clase de C1; en C2, exactitud sobre las aceptadas ≥ 0,85 y ≤ 20% enviado a «otro»"],
       ["E2", "C2 con A2", "Matriz abierta/cerrada frente a aplica/no aplica por condición", "0 falsos positivos por condición (definición estricta)"],
       ["E3", "T100", "Clase de cada cita; controles del crítico; fidelidad de la transcripción", "0 citas a métricas cerradas; ≥ 95% de citas exactas; 100% de críticas con las afirmaciones en orden"],
       ["E4", "T100, R", "Latencia, tokens, coste, memoria y fallos; identidad entre repeticiones", "100% de informes transcritos idénticos a la herramienta y arbitraje idéntico en las tres repeticiones"]],
      [1.3, 2.4, 6.1, 6.2])

H("E1: clasificador de contexto (P1)", 3)
P("Se infieren las imágenes de C1, O, val y C2 con el modelo desplegado y su preprocesado. Sobre C1 se "
  "miden la exactitud, la F1 por clase y la matriz de confusión sin umbral. Con el umbral de rechazo se "
  "calculan la cobertura (fracción de imágenes aceptadas), la exactitud sobre las aceptadas, la fracción de "
  "errores que el rechazo captura y el AUROC entre C1 y O; la misma curva se repite sobre val para "
  "comprobar que el umbral no es un artefacto de haberlo elegido con C1. En C2 se compara la etiqueta del "
  "sistema con A1 y se comprueba que el orquestador copia sin cambios la etiqueta del clasificador. El "
  "par de clases más confundido se analiza con Grad-CAM. El criterio sobre C1 es descriptivo, porque la F1 "
  "mínima ya se conocía; los dos criterios sobre C2 se fijaron antes de anotar A1.")

H("E2: condiciones de aplicabilidad (P2)", 3)
P("Se ejecutan los cuatro motores sobre C2 y, para cada condición, se cruza si la métrica queda abierta "
  "con si la regla aplica según A2. Hay ocho métricas con condición, pero seis decisiones independientes, "
  "porque las tres de composición comparten la suya. Un falso positivo (métrica abierta cuando la regla no "
  "aplica) es el error grave, porque permite citar una cifra sin sentido; un falso negativo (métrica cerrada "
  "cuando la regla aplica) solo hace que el sistema calle. La {T:aplica} define cuándo aplica cada regla. "
  "Las imágenes anotadas como dudosas se excluyen de la matriz y se cuentan aparte, y las tasas se dan con "
  "intervalos de Wilson al 95%. Se decidió de antemano no corregir los fallos ya conocidos del sistema.")
tabla("aplica", "Cuándo aplica la regla de cada condición de aplicabilidad según la anotación A2",
      ["Condición", "Métricas", "La regla aplica si…"],
      [["Cromática", "esquema_cromatico", "la imagen tiene color analizable"],
       ["Horizonte", "angulo_horizonte", "hay un horizonte natural legible"],
       ["Convergencia", "score_convergencia", "la profundidad se organiza por perspectiva lineal"],
       ["Identidad (espacio)", "ratio_espacio_negativo", "hay sujeto discreto y, si la fuente es YOLO, la segmentación no es mala (estricta) o es buena (amplia)"],
       ["Identidad y textura", "ratio_nitidez", "igual que la anterior"],
       ["Identidad (composición)", "d_tercios, d_centro, patron_dominante", "hay sujeto discreto y, si la fuente es YOLO, el objeto detectado es el protagonista"]],
      [3.4, 4.6, 8.0])

H("E3: fidelidad de las citas (P3)", 3)
P("Se ejecuta el sistema completo una vez sobre cada imagen de C2 (tanda T100, con lectura visual). El "
  "verificador se aplica a los diagnósticos de los cuatro especialistas y a la crítica. Además, se comprueba "
  "en la crítica que el arbitraje transcrito coincide con el recalculado, que las afirmaciones siguen el "
  "orden de dimensiones del arbitraje, que hay salvedades para las métricas cerradas y que la etiqueta "
  "coincide con la del orquestador. Una ejecución que falla se repite y el fallo se contabiliza.")

H("E4: coste, latencia y reproducibilidad (P4)", 3)
P("El coste y la latencia se miden sobre T100. La primera ejecución de cada proceso incluye la carga de "
  "los modelos de visión y se excluye de la latencia. La reproducibilidad se mide sobre R, con tres "
  "ejecuciones por imagen y la lectura visual reutilizada de la primera, de modo que la variabilidad "
  "medida es la de los especialistas y el crítico. Se compara si los informes transcritos coinciden con los "
  "de la herramienta, si el arbitraje y la etiqueta coinciden entre repeticiones y qué parte de las citas se "
  "repite. El coste y la latencia son descriptivos y no tienen umbral.")

H("Amenazas a la validez", 2)
P("Las siguientes circunstancias pueden sesgar los resultados y deben tenerse en cuenta al leerlos:")
V(["**Anotador único.** A1 y A2 las anota el autor, sin medida de acuerdo entre anotadores.",
   "**Umbrales ajustados sobre el mismo corpus.** Los umbrales del capítulo 6 se calibraron con estas "
   "imágenes de desarrollo, por lo que las tasas de E2 serán optimistas frente a fotografías nuevas.",
   "**Selección intencional de C2.** Contiene ejemplos claros de cada categoría y no representa las "
   "fotografías de un usuario cualquiera; la exactitud sobre C2 será optimista.",
   "**Umbral de rechazo elegido con C1.** La cobertura y la exactitud con umbral sobre C1 son optimistas; "
   "por eso se repiten sobre val.",
   "**O es near-OOD.** Aproxima, pero no equivale, a imágenes realmente fuera de distribución.",
   "**Tamaño de muestra.** Con 100 imágenes los intervalos de confianza son amplios.",
   "**Modelo de lenguaje no determinista.** T100 tiene una sola ejecución por imagen; la variabilidad solo "
   "se mide en R, con cinco imágenes y la lectura visual fija.",
   "**Anotación acotada.** Solo se anota el horizonte natural, no las horizontales estructurales (bordes "
   "de edificios, mesas), así que el horizonte se evalúa con una única definición.",
   "**Coste aproximado.** Se calcula con la tarifa de una fecha concreta y sin descontar la caché "
   "implícita de la API, por lo que es una cota superior.",
   "**Sin comparación externa.** No se compara con un agente único ni con otros sistemas."])

# =============================================================================================
# CAPÍTULO 9
# =============================================================================================
H("Resultados", 1)
P("Este capítulo presenta los resultados en el orden de los protocolos del apartado 8.5. Todas las "
  "ejecuciones se realizaron el 28 de septiembre de 2026 con el sistema congelado, y cada apartado termina "
  "con el veredicto de su criterio de éxito.")

H("Clasificador de contexto (E1)", 2)
P("Según el protocolo 8.5.1, el clasificador se evaluó primero sobre C1 sin umbral de rechazo. La "
  "exactitud es 0,851 (IC 95%: 0,823-0,875) y la F1 macro, 0,851. Estas cifras coinciden con las del "
  "apartado 5.1.5, lo que confirma que se evalúa exactamente el modelo desplegado. La {T:e1_clases} "
  "desglosa el resultado por clase y la {F:matriz_c1} muestra la matriz de confusión.")
tabla("e1_clases", "Precisión, recall y F1 por clase del clasificador de contexto sobre C1, sin umbral",
      ["Clase", "Precisión", "Recall", "F1", "n"],
      [["Animal", "0,919", "0,919", "0,919", "136"],
       ["Arquitectura", "0,792", "0,853", "0,822", "143"],
       ["Paisaje", "0,805", "0,868", "0,835", "152"],
       ["Producto", "0,798", "0,741", "0,769", "139"],
       ["Retrato", "0,951", "0,871", "0,909", "155"],
       ["Macro", "0,853", "0,850", "0,851", "725"]],
      [3.6, 2.6, 2.6, 2.6, 1.8], negrita_ultima=True)
figura("matriz_c1", "Matriz de confusión del clasificador de contexto sobre C1 (725 imágenes, sin umbral)",
       RES / "E1" / "matriz_C1.png", 9.5)
P("La clase con peor F1 es producto (0,769). La mitad de sus errores, 18 de 36, se clasifican como "
  "arquitectura; le siguen las confusiones de arquitectura con paisaje (15) y de paisaje con arquitectura "
  "(11). La {F:gradcam} compara un acierto y un error del par producto-arquitectura. En el error, la red "
  "atiende a las estructuras lineales repetidas del objeto, un patrón propio de las fachadas, y lo asigna a "
  "arquitectura con una confianza de 0,993. De los 18 errores de este par, 5 superan el umbral de 0,85, de "
  "modo que el rechazo no los detiene.")
figura("gradcam", "Grad-CAM de un acierto (arriba) y de un error producto → arquitectura (abajo) sobre C1",
       RES / "E1" / "gradcam_producto_arquitectura.png", 11)
P("Con el umbral de rechazo de 0,85 ({T:e1_umbral}), el sistema acepta el 72,6% de C1 y acierta el 94,3% "
  "de lo que acepta, frente al 85,1% sin umbral. El rechazo captura el 72,2% de los errores (78 de 108) a "
  "costa de descartar el 19,6% de los aciertos, y envía a «otro» el 64,8% de O. Sobre val, que no intervino "
  "en la elección del umbral, los resultados son equivalentes. Además, 0,85 es el mejor umbral en val según "
  "un criterio de ganancia marginal (el porcentaje adicional de O rechazado por cada punto de cobertura "
  "perdido); sobre C1 lo sería 0,80. El umbral, por tanto, no es un artefacto de haberlo elegido con C1.")
tabla("e1_umbral", "Efecto del umbral de rechazo 0,85 sobre C1 y sobre val, frente a O",
      ["Métrica", "C1 + O", "val + O"],
      [["Cobertura (imágenes aceptadas)", "72,6%", "70,3%"],
       ["Exactitud sobre las aceptadas", "0,943", "0,951"],
       ["Errores capturados por el rechazo", "72,2%", "76,4%"],
       ["Aciertos perdidos", "19,6%", "21,6%"],
       ["O enviado a «otro»", "64,8%", "64,8%"],
       ["AUROC (distribución propia frente a O)", "0,754", "0,735"]],
      [8.0, 3.5, 3.5])
P("Sobre C2, la etiqueta del sistema coincide con la anotación A1 en 99 de las 100 imágenes (0,99; IC 95%: "
  "0,946-0,998). El sistema acepta 90 imágenes y acierta todas ellas (1,00; IC 95%: 0,959-1,00), y envía "
  "a «otro» el 10% (IC 95%: 5,5-17,4%): las 9 imágenes anotadas como «otro» y una anotada como animal. "
  "Esta última es precisamente la imagen diseñada como empate entre animal y paisaje, en la que el "
  "clasificador duda (confianza 0,74). La confianza mediana en C2 es 0,984, frente a 0,965 en C1 y 0,765 en "
  "O. Por último, en las 100 ejecuciones de T100 el orquestador transcribe sin cambios la etiqueta del "
  "clasificador.")
P("**Veredicto.** La F1 mínima en C1 es 0,769 ≥ 0,75 (criterio descriptivo): se cumple. En C2, la "
  "exactitud sobre las aceptadas es 1,00 ≥ 0,85 y se envía a «otro» el 10% ≤ 20%: se cumplen. **P1: sí.**")

H("Condiciones de aplicabilidad (E2)", 2)
P("Según el protocolo 8.5.2, la réplica de las decisiones coincide con el código en las 600 decisiones (6 "
  "condiciones × 100 imágenes). La {T:e2_t1} muestra con qué frecuencia abre el sistema cada métrica en C2 "
  "y por qué la cierra. YOLO aporta el sujeto en 60 imágenes y el mapa de saliencia en 40; la vía sin sujeto "
  "claro no se activa en ninguna, aunque el diseño de C2 incluía paisajes sin horizonte pensados para ella, "
  "de modo que queda sin validar con fotografías reales. La cobertura media por dimensión es 0,90 en luz y "
  "tono, 0,70 en composición, 0,575 en espacio y 0,255 en líneas.")
tabla("e2_t1", "Métricas abiertas por el sistema en C2 y motivos de cierre (n = 100)",
      ["Métrica", "Abierta (IC 95%)", "Motivos de cierre"],
      [["esquema_cromatico", "80% (71-87)", "sin color analizable (20)"],
       ["angulo_horizonte", "26% (18-35)", "fragmento corto (58), sin candidatos (12), demasiado inclinado (4)"],
       ["score_convergencia", "25% (18-34)", "no supera el azar (31), haz casi paralelo (26), pocas líneas (18)"],
       ["ratio_espacio_negativo", "60% (50-69)", "sin identidad: sujeto por saliencia (40)"],
       ["ratio_nitidez", "55% (45-64)", "sin identidad (40), sin textura (5)"],
       ["d_tercios, d_centro, patron_dominante", "60% (50-69)", "sin identidad (40)"],
       ["media_L, d_equilibrio", "100%", "sin condición"]],
      [5.2, 3.0, 7.8])
P("La {T:e2_t2} cruza estas decisiones con la anotación. Con la definición amplia, que considera que la "
  "regla no aplica cuando la segmentación es solo parcial, los falsos positivos de espacio y nitidez suben "
  "a 31 y 30; el resto de condiciones no cambia.")
tabla("e2_t2", "Aciertos y errores de cada condición de aplicabilidad frente a A2 (definición estricta)",
      ["Condición", "VP", "FP", "FN", "VN", "Dud.", "Tasa FP (IC 95%)", "Tasa FN"],
      [["Cromática", "77", "1", "0", "20", "2", "4,8% (0,8-22,7)", "0%"],
       ["Horizonte", "9", "17", "10", "63", "1", "21,2% (13,7-31,4)", "52,6%"],
       ["Convergencia", "12", "13", "4", "70", "1", "15,7% (9,4-25,0)", "25,0%"],
       ["Espacio", "54", "5", "11", "29", "1", "14,7% (6,4-30,1)", "16,9%"],
       ["Nitidez", "49", "5", "16", "29", "1", "14,7% (6,4-30,1)", "24,6%"],
       ["Composición", "51", "8", "11", "29", "1", "21,6% (11,4-37,2)", "17,7%"]],
      [2.8, 1.1, 1.1, 1.1, 1.1, 1.1, 4.2, 1.9])
P("La condición cromática se comporta como se diseñó: un único falso positivo y ningún falso negativo. Las "
  "tres condiciones de identidad fallan en pocas imágenes, casi siempre las mismas: aquellas en las que "
  "YOLO detecta con confianza un objeto que no es el protagonista o en las que la segmentación falla (en un "
  "caso, la máscara queda vacía y ambas métricas de espacio se abren igualmente). Sus falsos negativos "
  "corresponden sobre todo a sujetos que COCO no reconoce, como un árbol, un faro o un edificio, una "
  "limitación ya declarada en el apartado 1.5. Las condiciones geométricas son las más débiles. La de horizonte abre la "
  "métrica en 26 imágenes, pero solo 9 tienen un horizonte natural: acepta como referencia cualquier "
  "horizontal larga de la escena, como la cornisa de un edificio, y deja fuera más de la mitad de los "
  "horizontes reales. La de convergencia abre 13 escenas que el anotador no considera organizadas por "
  "perspectiva. Reducir estos errores exigiría añadir reglas a cada condición, con la consiguiente pérdida "
  "de simplicidad, o sustituirlas por un detector aprendido, que no se ha evaluado en este trabajo.")
P("**Veredicto.** Las seis condiciones tienen algún falso positivo, entre 1 (cromática) y 17 (horizonte). "
  "**P2: no.**")

H("Fidelidad de las citas (E3)", 2)
P("Según el protocolo 8.5.3, el verificador analizó 3.580 citas en las 100 ejecuciones de T100, 35,8 por "
  "ejecución ({T:e3_autores}). Todas apuntan a un campo que existe en el informe y el 99,92% son exactas "
  "(IC 95%: 99,75-99,97%). Solo tres citas no lo son: dos truncadas y una alterada. Esta última se debe a "
  "que el informe transcrito por el LLM contiene un valor distinto del que calculó la herramienta (caso c "
  "del apartado 9.5).")
tabla("e3_autores", "Citas por autor en T100 (100 ejecuciones) y su clasificación",
      ["Autor", "Citas", "Exactas", "Truncadas", "Alteradas", "A cerradas"],
      [["Composición espacial", "511", "99,80%", "1", "0", "1"],
       ["Líneas y dirección", "317", "99,68%", "1", "0", "1"],
       ["Espacio y aislamiento", "332", "99,70%", "0", "1", "0"],
       ["Luz y tono", "914", "100%", "0", "0", "0"],
       ["Crítico", "1.506", "100%", "0", "0", "0"],
       ["Total", "3.580", "99,92%", "2", "1", "2"]],
      [4.4, 1.9, 2.1, 2.2, 2.2, 2.2], negrita_ultima=True)
P("Hay dos citas a métricas cerradas, ambas de especialistas y ambas a campos de evidencia que dependen "
  "de una métrica cerrada, no al valor de la métrica: un especialista de composición cita la posición del "
  "sujeto con la condición de identidad cerrada, y uno de líneas cita la apertura del haz con la "
  "convergencia cerrada. El crítico no cita ninguna métrica cerrada en las 100 críticas. Además, en 43 "
  "ocasiones cita W, la etiqueta de contexto o la cobertura con una notación distinta de la prevista, "
  "aunque siempre con el valor correcto.")
tabla("e3_critico", "Controles sobre las 100 críticas de T100",
      ["Control", "Cumple"],
      [["Afirmaciones en el orden de dimensiones del arbitraje", "100 / 100"],
       ["Etiqueta de contexto igual a la del orquestador", "100 / 100"],
       ["Salvedades suficientes para las métricas cerradas", "100 / 100"],
       ["Arbitraje transcrito igual al recalculado", "97 / 100"],
       ["Síntesis sin cifras fuera de las citas", "99 / 100"],
       ["Campos declarados coherentes con los citados", "75 / 100"]],
      [11.0, 3.0])
P("Los controles del crítico ({T:e3_critico}) muestran el mismo patrón. El orden de las afirmaciones, la "
  "etiqueta y las salvedades son correctos en todas las críticas, pero en tres de ellas el arbitraje "
  "transcrito no coincide con el recalculado: en dos, el crítico copia un vector W distinto del fijado por "
  "la herramienta y, en la tercera, un orden de dimensiones distinto. La transcripción de los informes de "
  "los especialistas es fiel en 398 de 400 informes. Por último, 3 de las 100 ejecuciones fallaron en el "
  "primer intento porque la crítica no superaba la validación del esquema de salida: dos incluían una "
  "tensión con una sola dimensión y otra, una afirmación sin campos citados. Al repetirlas, las tres se "
  "completaron.")
P("**Veredicto.** Hay 2 citas a métricas cerradas, frente a un máximo de 0: no se cumple. El 99,92% de "
  "citas exactas (≥ 95%) y el 100% de críticas con las afirmaciones en orden sí se cumplen. **P3: no.**")

H("Coste, latencia y reproducibilidad (E4)", 2)
P("Según el protocolo 8.5.4, un análisis completo tarda una mediana de 97,6 s (media 100,7 s; percentil "
  "95, 125,8 s; máximo 195 s; n = 98). De ese tiempo, el orquestador ocupa unos 13 s, los cuatro "
  "especialistas en paralelo unos 31 s y el crítico unos 53 s. La memoria de GPU máxima es de 342 MB de "
  "mediana (375 MB como máximo). La {T:e4_coste} desglosa los tokens y el coste, y la {F:latencia} "
  "representa la distribución de la latencia y el coste por agente.")
tabla("e4_coste", "Tokens y coste medios por análisis en T100",
      ["Modelo (agentes)", "Tokens de entrada", "Tokens de salida (razonamiento)", "USD por análisis"],
      [["Gemini 2.5 Flash (orquestador y 4 especialistas)", "74.326", "12.908 (9.508)", "0,055"],
       ["Gemini 2.5 Pro (crítico)", "22.572", "4.909 (3.226)", "0,077"],
       ["Gemini 2.5 Flash (lectura visual)", "745", "854", "0,002"],
       ["Total, sin descontar la caché", "", "", "0,134"],
       ["Total, descontando la caché implícita", "", "", "0,124"]],
      [6.4, 2.8, 3.8, 3.0], negrita_ultima=True)
figura("latencia", "Distribución de la latencia de un análisis en T100 (izquierda) y coste medio por agente (derecha)",
       RES / "E4" / "latencia_coste_T100.png", 16)
P("El crítico, que usa Gemini 2.5 Pro, supone el 58% del coste y más de la mitad del tiempo. El gasto "
  "estimado de toda la campaña experimental (piloto, T100 y R, incluidos los intentos fallidos) es de "
  "16,23 USD sin descontar la caché y de 14,83 USD descontándola; el cargo real registrado por la API fue "
  "de **[PENDIENTE: gasto real según el panel de Google AI Studio]**. Fallaron en el primer intento el 3% "
  "de las ejecuciones de T100 y 1 de las 15 de R, todas por la validación del esquema de la crítica.")
P("La {T:e4_repro} resume la reproducibilidad sobre R. Los informes transcritos coinciden con la "
  "herramienta en 59 de 60 casos: en una repetición, el LLM degeneró al transcribir un campo de texto del "
  "informe de composición y alteró otros dos. El arbitraje es idéntico en las tres repeticiones en cuatro de "
  "las cinco imágenes; en la quinta, una de las repeticiones transcribe un vector W y una cobertura "
  "distintos de los calculados. La etiqueta de contexto coincide siempre. En cuanto al texto, los "
  "especialistas repiten entre repeticiones buena parte de sus citas (índice de Jaccard medio entre 0,65 y "
  "0,91 según el especialista), mientras que el crítico elige en cada ejecución qué citar (0,29); aun así, "
  "las cifras que cita son exactas.")
tabla("e4_repro", "Reproducibilidad sobre R (cinco imágenes, tres repeticiones cada una)",
      ["Imagen", "Contexto", "Informes idénticos a la herramienta", "Arbitraje igual en las 3", "Etiqueta igual en las 3"],
      [["imagen62", "animal", "11 / 12", "sí", "sí"],
       ["imagen9", "arquitectura", "12 / 12", "sí", "sí"],
       ["imagen35", "paisaje", "12 / 12", "sí", "sí"],
       ["imagen86", "producto", "12 / 12", "sí", "sí"],
       ["imagen50", "retrato", "12 / 12", "no", "sí"]],
      [2.4, 2.8, 4.0, 3.4, 3.4])
P("**Veredicto.** El 98,3% de informes transcritos idénticos y el 80% de imágenes con el arbitraje idéntico "
  "no alcanzan el 100% exigido. **P4: no.** El coste, de unos 0,13 USD, y la latencia, de 98 s por "
  "análisis, son descriptivos.")

H("Casos ilustrativos", 2)
P("La {F:casos} reúne tres casos de T100 elegidos a partir de los resultados anteriores. Las imágenes son "
  "las de verificación que genera el propio sistema, con las medidas sobreimpresas.")
figura("casos", "Casos ilustrativos: (a) acierto típico (imagen38, composición y espacio); (b) falso negativo "
       "por la cobertura de COCO (imagen33, percepción y espacio); (c) error de transcripción del informe "
       "(imagen15, espacio)", RES / "E5" / "casos_ilustrativos.png", 16)
P("**(a) Acierto típico.** En este retrato, YOLO detecta a la persona, la segmentación la rodea y las "
  "condiciones de espacio y composición se abren con razón; las de horizonte y convergencia se cierran, y el "
  "crítico escribe una salvedad por cada una en lugar de dar una cifra. El contexto retrato sitúa luz y tono "
  "en primer lugar, y las afirmaciones siguen ese orden. Todas las citas son exactas, por ejemplo: «El joven "
  "de camisa de cuadros se encuentra fuertemente aislado del fondo, tanto por la gran proporción de espacio "
  "libre [ratio_espacio_negativo.valor = 0.7969] como por una marcada diferencia de nitidez que concentra el "
  "detalle en su figura [ratio_nitidez.valor = 0.9197]». La única tensión, entre el patrón de tercios y el "
  "equilibrio global, se resuelve citando W.")
P("**(b) Falso negativo por la cobertura de COCO.** El sujeto de este paisaje es un árbol solitario, una "
  "clase que YOLO no conoce. El mapa de saliencia lo localiza bien y la máscara lo rodea, pero, al no haber "
  "identidad garantizada, las condiciones de espacio y composición se cierran. La métrica se habría podido "
  "citar, así que es un falso negativo, pero el sistema calla en lugar de afirmar sin garantías: «Al no "
  "aislarse una figura principal del fondo, no es posible cuantificar el espacio negativo que la rodea ni "
  "comparar su nitidez relativa».")
P("**(c) Error de transcripción.** La herramienta de espacio calcula una diferencia de nitidez de -0,6248, "
  "visible en la imagen de verificación, y el diagnóstico la cita correctamente. Sin embargo, el informe que "
  "el LLM copia en la salida estructurada registra -0,0. El texto es correcto, pero el registro que "
  "debería servir de prueba no lo es. El error solo se detecta comparando con la herramienta, y muestra "
  "dónde reside el punto débil del diseño: en las copias que hace el modelo de lenguaje, no en el cálculo.")

H("Qué demuestran los resultados y limitaciones", 2)
P("La {T:resumen} resume el resultado de cada pregunta frente al criterio fijado en el capítulo 8.")
tabla("resumen", "Resumen de la validación: criterio, resultado y cifra clave de cada pregunta",
      ["Pregunta", "Criterio", "Resultado", "Cifra clave"],
      [["P1 Contexto", "F1 ≥ 0,75; exactitud de aceptadas ≥ 0,85; ≤ 20% a «otro»", "Sí", "F1 mínima 0,769; 90 de 90 aceptadas correctas; 10%"],
       ["P2 Cierres", "0 falsos positivos por condición", "No", "De 1 (cromática) a 17 (horizonte)"],
       ["P3 Citas", "0 a cerradas; ≥ 95% exactas; 100% en orden", "No (en parte)", "2 de 3.580 a cerradas; 99,92% exactas; 100% en orden"],
       ["P4 Coste y reproducibilidad", "100% de informes y arbitrajes idénticos", "No (en parte)", "0,13 USD y 98 s por análisis; 59/60 informes; 4/5 arbitrajes"]],
      [3.2, 5.0, 2.2, 5.6])
P("Los resultados demuestran que la parte determinista del sistema se comporta como se diseñó y puede "
  "auditarse. El clasificador asigna el contexto correcto y su rechazo captura la mayoría de sus errores. "
  "El arbitraje se puede recalcular, y la inmensa mayoría de las cifras de la crítica son copias exactas del "
  "informe: el crítico no cita ninguna métrica cerrada y respeta siempre el orden del arbitraje.")
P("También muestran dos debilidades. La primera está en las condiciones de aplicabilidad geométricas: la "
  "de horizonte y la de convergencia abren la métrica en escenas en las que la regla no aplica, y las de "
  "identidad dependen de que YOLO detecte al protagonista y no a otro objeto. La segunda está en el modelo de "
  "lenguaje como transcriptor: entre un 2% y un 3% de las ejecuciones copian mal un informe o el arbitraje, o "
  "producen una salida que el esquema rechaza. Los errores aparecen donde el LLM copia datos estructurados, "
  "no donde se calculan. Inyectar el informe directamente desde la herramienta, en lugar de pedir al modelo "
  "que lo transcriba, eliminaría esta fuente de error; es una propuesta que no se ha evaluado.")
P("Estas conclusiones están limitadas por las amenazas del apartado 8.6. Las que más pesan son que los "
  "umbrales se calibraron sobre el mismo corpus, que C2 es una selección intencional de 100 imágenes con "
  "anotador único y que la reproducibilidad se mide sobre solo cinco imágenes.")

doc.core_properties.title = "Capítulos 8 y 9 (borrador)"
doc.core_properties.author = "Edison Javier Tayan Gualoto"
doc.save(DESTINO)
print("guardado", DESTINO)
