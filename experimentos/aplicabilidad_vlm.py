"""Aplicabilidad asistida por un VLM frente a las condiciones heurísticas (línea 10.4).

Hipótesis que se pone a prueba: las condiciones de aplicabilidad que dependen del
SIGNIFICADO de la escena (¿hay un horizonte natural?, ¿la perspectiva organiza la
composición?, ¿hay un protagonista?) las decide mejor un modelo de visión y lenguaje que
una heurística geométrica, aunque la MEDICIÓN siga en manos de las herramientas
deterministas.

Para cada imagen del corpus el script obtiene dos decisiones por condición:
  1. HEURÍSTICA — la `confianza` que publican los engines actuales, ejecutados
     directamente (no se leen los JSON de las ejecuciones, que son copias del LLM).
  2. VLM — Gemini mira la fotografía, ciego a las métricas y a los gates, y responde
     si / no / dudoso con un motivo por condición, siguiendo la misma definición
     operativa de «la regla aplica» que la guía de anotación A2.

El resultado es un Excel con:
  - Resumen:   por condición, acuerdo heurística–VLM y, si hay anotación, falsos
               positivos y negativos de la heurística, del VLM y de la variante
               conservadora (abre solo si ambos coinciden), con IC de Wilson al 95 %.
               Las cifras son FÓRMULAS: si se pega la anotación en la hoja Detalle, el
               resumen se recalcula al abrir el fichero.
  - Detalle:   una fila por imagen y condición, con la columna `anotacion` editable.
  - Por imagen: vista ancha con las variables de decisión, el protagonista que nombra el
               VLM, su IoU con la caja que eligió YOLO, tokens, coste y latencia.
  - Prompt y Parámetros: lo necesario para reproducir la ejecución.

Política ante «dudoso»: cuenta como NO abrir. Es el mismo criterio de calibración del
apartado 3.3 (preferir el falso negativo al falso positivo).

Uso (desde la raíz del proyecto):
    uv run python experimentos/aplicabilidad_vlm.py
    uv run python experimentos/aplicabilidad_vlm.py --anotaciones anotacion_A2.csv
    uv run python experimentos/aplicabilidad_vlm.py --limite 5          # prueba corta
    uv run python experimentos/aplicabilidad_vlm.py --modelo gemini-2.5-pro --forzar-vlm

El fichero de anotaciones (CSV o XLSX) lleva una columna `imagen` (p. ej. `imagen17`) y
una columna por condición con valores si / no / dudoso (también acepta 1 / 0):
    color, horizonte_natural, horizonte_amplio, convergencia, espacio, nitidez, composicion
Las columnas que falten se dejan en blanco.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import cv2
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pydantic import BaseModel, Field

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from tfg_multiagente_fotografia.rutas import OUTPUTS_PATH  # noqa: E402

DATA_PATH = RAIZ / "data"
CACHE_PATH = OUTPUTS_PATH / "aplicabilidad_vlm"
RESULTADOS_PATH = RAIZ / "experimentos" / "resultados" / "aplicabilidad_vlm"

MODELO_POR_DEFECTO = "gemini-2.5-flash"

# Temperatura 0: la decisión debe ser lo más estable posible entre ejecuciones. No la
# hace determinista (el servicio no lo garantiza), por eso existe --repeticion.
TEMPERATURA = 0.0

# Mismo envío que la lectura visual del crítico, para que el coste sea comparable.
LADO_MAYOR_ENVIO = 1024
CALIDAD_JPEG = 90

# Precio en USD por millón de tokens de gemini-2.5-flash (nivel de pago, entrada de
# texto/imagen y salida con razonamiento incluido). Compruébalo en la tarifa vigente
# antes de citar el coste en la memoria.
PRECIO_ENTRADA_MTOK = 0.30
PRECIO_SALIDA_MTOK = 2.50

MAX_REINTENTOS = 6

# IoU a partir del cual la caja de YOLO se considera la del protagonista.
IOU_PROTAGONISTA = 0.5

# La versión del prompt se elige con --version-prompt (ver PROMPTS).

# Las siete condiciones evaluadas. El horizonte aparece dos veces porque la guía A2
# distingue dos definiciones de «la regla aplica» (estricta y amplia), y la heurística
# es la misma en ambas.
CONDICIONES = [
    ("color", "esquema_cromatico"),
    ("horizonte_natural", "angulo_horizonte"),
    ("horizonte_amplio", "angulo_horizonte"),
    ("convergencia", "score_convergencia"),
    ("espacio", "ratio_espacio_negativo"),
    ("nitidez", "ratio_nitidez"),
    ("composicion", "d_tercios, d_centro, patron_dominante"),
]

PROMPT = """Vas a ver UNA fotografía. Un sistema de análisis compositivo mide sobre ella varias
métricas, pero cada métrica solo tiene sentido si la escena cumple una condición. Tu única
tarea es decidir, para cada condición, si se cumple en ESTA fotografía. No mides nada ni
valoras la foto.

Para cada condición responde "si", "no" o "dudoso", y da antes un motivo de una sola frase
en español que nombre lo que ves en la imagen. Usa "dudoso" solo si de verdad no puedes
decidir; ante la duda real, el sistema preferirá no medir.

1. color — ¿La imagen tiene color real con el que analizar su esquema cromático?
   SÍ aunque domine un solo matiz saturado (una foto azulada o anaranjada SÍ tiene esquema).
   NO si es blanco y negro, casi acromática, muy desaturada o una escena nocturna con solo
   un resto de color.

2. horizonte_natural — ¿Hay una línea de horizonte legible —la separación entre el cielo y
   la tierra o el mar, o el horizonte visible del paisaje— que cruce buena parte del
   encuadre y sobre la que tenga sentido juzgar si la foto está nivelada?
   NO si solo hay horizontales de objetos o construcciones (cornisas, mesas, bordes de
   muros, escalones), si el horizonte está tapado o si no hay ninguno.

3. horizontal_estructural — ¿Hay una línea horizontal de la propia escena (cornisa, borde
   de una mesa, línea del suelo, fachada, barandilla) que cruce buena parte del encuadre y
   sirva como referencia de nivelación? Responde independientemente de la pregunta 2.

4. perspectiva — ¿Hay perspectiva lineal que ORGANIZA la profundidad de la composición:
   líneas reales de la escena (calles, raíles, pasillos, aristas de edificios, filas de
   objetos) que convergen hacia un punto de fuga y estructuran la imagen?
   NO si solo hay líneas paralelas, texturas, vegetación, un muro plano, o una convergencia
   marginal que no organiza la foto.

5. protagonista — ¿Hay un sujeto principal concreto y delimitado (persona, animal, objeto,
   edificio o elemento aislado) que sea claramente el protagonista? Si lo hay, nómbralo en
   pocas palabras y da su caja como [ymin, xmin, ymax, xmax] normalizada a 0-1000. Si no
   lo hay, nombre vacío y caja vacía.

6. espacio — ¿Tiene sentido medir qué parte del encuadre ocupa el protagonista frente al
   espacio libre que lo rodea? Requiere un protagonista delimitado y separable del fondo.
   NO en paisajes sin foco, patrones, escenas con muchos elementos de igual peso o cuando
   el sujeto ocupa todo el encuadre sin borde reconocible.

7. nitidez — ¿Tiene sentido comparar la nitidez del protagonista con la del fondo?
   Requiere un protagonista delimitado y que alguna de las dos regiones tenga detalle o
   textura. NO si no hay protagonista o si toda la imagen es lisa (niebla, cielo, pared
   uniforme).

8. anclaje — ¿El protagonista es un elemento único y localizable cuya posición en el
   encuadre (cerca de los tercios o del centro) tenga sentido evaluar?
   NO si no hay protagonista, si hay varios con el mismo peso o si la composición la
   organiza una masa, un patrón o el paisaje entero.

Devuelve únicamente el objeto JSON del esquema."""


class Respuesta(str, Enum):
    si = "si"
    no = "no"
    dudoso = "dudoso"


class Decision(BaseModel):
    # El motivo va ANTES de la decisión a propósito: el modelo genera en el orden del
    # esquema, así que primero describe lo que ve y luego decide sobre ello.
    motivo: str = Field(..., description="Una frase que nombra lo que se ve en la imagen.")
    aplica: Respuesta


class Protagonista(BaseModel):
    motivo: str = Field(..., description="Una frase que justifica la respuesta.")
    existe: Respuesta
    nombre: str = Field(..., description="Nombre corto del protagonista, o vacío.")
    caja: list[int] = Field(
        ..., description="[ymin, xmin, ymax, xmax] normalizada a 0-1000, o lista vacía."
    )


class AplicabilidadVLM(BaseModel):
    color: Decision
    horizonte_natural: Decision
    horizontal_estructural: Decision
    perspectiva: Decision
    protagonista: Protagonista
    espacio: Decision
    nitidez: Decision
    anclaje: Decision


# --- Versión 2 del prompt ---------------------------------------------------------
# Iterada SOLO sobre la partición de desarrollo (imágenes impares, más 24 y 26), a partir
# de los errores de la v1 frente a la anotación humana. Cada cambio corrige un modo de
# fallo general, no una foto concreta:
#   - color: decidía por objetos sueltos de color en escenas acromáticas;
#   - horizontal estructural: aceptaba líneas en fuga y bordes cortos;
#   - perspectiva: aceptaba caminos curvos y fugas en el fondo de un retrato;
#   - protagonista: aceptaba elementos pequeños dentro de un paisaje y bodegones de
#     varios objetos;
#   - nitidez: tomaba una superficie lisa enfocada por «detalle», y exigía detalle en el
#     fondo cuando basta con una de las dos regiones;
#   - espacio: rechazaba sujetos grandes con borde visible.
# Además se pide primero una descripción de la escena (campo `escena`), para que las
# ocho decisiones se tomen sobre una lectura global y no pregunta a pregunta.
PROMPT_V2 = """Vas a ver UNA fotografía. Un sistema de análisis compositivo mide sobre ella varias
métricas, pero cada métrica solo tiene sentido si la escena cumple una condición. Tu única
tarea es decidir, para cada condición, si se cumple en ESTA fotografía. No mides nada ni
valoras la foto: decides si tiene SENTIDO medir. Que una regla aplique no significa que la
foto la cumpla bien (un horizonte torcido sigue siendo un horizonte).

Empieza por el campo "escena": una frase con el tipo de escena y qué ocupa el encuadre.
Después, para cada condición, un motivo de una frase que nombre lo que ves y la respuesta
"si", "no" o "dudoso". Usa "dudoso" solo si de verdad no puedes decidir.

1. color — ¿La imagen ENTERA tiene color con el que analizar un esquema cromático?
   Juzga el encuadre completo, no los objetos sueltos.
   SÍ si el color saturado ocupa una parte sustancial de la imagen, aunque domine un solo
   matiz (una foto azulada o anaranjada SÍ tiene esquema).
   NO si la escena es blanco y negro, o está dominada por blancos, grises, negros o tonos
   apagados y el color se reduce a acentos pequeños (una manzana sobre un mantel blanco,
   unos carteles en un túnel blanco, una vela en un fondo negro, frescos en un interior
   blanco). NO tampoco si es una dominante tenue tipo sepia o un virado desaturado.

2. horizonte_natural — ¿Hay una línea de horizonte: el límite lejano donde termina la
   tierra o el agua contra el cielo, casi recta, que cruza buena parte del encuadre?
   Puede estar inclinada o cortada por el sujeto.
   NO si lo que toca el cielo es un perfil irregular de montañas, una arboleda cercana o
   edificios, si no se ve cielo, o si la única horizontal es de un objeto.

3. horizontal_estructural — ¿Hay una línea REALMENTE horizontal de la escena (cornisa,
   borde de mesa, peldaño, línea entre pared y mesa, pretil, lamas) que se ve de frente,
   es continua y cruza MÁS DE LA MITAD del ancho?
   NO cuentan las líneas que se alejan en perspectiva (barandillas, bordes de un pasillo,
   tejados o aceras que convergen), los bordes cortos de objetos, ni las líneas curvas o
   diagonales. Responde con independencia de la pregunta 2.

4. perspectiva — ¿La composición está ORGANIZADA por perspectiva lineal: líneas RECTAS
   reales (calles, raíles, pasillos, aristas de edificios) que convergen hacia un punto de
   fuga y estructuran toda la imagen?
   NO si las líneas son curvas (una carretera que gira, un sendero), si la foto la
   organiza un sujeto (persona, animal, objeto) y la convergencia queda en el fondo, o si
   solo hay paralelas, texturas o vegetación.

5. protagonista — ¿Hay un sujeto concreto y delimitado que es claramente el centro de
   atención y la razón de la foto (persona, animal, objeto, edificio, árbol aislado)?
   Si lo hay, nómbralo y da su caja como [ymin, xmin, ymax, xmax] normalizada a 0-1000.
   NO si el sujeto de la foto es el paisaje o el lugar y el elemento es pequeño dentro de
   él (una casa en un valle, un poste en el agua), si es un bodegón o mesa con varios
   objetos de peso parecido, un patrón o una masa. Sin protagonista: nombre y caja vacíos.

6. espacio — ¿Tiene sentido medir qué parte del encuadre ocupa el protagonista frente al
   espacio libre? Requiere un protagonista con un BORDE reconocible contra el fondo.
   SÍ también si es grande (un retrato cercano, un sujeto que ocupa casi todo el
   encuadre): la métrica dirá que hay poco espacio libre, y esa medida es válida.
   NO si no hay protagonista, si es amorfo y sin borde (humo, fuegos artificiales), si
   se confunde con el fondo (camuflaje) o si es un grupo de objetos.

7. nitidez — ¿Tiene sentido comparar el detalle del protagonista con el del fondo?
   Detalle significa TEXTURA FINA visible: pelo, piel, plumas, tejido, hojas, grano,
   letras. Una superficie lisa aunque esté enfocada (cerámica, porcelana, plástico, cielo,
   pared) NO es detalle.
   SÍ si hay protagonista y AL MENOS UNA de las dos regiones (sujeto o fondo) tiene
   detalle; basta con una.
   NO si no hay protagonista, si es amorfo, o si sujeto y fondo son ambos lisos.

8. anclaje — ¿El protagonista es UN elemento con un centro definido cuya posición en el
   encuadre (tercios o centro) tenga sentido evaluar?
   NO si no hay protagonista, si es un grupo de objetos, o si es amorfo o está repartido
   por el encuadre (humo, fuegos artificiales).

Coherencia: si protagonista es "no", espacio, nitidez y anclaje son "no".

Devuelve únicamente el objeto JSON del esquema."""


class AplicabilidadVLMv2(BaseModel):
    # Sin herencia a propósito: el modelo genera en el orden de los campos y `escena`
    # tiene que ir el primero.
    escena: str = Field(..., description="Una frase: tipo de escena y qué ocupa el encuadre.")
    color: Decision
    horizonte_natural: Decision
    horizontal_estructural: Decision
    perspectiva: Decision
    protagonista: Protagonista
    espacio: Decision
    nitidez: Decision
    anclaje: Decision


PROMPTS = {1: (PROMPT, AplicabilidadVLM), 2: (PROMPT_V2, AplicabilidadVLMv2)}


# =============================================================================
# Decisión heurística (engines actuales)
# =============================================================================

def md5_fichero(ruta):
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


def cache_vigente(guardada, ruta):
    """Una entrada de caché vale si se calculó sobre ESTE fichero.

    Las entradas anteriores a esta comprobación no llevan `md5`: se aceptan y se les
    añade el actual, porque borrar la de una imagen sustituida basta para invalidarla.
    """
    actual = md5_fichero(ruta)
    if "md5" not in guardada:
        guardada["md5"] = actual
        return True
    return guardada["md5"] == actual


def decisiones_heuristicas(rutas, rehacer=False):
    """Ejecuta los cinco engines sobre cada imagen y devuelve sus gates por imagen.

    Se cachean en un JSON porque la percepción compartida (YOLO) y GrabCut tardan varios
    minutos sobre el corpus, y su resultado no cambia mientras no cambie el código ni la
    imagen (por eso cada entrada guarda el MD5 del fichero).
    """
    cache = CACHE_PATH / "heuristica.json"
    guardadas = {}
    if cache.exists() and not rehacer:
        guardadas = json.loads(cache.read_text(encoding="utf-8"))

    pendientes = [
        r for r in rutas if r.stem not in guardadas or not cache_vigente(guardadas[r.stem], r)
    ]
    if pendientes:
        # Importación diferida: cargar YOLO y torch solo si hay algo que calcular.
        from tfg_multiagente_fotografia.tools.composicion_espacial_tool import (
            composicion_espacial_engine,
        )
        from tfg_multiagente_fotografia.tools.espacio_aislamiento_tool import (
            espacio_aislamiento_engine,
        )
        from tfg_multiagente_fotografia.tools.lineas_direccion_tool import (
            LineasDireccionEngine,
        )
        from tfg_multiagente_fotografia.tools.luz_tono_tool import LuzTonoEngine
        from tfg_multiagente_fotografia.tools.shared_perception_tool import (
            SharedPerceptionEngine,
        )

        percepcion = SharedPerceptionEngine()
        luz, lineas = LuzTonoEngine(), LineasDireccionEngine()

        for i, ruta in enumerate(pendientes, 1):
            print(f"  heurística {i}/{len(pendientes)}: {ruta.name}", flush=True)
            p = percepcion.detect(ruta)
            lt = luz.analizar(ruta)
            ld = lineas.analizar(ruta)
            ea = espacio_aislamiento_engine.analizar(
                ruta, p["bbox"], p["sujeto_discreto"], p["fuente"], p["mapa_saliencia_path"]
            )
            ce = composicion_espacial_engine.analizar(
                ruta, p["bbox"], p["centroide_saliencia"], p["sujeto_discreto"],
                p["fuente"], p["mapa_saliencia_path"],
            )
            alto, ancho = cv2.imread(str(ruta)).shape[:2]
            guardadas[ruta.stem] = {
                "md5": md5_fichero(ruta),
                "dimensiones": [ancho, alto],
                "fuente": p["fuente"],
                "clase_yolo": p["clase"],
                "confianza_yolo": p["confianza_yolo"],
                "bbox": p["bbox"],
                "color": {
                    "abre": int(lt.esquema_cromatico.confianza >= 1.0),
                    "variable": f"ratio_cromatico={lt.ratio_pixeles_cromaticos:.3f}",
                    "motivo": lt.esquema_cromatico.fuente_confianza,
                },
                "horizonte": {
                    "abre": int(ld.angulo_horizonte.confianza >= 1.0),
                    "variable": (
                        f"longitud={ld.longitud_horizonte:.3f}, "
                        f"angulo={ld.angulo_horizonte.valor:+.2f}"
                    ),
                    "motivo": ld.angulo_horizonte.fuente_confianza,
                },
                "convergencia": {
                    "abre": int(ld.score_convergencia.confianza >= 1.0),
                    "variable": (
                        f"score={ld.score_convergencia.valor:.3f}, "
                        f"nulo={ld.score_convergencia_nulo:.3f}, "
                        f"apertura={ld.apertura_haz:.1f}"
                    ),
                    "motivo": ld.score_convergencia.fuente_confianza,
                },
                "espacio": {
                    "abre": int(ea.ratio_espacio_negativo.confianza >= 1.0),
                    "variable": f"fuente={p['fuente']}",
                    "motivo": ea.ratio_espacio_negativo.fuente_confianza
                    or "sujeto detectado por YOLO",
                },
                "nitidez": {
                    "abre": int(ea.ratio_nitidez.confianza >= 1.0),
                    "variable": (
                        f"fuente={p['fuente']}, var_max="
                        f"{max(ea.var_laplaciano_figura, ea.var_laplaciano_fondo):.1f}"
                    ),
                    "motivo": ea.ratio_nitidez.fuente_confianza
                    or "sujeto detectado por YOLO y escena con detalle",
                },
                "composicion": {
                    "abre": int(ce.d_tercios.confianza >= 1.0),
                    "variable": f"fuente={p['fuente']}",
                    "motivo": ce.d_tercios.fuente_confianza,
                },
            }
            # Se guarda tras cada imagen: una interrupción no obliga a repetir lo hecho.
            CACHE_PATH.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(guardadas, ensure_ascii=False, indent=1), "utf-8")
    elif guardadas:
        # Persiste los MD5 añadidos a entradas antiguas por `cache_vigente`.
        cache.write_text(json.dumps(guardadas, ensure_ascii=False, indent=1), "utf-8")

    return {r.stem: guardadas[r.stem] for r in rutas}


# =============================================================================
# Decisión del VLM (Gemini)
# =============================================================================

def preparar_imagen(ruta):
    img = cv2.imread(str(ruta))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    alto, ancho = img.shape[:2]
    escala = LADO_MAYOR_ENVIO / max(alto, ancho)
    if escala < 1.0:
        nuevo = (int(round(ancho * escala)), int(round(alto * escala)))
        img = cv2.resize(img, nuevo, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPEG])
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen: {ruta}")
    return buffer.tobytes()


def ruta_cache_vlm(ruta, modelo, repeticion, version):
    sufijo = f"_r{repeticion}" if repeticion else ""
    # La v1 vive en la raíz de la carpeta del modelo (es la caché original); cada versión
    # posterior tiene su subcarpeta, así que comparar versiones no pisa ninguna respuesta.
    carpeta = CACHE_PATH / modelo / (f"v{version}" if version > 1 else "")
    return carpeta / f"{ruta.stem}{sufijo}.json"


def consultar_vlm(cliente, ruta, modelo, repeticion, forzar, version):
    prompt, esquema = PROMPTS[version]
    destino = ruta_cache_vlm(ruta, modelo, repeticion, version)
    if destino.exists() and not forzar:
        guardada = json.loads(destino.read_text(encoding="utf-8"))
        tenia_md5 = "md5" in guardada
        if guardada.get("version_prompt") == version and cache_vigente(guardada, ruta):
            if not tenia_md5:
                destino.write_text(json.dumps(guardada, ensure_ascii=False, indent=2), "utf-8")
            guardada["desde_cache"] = True
            return guardada

    contenido = [
        types.Part.from_bytes(data=preparar_imagen(ruta), mime_type="image/jpeg"),
        types.Part.from_text(text=prompt),
    ]
    config = types.GenerateContentConfig(
        temperature=TEMPERATURA,
        response_mime_type="application/json",
        response_schema=esquema,
    )

    for intento in range(MAX_REINTENTOS):
        try:
            inicio = time.perf_counter()
            respuesta = cliente.models.generate_content(
                model=modelo, contents=contenido, config=config
            )
            latencia = time.perf_counter() - inicio
            datos = respuesta.parsed or esquema.model_validate_json(respuesta.text)
            break
        except errors.APIError as exc:
            # 429 (cuota) y 5xx (servicio saturado) son transitorios: se reintenta con
            # espera exponencial. Cualquier otro error es del script y debe verse.
            if exc.code not in (429, 500, 502, 503, 504) or intento == MAX_REINTENTOS - 1:
                raise
            espera = 2 ** intento * 5
            print(f"    {ruta.name}: error {exc.code}, reintento en {espera}s", flush=True)
            time.sleep(espera)
        except ValueError:
            # JSON que no valida contra el esquema: se repite la llamada.
            if intento == MAX_REINTENTOS - 1:
                raise
            time.sleep(2)

    uso = respuesta.usage_metadata
    tokens_entrada = uso.prompt_token_count or 0
    tokens_salida = (uso.candidates_token_count or 0) + (uso.thoughts_token_count or 0)
    resultado = {
        "imagen": ruta.stem,
        "md5": md5_fichero(ruta),
        "modelo": modelo,
        "version_prompt": version,
        "temperatura": TEMPERATURA,
        "repeticion": repeticion,
        "respuesta": json.loads(datos.model_dump_json()),
        "tokens_entrada": tokens_entrada,
        "tokens_salida": tokens_salida,
        "tokens_razonamiento": uso.thoughts_token_count or 0,
        "coste_usd": (tokens_entrada * PRECIO_ENTRADA_MTOK
                      + tokens_salida * PRECIO_SALIDA_MTOK) / 1e6,
        "latencia_s": round(latencia, 2),
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), "utf-8")
    resultado["desde_cache"] = False
    return resultado


def decisiones_vlm(rutas, modelo, hilos, repeticion, forzar, version):
    clave = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not clave:
        raise SystemExit("Falta GEMINI_API_KEY (o GOOGLE_API_KEY) en .env")
    cliente = genai.Client(api_key=clave)

    resultados = {}
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        futuros = {
            pool.submit(consultar_vlm, cliente, r, modelo, repeticion, forzar, version): r
            for r in rutas
        }
        for i, futuro in enumerate(as_completed(futuros), 1):
            ruta = futuros[futuro]
            try:
                resultados[ruta.stem] = futuro.result()
                origen = "caché" if resultados[ruta.stem]["desde_cache"] else "API"
                print(f"  VLM {i}/{len(rutas)}: {ruta.name} ({origen})", flush=True)
            except Exception as exc:  # una imagen fallida no tumba el corpus
                print(f"  VLM {i}/{len(rutas)}: {ruta.name} FALLÓ: {exc}", flush=True)
    return resultados


# =============================================================================
# Cruce de ambas decisiones
# =============================================================================

def iou(caja_a, caja_b):
    """IoU de dos cajas [x1, y1, x2, y2] en las mismas unidades."""
    ix1, iy1 = max(caja_a[0], caja_b[0]), max(caja_a[1], caja_b[1])
    ix2, iy2 = min(caja_a[2], caja_b[2]), min(caja_a[3], caja_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = lambda c: max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])  # noqa: E731
    union = area(caja_a) + area(caja_b) - inter
    return inter / union if union > 0 else 0.0


def iou_con_yolo(heur, protagonista):
    """IoU entre la caja que eligió YOLO y la del protagonista según el VLM.

    Ambas se llevan a coordenadas normalizadas [0, 1]: la de Gemini viene en 0-1000
    sobre la imagen enviada, que se reescaló de forma uniforme, así que la
    normalización coincide con la de la original.
    """
    caja = protagonista.get("caja") or []
    if heur["fuente"] != "yolo" or len(caja) != 4:
        return None
    ancho, alto = heur["dimensiones"]
    cx, cy, w, h = heur["bbox"]
    yolo = [(cx - w / 2) / ancho, (cy - h / 2) / alto, (cx + w / 2) / ancho, (cy + h / 2) / alto]
    ymin, xmin, ymax, xmax = (v / 1000 for v in caja)
    return iou(yolo, [xmin, ymin, xmax, ymax])


# Pregunta del prompt que responde a cada condición del cruce.
PREGUNTA = {"convergencia": "perspectiva", "composicion": "anclaje"}


def horizonte_amplio(natural, estructural):
    """Definición amplia de la guía A2: aplica si hay horizonte natural O una
    horizontal estructural. Se combina con la misma prudencia que el resto."""
    if "si" in (natural, estructural):
        return "si"
    if "dudoso" in (natural, estructural):
        return "dudoso"
    return "no"


def respuesta_vlm(vlm, condicion):
    """Decisión y motivo del VLM para una condición del cruce."""
    r = vlm["respuesta"]
    if condicion == "horizonte_amplio":
        decision = horizonte_amplio(
            r["horizonte_natural"]["aplica"], r["horizontal_estructural"]["aplica"]
        )
        motivo = (f"natural: {r['horizonte_natural']['motivo']} | "
                  f"estructural: {r['horizontal_estructural']['motivo']}")
        return decision, motivo
    clave = PREGUNTA.get(condicion, condicion)
    return r[clave]["aplica"], r[clave]["motivo"]


def leer_juez(ruta):
    """Lee el juicio de un segundo LLM (p. ej. juicio_claude.csv) con las mismas ocho
    preguntas del prompt, una columna por pregunta y un comentario por imagen."""
    if ruta is None:
        return {}
    df = pd.read_csv(ruta, dtype=str).fillna("")
    juez = {}
    for _, fila in df.iterrows():
        decisiones = {}
        for condicion, _ in CONDICIONES:
            if condicion == "horizonte_amplio":
                decisiones[condicion] = horizonte_amplio(
                    fila["horizonte_natural"], fila["horizontal_estructural"]
                )
            else:
                decisiones[condicion] = fila[PREGUNTA.get(condicion, condicion)]
        juez[Path(fila["imagen"]).stem] = {
            "decisiones": decisiones, "comentario": fila.get("comentario", "")
        }
    return juez


def clave_heuristica(condicion):
    return "horizonte" if condicion.startswith("horizonte") else condicion


def normalizar_anotacion(valor):
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip().lower()
    if texto in {"si", "sí", "s", "1", "1.0", "true", "verdadero", "x"}:
        return "si"
    if texto in {"no", "n", "0", "0.0", "false", "falso"}:
        return "no"
    if texto in {"dudoso", "?", "duda"}:
        return "dudoso"
    return None


def leer_anotaciones(ruta):
    if ruta is None:
        return {}
    ruta = Path(ruta)
    if ruta.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(ruta)
        # La plantilla de revisión lleva una segunda cabecera legible bajo la de máquina.
        if len(df) and str(df.iloc[0].get("imagen", "")).strip().lower() == "imagen":
            df = df.iloc[1:]
    else:
        # Los CSV de revisión del proyecto mezclan separador y codificación.
        for codificacion in ("utf-8-sig", "cp1252"):
            try:
                df = pd.read_csv(ruta, sep=None, engine="python", encoding=codificacion)
                break
            except UnicodeDecodeError:
                continue
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "imagen" not in df.columns:
        raise SystemExit(f"{ruta} no tiene columna 'imagen'")
    # La plantilla de revisión (plantilla_revision.py) usa las ocho PREGUNTAS del prompt;
    # se traducen aquí a las condiciones del cruce. También se aceptan directamente los
    # nombres de las condiciones.
    for condicion, pregunta in PREGUNTA.items():
        if condicion not in df.columns and pregunta in df.columns:
            df[condicion] = df[pregunta]
    anot = {}
    for _, fila in df.dropna(subset=["imagen"]).iterrows():
        stem = Path(str(fila["imagen"]).strip()).stem
        anot[stem] = {
            c: normalizar_anotacion(fila[c]) for c, _ in CONDICIONES if c in df.columns
        }
        if "horizonte_amplio" not in df.columns and {
            "horizonte_natural", "horizontal_estructural"
        } <= set(df.columns):
            natural = normalizar_anotacion(fila["horizonte_natural"])
            estructural = normalizar_anotacion(fila["horizontal_estructural"])
            # Sin las dos respuestas no hay decisión amplia: se deja en blanco.
            anot[stem]["horizonte_amplio"] = (
                horizonte_amplio(natural, estructural)
                if natural is not None and estructural is not None else None
            )
    return anot


def construir_tablas(rutas, heur, vlm, anotaciones, juez):
    detalle, ancha = [], []
    for ruta in rutas:
        stem = ruta.stem
        if stem not in vlm:
            continue
        h, v = heur[stem], vlm[stem]
        prot = v["respuesta"]["protagonista"]
        valor_iou = iou_con_yolo(h, prot)
        fila_ancha = {
            "imagen": stem,
            "fuente_percepcion": h["fuente"],
            "clase_yolo": h["clase_yolo"],
            "confianza_yolo": h["confianza_yolo"],
            "vlm_protagonista": prot["existe"],
            "vlm_protagonista_nombre": prot["nombre"],
            "iou_yolo_protagonista": None if valor_iou is None else round(valor_iou, 3),
            "yolo_es_protagonista": (
                None if valor_iou is None else ("si" if valor_iou >= IOU_PROTAGONISTA else "no")
            ),
        }
        for condicion, metrica in CONDICIONES:
            hc = h[clave_heuristica(condicion)]
            decision, motivo = respuesta_vlm(v, condicion)
            vlm_abre = int(decision == "si")
            detalle.append({
                "imagen": stem,
                "condicion": condicion,
                "metrica": metrica,
                "heur_abre": hc["abre"],
                "vlm": decision,
                "vlm_abre": vlm_abre,
                "ambos_abren": hc["abre"] * vlm_abre,
                "anotacion": anotaciones.get(stem, {}).get(condicion),
                "variable_decision": hc["variable"],
                "motivo_heuristica": hc["motivo"],
                "motivo_vlm": motivo,
            })
            # Las columnas del juez van al FINAL para no mover las letras de las que
            # dependen las fórmulas de la hoja Resumen (B, D, E, F, G y H).
            if juez:
                decision_juez = juez.get(stem, {}).get("decisiones", {}).get(condicion)
                detalle[-1]["juez"] = decision_juez
                detalle[-1]["juez_abre"] = None if decision_juez is None else int(
                    decision_juez == "si"
                )
                detalle[-1]["comentario_juez"] = juez.get(stem, {}).get("comentario")
                fila_ancha[f"{condicion}_juez"] = decision_juez
            fila_ancha[f"{condicion}_heur"] = hc["abre"]
            fila_ancha[f"{condicion}_vlm"] = decision
        for condicion in ("color", "horizonte_natural", "horizontal_estructural",
                          "perspectiva", "espacio", "nitidez", "anclaje"):
            fila_ancha[f"motivo_vlm_{condicion}"] = v["respuesta"][condicion]["motivo"]
        fila_ancha["motivo_vlm_protagonista"] = prot["motivo"]
        fila_ancha.update({
            "modelo": v["modelo"],
            "tokens_entrada": v["tokens_entrada"],
            "tokens_salida": v["tokens_salida"],
            "tokens_razonamiento": v["tokens_razonamiento"],
            "coste_usd": round(v["coste_usd"], 6),
            "latencia_s": v["latencia_s"],
        })
        ancha.append(fila_ancha)
    return pd.DataFrame(detalle), pd.DataFrame(ancha)


# =============================================================================
# Excel
# =============================================================================

CABECERA = PatternFill("solid", start_color="1F3864")
DESACUERDO = PatternFill("solid", start_color="FCE4D6")
EDITABLE = PatternFill("solid", start_color="FFF2CC")
BLANCO_NEGRITA = Font(bold=True, color="FFFFFF")


def escribir_tabla(ws, df, anchos=None, fila=1):
    for j, col in enumerate(df.columns, 1):
        celda = ws.cell(row=fila, column=j, value=col)
        celda.fill, celda.font = CABECERA, BLANCO_NEGRITA
        celda.alignment = Alignment(wrap_text=True, vertical="center")
    for i, registro in enumerate(df.itertuples(index=False), fila + 1):
        for j, valor in enumerate(registro, 1):
            ws.cell(row=i, column=j, value=None if pd.isna(valor) else valor)
    for j, col in enumerate(df.columns, 1):
        ancho = (anchos or {}).get(col, max(10, min(40, len(str(col)) + 2)))
        ws.column_dimensions[get_column_letter(j)].width = ancho
    ws.freeze_panes = ws.cell(row=fila + 1, column=2)
    ws.auto_filter.ref = f"A{fila}:{get_column_letter(len(df.columns))}{fila + len(df)}"


def wilson(fp, n, signo):
    """Extremo del IC de Wilson al 95 % como fórmula de Excel (mismo método que §9.2)."""
    p = f"({fp}/{n})"
    return (f'=IF({n}=0,"",({p}+1.96^2/(2*{n}){signo}1.96*SQRT({p}*(1-{p})/{n}'
            f'+1.96^2/(4*{n}^2)))/(1+1.96^2/{n}))')


def hoja_resumen(ws, n_filas, col_juez_abre=None):
    """Resumen por condición, todo con fórmulas sobre la hoja Detalle."""
    rango = lambda col: f"Detalle!${col}$2:${col}${n_filas + 1}"  # noqa: E731
    cond, heur, vabre, ambos, anot = (rango(c) for c in "BDFGH")
    vlm = rango("E")
    jabre = rango(col_juez_abre) if col_juez_abre else None

    ws["A1"] = "Aplicabilidad: condiciones heurísticas frente a un VLM"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = ("«dudoso» del VLM cuenta como no abrir. Los bloques de anotación se rellenan "
                "al escribir si/no/dudoso en la columna 'anotacion' de la hoja Detalle.")
    ws["A2"].font = Font(italic=True)

    columnas = [
        # (cabecera, plantilla de fórmula con {c} = celda de la condición)
        ("Condición", None),
        ("Métrica", None),
        ("N", f"=COUNTIFS({cond},{{c}})"),
        ("Abre heurística", f"=COUNTIFS({cond},{{c}},{heur},1)"),
        ("Abre VLM (si)", f"=COUNTIFS({cond},{{c}},{vabre},1)"),
        ("VLM dudoso", f'=COUNTIFS({cond},{{c}},{vlm},"dudoso")'),
        ("Abren ambos", f"=COUNTIFS({cond},{{c}},{ambos},1)"),
        ("Heur abre / VLM no", f"=COUNTIFS({cond},{{c}},{heur},1,{vabre},0)"),
        ("Heur cierra / VLM sí", f"=COUNTIFS({cond},{{c}},{heur},0,{vabre},1)"),
        ("Acuerdo", "=IF({N}=0,\"\",1-({HV}+{VH})/{N})"),
        ("κ de Cohen", "kappa"),
        ("Anotadas (si/no)", f'=COUNTIFS({cond},{{c}},{anot},"si")+COUNTIFS({cond},{{c}},{anot},"no")'),
        ("Aplica (anotación)", f'=COUNTIFS({cond},{{c}},{anot},"si")'),
        ("No aplica (anotación)", f'=COUNTIFS({cond},{{c}},{anot},"no")'),
        ("FP heurística", f'=COUNTIFS({cond},{{c}},{heur},1,{anot},"no")'),
        ("FN heurística", f'=COUNTIFS({cond},{{c}},{heur},0,{anot},"si")'),
        ("FP VLM", f'=COUNTIFS({cond},{{c}},{vabre},1,{anot},"no")'),
        ("FN VLM", f'=COUNTIFS({cond},{{c}},{vabre},0,{anot},"si")'),
        ("FP heur ∧ VLM", f'=COUNTIFS({cond},{{c}},{ambos},1,{anot},"no")'),
        ("FN heur ∧ VLM", f'=COUNTIFS({cond},{{c}},{ambos},0,{anot},"si")'),
        ("Tasa FP heur", "tasa:FP heurística"),
        ("IC95 inf heur", "lo:FP heurística"),
        ("IC95 sup heur", "hi:FP heurística"),
        ("Tasa FP VLM", "tasa:FP VLM"),
        ("IC95 inf VLM", "lo:FP VLM"),
        ("IC95 sup VLM", "hi:FP VLM"),
        ("Tasa FP heur ∧ VLM", "tasa:FP heur ∧ VLM"),
        ("IC95 inf ∧", "lo:FP heur ∧ VLM"),
        ("IC95 sup ∧", "hi:FP heur ∧ VLM"),
    ]
    if jabre:
        # Segundo LLM juez (Claude): NO es la verdad de referencia, sino otra opinión
        # independiente. Sirve para ver si dos modelos coinciden frente a la heurística.
        columnas += [
            ("Abre juez", f"=COUNTIFS({cond},{{c}},{jabre},1)"),
            ("Heur abre / juez no", f"=COUNTIFS({cond},{{c}},{heur},1,{jabre},0)"),
            ("Heur cierra / juez sí", f"=COUNTIFS({cond},{{c}},{heur},0,{jabre},1)"),
            ("VLM abre / juez no", f"=COUNTIFS({cond},{{c}},{vabre},1,{jabre},0)"),
            ("VLM cierra / juez sí", f"=COUNTIFS({cond},{{c}},{vabre},0,{jabre},1)"),
            ("Acuerdo heur–juez", "acuerdo:Heur abre / juez no|Heur cierra / juez sí"),
            ("Acuerdo VLM–juez", "acuerdo:VLM abre / juez no|VLM cierra / juez sí"),
        ]
    letra ={cab: get_column_letter(j) for j, (cab, _) in enumerate(columnas, 1)}
    fila0 = 4
    for j, (cab, _) in enumerate(columnas, 1):
        celda = ws.cell(row=fila0, column=j, value=cab)
        celda.fill, celda.font = CABECERA, BLANCO_NEGRITA
        celda.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        ws.column_dimensions[get_column_letter(j)].width = 22 if j == 2 else 12
    ws.column_dimensions["A"].width = 20
    ws.row_dimensions[fila0].height = 45

    for i, (condicion, metrica) in enumerate(CONDICIONES, fila0 + 1):
        ref = lambda cab: f"{letra[cab]}{i}"  # noqa: E731
        negativos = ref("No aplica (anotación)")
        for j, (cab, plantilla) in enumerate(columnas, 1):
            if cab == "Condición":
                valor = condicion
            elif cab == "Métrica":
                valor = metrica
            elif plantilla == "kappa":
                a, n = ref("Abren ambos"), ref("N")
                b, c = ref("Heur abre / VLM no"), ref("Heur cierra / VLM sí")
                d = f"({n}-{a}-{b}-{c})"
                po = f"(({a}+{d})/{n})"
                pe = f"((({a}+{b})*({a}+{c})+({c}+{d})*({b}+{d}))/{n}^2)"
                valor = f'=IF(OR({n}=0,{pe}=1),"",({po}-{pe})/(1-{pe}))'
            elif plantilla.startswith("acuerdo:"):
                a, b = (ref(x) for x in plantilla.split(":", 1)[1].split("|"))
                n = ref("N")
                valor = f'=IF({n}=0,"",1-({a}+{b})/{n})'
            elif plantilla.startswith(("tasa:", "lo:", "hi:")):
                tipo, origen = plantilla.split(":", 1)
                fp = ref(origen)
                if tipo == "tasa":
                    valor = f'=IF({negativos}=0,"",{fp}/{negativos})'
                else:
                    valor = wilson(fp, negativos, "-" if tipo == "lo" else "+")
            elif cab == "Acuerdo":
                valor = plantilla.format(
                    N=ref("N"), HV=ref("Heur abre / VLM no"), VH=ref("Heur cierra / VLM sí")
                )
            else:
                valor = plantilla.format(c=f"$A{i}")
            celda = ws.cell(row=i, column=j, value=valor)
            if cab.startswith(("Acuerdo", "κ de Cohen", "Tasa", "IC95")):
                celda.number_format = "0.0%" if cab != "κ de Cohen" else "0.00"
    ws.freeze_panes = "C5"

    nota = fila0 + len(CONDICIONES) + 2
    notas = [
        "Lectura:",
        "· «Heur abre / VLM no» son los candidatos a falso positivo de la heurística según el VLM.",
        "· FP = la métrica se abre y la anotación dice que la regla no aplica; tasa FP = FP / «No aplica».",
        "· heur ∧ VLM es la variante conservadora de 10.4: abre solo si ambos coinciden, "
        "así que solo puede reducir los FP de la heurística.",
        "· horizonte_natural usa la definición estricta de A2; horizonte_amplio acepta "
        "también una horizontal estructural.",
        "· Bloque «juez»: segundo LLM (Claude) con las mismas preguntas; es otra opinión, "
        "no la verdad de referencia. La anotación humana es la que decide los FP y FN.",
    ]
    for k, texto in enumerate(notas):
        ws.cell(row=nota + k, column=1, value=texto).font = Font(bold=(k == 0))


def escribir_excel(destino, detalle, ancha, parametros):
    wb = Workbook()
    resumen = wb.active
    resumen.title = "Resumen"
    col_juez_abre = (
        get_column_letter(list(detalle.columns).index("juez_abre") + 1)
        if "juez_abre" in detalle.columns else None
    )
    hoja_resumen(resumen, len(detalle), col_juez_abre)

    ws = wb.create_sheet("Detalle")
    escribir_tabla(ws, detalle, anchos={
        "imagen": 11, "condicion": 18, "metrica": 20, "variable_decision": 34,
        "motivo_heuristica": 60, "motivo_vlm": 70, "anotacion": 11,
        "comentario_juez": 70,
    })
    col_anot = list(detalle.columns).index("anotacion") + 1
    for i, fila in enumerate(detalle.itertuples(index=False), 2):
        ws.cell(row=i, column=col_anot).fill = EDITABLE
        if fila.heur_abre != fila.vlm_abre:
            for j in range(1, col_anot):
                ws.cell(row=i, column=j).fill = DESACUERDO

    ws = wb.create_sheet("Por imagen")
    anchos = {c: 45 for c in ancha.columns if c.startswith("motivo")}
    anchos.update({"vlm_protagonista_nombre": 28, "imagen": 11})
    escribir_tabla(ws, ancha, anchos=anchos)

    ws = wb.create_sheet("Prompt")
    ws.column_dimensions["A"].width = 110
    for i, linea in enumerate(PROMPTS[parametros["version_prompt"]][0].splitlines(), 1):
        ws.cell(row=i, column=1, value=linea)

    ws = wb.create_sheet("Parámetros")
    escribir_tabla(ws, pd.DataFrame(list(parametros.items()), columns=["parametro", "valor"]),
                   anchos={"parametro": 28, "valor": 70})

    destino.parent.mkdir(parents=True, exist_ok=True)
    wb.save(destino)


# =============================================================================
# Main
# =============================================================================

def numero_imagen(ruta):
    m = re.search(r"(\d+)$", ruta.stem)
    return int(m.group(1)) if m else 10**9


# Las imágenes 24, 25 y 26 son las 19, 20 y 21 giradas: van a la partición de su
# original para que un casi duplicado no quede a ambos lados de la separación.
GIRADAS = {24: 19, 25: 20, 26: 21}


def particion(ruta):
    """Desarrollo = impares (y las giradas de una impar); prueba = el resto.

    Alternar por número reparte por igual los contextos, porque el corpus está ordenado
    por categoría. Sobre `desarrollo` se itera el prompt; `prueba` solo se usa para la
    comparación final entre versiones.
    """
    n = numero_imagen(ruta)
    return "desarrollo" if GIRADAS.get(n, n) % 2 == 1 else "prueba"


def commit_actual():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def main():
    load_dotenv(RAIZ / ".env")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--carpeta", type=Path, default=DATA_PATH,
                        help="carpeta con las imágenes del corpus (por defecto data/)")
    parser.add_argument("--modelo", default=os.environ.get("MODEL", "").strip()
                        or MODELO_POR_DEFECTO)
    parser.add_argument("--anotaciones", type=Path, default=None,
                        help="CSV/XLSX con la anotación A2 (columna imagen + condiciones)")
    parser.add_argument("--juez", type=Path, default=None,
                        help="CSV con el juicio de un segundo LLM (p. ej. juicio_claude.csv)")
    parser.add_argument("--limite", type=int, default=None,
                        help="analizar solo las N primeras imágenes (prueba)")
    parser.add_argument("--hilos", type=int, default=4, help="consultas simultáneas al VLM")
    parser.add_argument("--version-prompt", type=int, default=1, choices=sorted(PROMPTS),
                        help="versión del prompt del VLM (la v1 es la original)")
    parser.add_argument("--particion", choices=["todas", "desarrollo", "prueba"],
                        default="todas",
                        help="desarrollo: impares, para iterar el prompt; prueba: pares, "
                             "solo para la comparación final")
    parser.add_argument("--repeticion", type=int, default=0,
                        help="índice de repetición: >0 guarda una caché aparte para medir "
                             "la reproducibilidad de las decisiones del VLM")
    parser.add_argument("--forzar-vlm", action="store_true",
                        help="ignorar la caché del VLM y volver a consultar")
    parser.add_argument("--rehacer-heuristica", action="store_true",
                        help="volver a ejecutar los engines aunque haya caché")
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    rutas = sorted(
        (p for p in args.carpeta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: (numero_imagen(p), p.name),
    )
    if args.particion != "todas":
        rutas = [r for r in rutas if particion(r) == args.particion]
    rutas = rutas[: args.limite]
    if not rutas:
        raise SystemExit(f"No hay imágenes en {args.carpeta}")
    print(f"{len(rutas)} imágenes ({args.particion}) · modelo {args.modelo} · "
          f"prompt v{args.version_prompt}")

    print("1/3 Decisiones heurísticas (engines)")
    heur = decisiones_heuristicas(rutas, rehacer=args.rehacer_heuristica)

    print("2/3 Decisiones del VLM")
    vlm = decisiones_vlm(rutas, args.modelo, args.hilos, args.repeticion, args.forzar_vlm,
                         args.version_prompt)
    fallidas = [r.stem for r in rutas if r.stem not in vlm]

    print("3/3 Excel")
    anotaciones = leer_anotaciones(args.anotaciones)
    juez = leer_juez(args.juez)
    detalle, ancha = construir_tablas(rutas, heur, vlm, anotaciones, juez)

    # El nombre de la v1 sobre todo el corpus se mantiene, para no romper las rutas ya
    # citadas; las demás combinaciones llevan versión y partición en el nombre.
    sufijo = "" if args.version_prompt == 1 else f"_v{args.version_prompt}"
    sufijo += "" if args.particion == "todas" else f"_{args.particion}"
    sufijo += f"_r{args.repeticion}" if args.repeticion else ""
    destino = args.salida or RESULTADOS_PATH / f"aplicabilidad_vlm_{args.modelo}{sufijo}.xlsx"
    parametros = {
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit_actual(),
        "modelo": args.modelo,
        "temperatura": TEMPERATURA,
        "version_prompt": args.version_prompt,
        "particion": args.particion,
        "lado_mayor_envio_px": LADO_MAYOR_ENVIO,
        "imagenes": len(rutas),
        "imagenes_fallidas_vlm": ", ".join(fallidas) or "ninguna",
        "anotaciones": str(args.anotaciones) if args.anotaciones else "sin anotación",
        "juez": str(args.juez) if args.juez else "sin juez",
        "politica_dudoso": "cuenta como no abrir",
        "iou_protagonista": IOU_PROTAGONISTA,
        "precio_entrada_usd_mtok": PRECIO_ENTRADA_MTOK,
        "precio_salida_usd_mtok": PRECIO_SALIDA_MTOK,
        "coste_total_usd": round(float(ancha["coste_usd"].sum()), 4) if len(ancha) else 0,
        "latencia_media_s": round(float(ancha["latencia_s"].mean()), 2) if len(ancha) else 0,
    }
    escribir_excel(destino, detalle, ancha, parametros)

    # Resumen en consola: lo mismo que la hoja Resumen, sin depender de abrir Excel.
    print()
    tabla = detalle.groupby("condicion", sort=False).apply(
        lambda g: pd.Series({
            "abre_heur": int(g.heur_abre.sum()),
            "abre_vlm": int(g.vlm_abre.sum()),
            "dudoso": int((g.vlm == "dudoso").sum()),
            "heur_si_vlm_no": int(((g.heur_abre == 1) & (g.vlm_abre == 0)).sum()),
            "heur_no_vlm_si": int(((g.heur_abre == 0) & (g.vlm_abre == 1)).sum()),
            "acuerdo": f"{(g.heur_abre == g.vlm_abre).mean():.0%}",
        }),
        include_groups=False,
    )
    print(tabla.to_string())
    if juez:
        tabla_juez = detalle.groupby("condicion", sort=False).apply(
            lambda g: pd.Series({
                "abre_juez": int(g.juez_abre.sum()),
                "heur_si_juez_no": int(((g.heur_abre == 1) & (g.juez_abre == 0)).sum()),
                "heur_no_juez_si": int(((g.heur_abre == 0) & (g.juez_abre == 1)).sum()),
                "acuerdo_heur_juez": f"{(g.heur_abre == g.juez_abre).mean():.0%}",
                "acuerdo_vlm_juez": f"{(g.vlm_abre == g.juez_abre).mean():.0%}",
            }),
            include_groups=False,
        )
        print("\nFrente al juez:")
        print(tabla_juez.to_string())
    if anotaciones:
        anotado = detalle[detalle.anotacion.isin(["si", "no"])]
        fp = anotado[anotado.anotacion == "no"].groupby("condicion", sort=False)[
            ["heur_abre", "vlm_abre", "ambos_abren"]].sum()
        print("\nFalsos positivos frente a la anotación:")
        print(fp.rename(columns={"heur_abre": "FP_heur", "vlm_abre": "FP_vlm",
                                 "ambos_abren": "FP_ambos"}).to_string())
    print(f"\nCoste VLM: {parametros['coste_total_usd']} USD · "
          f"latencia media {parametros['latencia_media_s']} s")
    if fallidas:
        print(f"Fallaron en el VLM: {', '.join(fallidas)} (vuelve a ejecutar para reintentar)")
    print(f"Excel: {destino}")


if __name__ == "__main__":
    main()
