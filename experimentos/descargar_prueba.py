"""Selección y descarga del conjunto de prueba (40 fotos de Unsplash) en `data_prueba/`.

Aplica las reglas del §5.2 de `experimentos.md`:
  - una búsqueda por fila de la tabla del §5.1, con la consulta fijada aquí ANTES de buscar;
  - de los resultados, por orden de relevancia, se elige la PRIMERA foto que cumple
    claramente la descripción de la fila;
  - se descartan automáticamente: fotos Unsplash+ (de pago), fotos fuera de 2–26 MP
    (rango del conjunto de calibración) y fotos casi idénticas a alguna de `data/` o a otra ya elegida;
  - ninguna foto pasa por el sistema: aquí solo se calcula un hash perceptual con PIL.

Uso responsable de la API (https://unsplash.com/api-terms):
  - una única petición de búsqueda por fila, cacheada en disco (no se repite);
  - pausa entre peticiones y parada si `X-Ratelimit-Remaining` baja de un margen;
  - las imágenes se bajan del CDN (`images.unsplash.com`), que no consume cuota;
  - cada descarga dispara `links.download_location`, como exigen las directrices;
  - el manifiesto guarda autor y enlace para la atribución.

Subcomandos (desde la raíz del proyecto):
    uv run python experimentos/descargar_prueba.py buscar      # búsquedas pendientes + hojas de contacto
    uv run python experimentos/descargar_prueba.py hojas       # rehace las hojas sin llamar a la API
    uv run python experimentos/descargar_prueba.py descargar   # baja las elegidas en seleccion_prueba.json
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

RAIZ = Path(__file__).resolve().parents[1]
DIR_CALIBRACION = RAIZ / "data"
DIR_PRUEBA = RAIZ / "data_prueba"
DIR_CACHE = DIR_PRUEBA / "_seleccion"          # respuestas de la API, miniaturas y hojas (no versionado)
FICHERO_SELECCION = RAIZ / "experimentos" / "seleccion_prueba.json"
FICHERO_MANIFIESTO = RAIZ / "experimentos" / "manifiesto_prueba.csv"

API = "https://api.unsplash.com"
USER_AGENT = "tfg-multiagente-fotografia/1.0 (seleccion de conjunto de prueba)"
POR_PAGINA = 15
PAUSA_S = 2.0
MARGEN_CUOTA = 5
MP_MIN, MP_MAX = 2.0, 26.0
DIST_DUPLICADO = 6                          # distancia de Hamming máxima (dHash de 64 bits)
SEMILLA_NUMERACION = 2026                   # testN se asigna por permutación, no por fila

# fila, categoría, descripción (§5.1), consulta fijada antes de buscar
FILAS = [
    ("A1", "arquitectura", "Calle o pasillo con un punto de fuga claro", "street vanishing point perspective"),
    ("A2", "arquitectura", "Fachada frontal plana, sin perspectiva", "flat building facade front view"),
    ("A3", "arquitectura", "Contrapicado de un edificio alto (verticales que convergen)", "looking up at skyscraper"),
    ("A4", "arquitectura", "Interior con perspectiva (estación, iglesia, biblioteca)", "library interior perspective"),
    ("A5", "arquitectura", "Edificio aislado como protagonista (una casa sola)", "lone house isolated"),
    ("A6", "arquitectura", "Arquitectura en blanco y negro", "black and white architecture"),
    ("A7", "arquitectura", "Plaza o calle con una persona pequeña de paso", "person walking across empty plaza"),
    ("P1", "paisaje", "Mar o lago con el horizonte nivelado", "calm sea horizon"),
    ("P2", "paisaje", "Horizonte claramente inclinado (más de 10°)", "tilted horizon ocean"),
    ("P3", "paisaje", "Bosque cerrado o montaña sin horizonte visible", "dense forest"),
    ("P4", "paisaje", "Camino, vía o embarcadero que se aleja", "road leading into the distance"),
    ("P5", "paisaje", "Árbol solitario o faro", "lone tree in field"),
    ("P6", "paisaje", "Niebla o paisaje casi monocromo", "foggy landscape"),
    ("P7", "paisaje", "Prado amplio con un animal o una persona pequeños", "vast meadow small horse"),
    ("R1", "retrato", "Fondo desenfocado y pelo suelto", "portrait blurred background long hair"),
    ("R2", "retrato", "Fondo nítido y texturado (calle, ladrillo)", "portrait brick wall"),
    ("R3", "retrato", "Persona pequeña en un entorno amplio", "small person vast landscape"),
    ("R4", "retrato", "Retrato en blanco y negro", "black and white portrait"),
    ("R5", "retrato", "Clave baja, sujeto en parte en sombra", "low key portrait"),
    ("R6", "retrato", "Primer plano que llena el encuadre o cortado por los bordes", "close up face portrait"),
    ("R7", "retrato", "Dos o más personas", "group of friends portrait"),
    ("N1", "animal", "Animal de COCO (perro, gato, pájaro) con fondo desenfocado", "dog portrait bokeh"),
    ("N2", "animal", "Animal de COCO en un entorno amplio (caballo, vaca)", "cow in wide pasture"),
    ("N3", "animal", "Pelaje largo o plumas", "fluffy long haired cat"),
    ("N4", "animal", "Especie fuera de COCO (zorro, ciervo, lagarto)", "red fox"),
    ("N5", "animal", "Animal camuflado", "camouflaged animal"),
    ("N6", "animal", "Animal tapado en parte (valla, ramas)", "animal behind fence"),
    ("N7", "animal", "Grupo o rebaño", "flock of sheep"),
    ("D1", "producto", "Objeto de COCO aislado sobre fondo neutro (taza, botella, reloj)", "coffee mug plain background"),
    ("D2", "producto", "Blanco sobre blanco", "white product on white background"),
    ("D3", "producto", "Objeto oscuro sobre fondo negro", "black object on black background"),
    ("D4", "producto", "Bodegón con varios objetos", "still life objects"),
    ("D5", "producto", "Objeto fuera de COCO (joya, perfume, cámara)", "ring jewelry product photography"),
    ("D6", "producto", "Vidrio o superficie reflectante (copa, botella transparente)", "wine glass reflection"),
    ("D7", "producto", "Objeto rectangular que llena su caja (libro, cuadro, tableta)", "hardcover book on table"),
    ("O1", "otro", "Vehículo sin personas", "parked vintage car"),
    ("O2", "otro", "Deporte con varias personas en movimiento", "football match players"),
    ("O3", "otro", "Fuegos artificiales o escena nocturna abstracta", "fireworks night sky"),
    ("O4", "otro", "Textura que llena todo el encuadre (hojas, arena, un patrón)", "leaves texture full frame"),
    ("O5", "otro", "Escena sin dominante (persona y animal a la par, un mercado)", "busy street market"),
]
FILA = {f[0]: f for f in FILAS}

# Consultas adicionales, solo para las filas en las que ninguna candidata de la consulta
# anterior cumplía la descripción (se declaran en el manifiesto: la usada y las previas).
OTRAS_CONSULTAS = {
    "P2": ["dutch angle landscape", "dutch angle ocean", "tilted horizon sunset"],
    "D2": ["white ceramic on white background"],
    "D3": ["black bottle on black background"],
}


def consultas(fila: str) -> list[str]:
    return [FILA[fila][3]] + OTRAS_CONSULTAS.get(fila, [])


def _fichero_busqueda(fila: str, n: int) -> Path:
    return DIR_CACHE / (f"{fila}.json" if n == 1 else f"{fila}_{n}.json")


# ----------------------------------------------------------------------------- utilidades
def _clave() -> str:
    clave = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not clave:
        for linea in (RAIZ / ".env").read_text(encoding="utf-8").splitlines():
            nombre, _, valor = linea.partition("=")
            if nombre.strip() == "UNSPLASH_ACCESS_KEY":
                clave = valor.strip().strip("'\"")
    if not clave:
        sys.exit("Falta UNSPLASH_ACCESS_KEY en el entorno o en .env")
    return clave


def _api(ruta: str, params: dict | None = None) -> dict:
    """Una petición a la API con pausa y control de cuota."""
    url = f"{API}{ruta}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Client-ID {_clave()}",
        "Accept-Version": "v1",
        "User-Agent": USER_AGENT,
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        restante = int(r.headers.get("X-Ratelimit-Remaining", "999"))
        datos = json.load(r)
    print(f"    [API] {ruta}  cuota restante: {restante}")
    time.sleep(PAUSA_S)
    if restante < MARGEN_CUOTA:
        raise CuotaAgotada(restante)
    return datos


class CuotaAgotada(Exception):
    pass


def _bajar(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def dhash(img: Image.Image) -> int:
    g = ImageOps.exif_transpose(img).convert("L").resize((9, 8), Image.LANCZOS)
    p = list(g.tobytes())
    return sum(1 << i for i in range(64) if p[(i // 8) * 9 + i % 8] > p[(i // 8) * 9 + i % 8 + 1])


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def hashes_calibracion() -> dict[str, int]:
    cache = DIR_CACHE / "dhash_calibracion.json"
    if cache.exists():
        return {k: int(v) for k, v in json.loads(cache.read_text()).items()}
    hs = {}
    for f in sorted(DIR_CALIBRACION.glob("*.jp*g")):
        with Image.open(f) as im:
            im.draft("RGB", (256, 256))
            hs[f.name] = dhash(im)
    cache.write_text(json.dumps({k: str(v) for k, v in hs.items()}))
    return hs


def _candidatas(fila: str, n: int = 1) -> list[dict]:
    return json.loads(_fichero_busqueda(fila, n).read_text())["results"]


def _leer_seleccion() -> dict:
    return json.loads(FICHERO_SELECCION.read_text(encoding="utf-8")) if FICHERO_SELECCION.exists() else {}


# ----------------------------------------------------------------------------- buscar / hojas
def buscar() -> None:
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    pendientes = [(f[0], n, c) for f in FILAS for n, c in enumerate(consultas(f[0]), 1)
                  if not _fichero_busqueda(f[0], n).exists()]
    print(f"Búsquedas pendientes: {len(pendientes)}")
    try:
        for fila, n, consulta in pendientes:
            print(f"  {fila} (consulta {n}): «{consulta}»")
            datos = _api("/search/photos", {"query": consulta, "per_page": POR_PAGINA, "page": 1})
            datos["_consulta"] = consulta
            datos["_fecha"] = date.today().isoformat()
            _fichero_busqueda(fila, n).write_text(json.dumps(datos, ensure_ascii=False))
    except CuotaAgotada as e:
        print(f"Parada por cuota (quedan {e}). Vuelve a ejecutar dentro de una hora.")
    hojas()


def _filtros(c: dict, h: int | None, hs_calibracion: dict[str, int], hs_elegidas: dict[str, int]) -> list[str]:
    motivos = []
    if c.get("premium") or c.get("plus"):
        motivos.append("Unsplash+")
    mp = c["width"] * c["height"] / 1e6
    if not MP_MIN <= mp <= MP_MAX:
        motivos.append(f"{mp:.0f} MP")
    if h is not None:
        d2 = min(((hamming(h, v), k) for k, v in hs_calibracion.items()), default=(99, ""))
        if d2[0] <= DIST_DUPLICADO:
            motivos.append(f"dup calibracion {d2[1]}")
        for fila, v in hs_elegidas.items():
            if hamming(h, v) <= DIST_DUPLICADO:
                motivos.append(f"dup {fila}")
    return motivos


def hojas() -> None:
    """Una hoja de contacto por fila con las candidatas numeradas por relevancia."""
    hs_calibracion = hashes_calibracion()
    dir_min = DIR_CACHE / "miniaturas"
    dir_min.mkdir(exist_ok=True)
    resumen = {}
    for fila, n, cat, desc, consulta in [(f[0], n, f[1], f[2], c) for f in FILAS
                                         for n, c in enumerate(consultas(f[0]), 1)]:
        if not _fichero_busqueda(fila, n).exists():
            continue
        celdas = []
        for pos, c in enumerate(_candidatas(fila, n), 1):
            f_min = dir_min / f"{c['id']}.jpg"
            if not f_min.exists():
                f_min.write_bytes(_bajar(c["urls"]["small"]))   # CDN, sin cuota
            im = Image.open(f_min).convert("RGB")
            h = dhash(im)
            motivos = _filtros(c, h, hs_calibracion, {})
            celdas.append((pos, c, im, motivos, h))
        resumen.setdefault(fila, {}).update({c["id"]: str(h) for _, c, _, _, h in celdas})
        sufijo = "" if n == 1 else f"_{n}"
        lado, cols = 300, 5
        filas_g = (len(celdas) + cols - 1) // cols
        hoja = Image.new("RGB", (cols * lado, filas_g * (lado + 34) + 40), "white")
        d = ImageDraw.Draw(hoja)
        d.text((8, 10), f"{fila} [{cat}] {desc} — «{consulta}»", fill="black")
        for i, (pos, c, im, motivos, _) in enumerate(celdas):
            x, y = (i % cols) * lado, 40 + (i // cols) * (lado + 34)
            t = ImageOps.contain(im, (lado - 6, lado - 6))
            hoja.paste(t, (x + 3, y))
            mp = c["width"] * c["height"] / 1e6
            etiqueta = f"#{pos} {mp:.0f}MP {c['user']['username'][:14]}"
            d.text((x + 4, y + lado - 2), etiqueta, fill="black")
            if motivos:
                d.text((x + 4, y + lado + 14), "DESCARTE: " + ", ".join(motivos), fill="red")
        hoja.save(DIR_CACHE / f"hoja_{fila}{sufijo}.jpg", quality=85)
    (DIR_CACHE / "dhash_candidatas.json").write_text(json.dumps(resumen))
    print(f"Hojas en {DIR_CACHE}")


# ----------------------------------------------------------------------------- descargar
def rotar_y_recortar(im: Image.Image, grados: float) -> Image.Image:
    """Gira la imagen y la recorta al mayor rectángulo centrado con la misma proporción
    que cabe dentro, para que no queden esquinas vacías (sus bordes rectos falsearían el
    detector de líneas, la saliencia y el ratio cromático).

    Un rectángulo centrado s·w × s·h cabe en el original w × h girado θ si
    s·(w·cosθ + h·sinθ) ≤ w  y  s·(w·sinθ + h·cosθ) ≤ h.
    """
    import math
    w, h = im.size
    t = math.radians(abs(grados))
    s = min(w / (w * math.cos(t) + h * math.sin(t)), h / (w * math.sin(t) + h * math.cos(t)))
    girada = im.rotate(grados, resample=Image.BICUBIC, expand=True)
    W, H = girada.size
    a, b = int(s * w), int(s * h)
    x0, y0 = (W - a) // 2, (H - b) // 2
    return girada.crop((x0, y0, x0 + a, y0 + b))


def descargar() -> None:
    """Baja las fotos elegidas en seleccion_prueba.json y escribe el manifiesto."""
    sel = _leer_seleccion()
    hs_calibracion = hashes_calibracion()
    orden = [f[0] for f in FILAS]
    numeros = list(range(1, len(FILAS) + 1))
    random.Random(SEMILLA_NUMERACION).shuffle(numeros)
    nombre_de = {fila: f"test{n}" for fila, n in zip(orden, numeros)}

    filas_csv, hs_elegidas = [], {}
    try:
        for fila in orden:
            if fila not in sel:
                continue
            elegida = sel[fila]
            n = elegida.get("consulta_n", 1)
            c = next(x for x in _candidatas(fila, n) if x["id"] == elegida["id"])
            destino = DIR_PRUEBA / f"{nombre_de[fila]}.jpg"
            meta = DIR_CACHE / f"descarga_{fila}.json"
            grados = elegida.get("rotacion_grados")
            # Foto derivada: el original se guarda aparte y en data_prueba/ va la versión girada.
            original = DIR_CACHE / f"original_{fila}.jpg" if grados else destino
            if not original.exists():
                print(f"  {fila} → {destino.name}")
                url = _api(urllib.parse.urlparse(c["links"]["download_location"]).path,
                           dict(urllib.parse.parse_qsl(urllib.parse.urlparse(c["links"]["download_location"]).query)))["url"]
                original.write_bytes(_bajar(url))
                meta.write_text(json.dumps({"fecha": date.today().isoformat()}))
            if grados and not destino.exists():
                with Image.open(original) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    rotar_y_recortar(im, grados).save(destino, quality=95)
            datos = destino.read_bytes()
            md5_original = hashlib.md5(original.read_bytes()).hexdigest()
            with Image.open(io.BytesIO(datos)) as im:
                ancho, alto = ImageOps.exif_transpose(im).size
                h = dhash(im)
            motivos = _filtros(c, h, hs_calibracion, hs_elegidas)
            if motivos:
                print(f"  ¡AVISO {fila}! {motivos}")
            hs_elegidas[fila] = h
            filas_csv.append({
                "nombre": destino.name,
                "fila": fila,
                "categoria_objetivo": FILA[fila][1],
                "descripcion_fila": FILA[fila][2],
                "consulta": consultas(fila)[n - 1],
                "consultas_previas": " | ".join(consultas(fila)[: n - 1]),
                "posicion_en_resultados": elegida["posicion"],
                "unsplash_id": c["id"],
                "url": c["links"]["html"],
                "autor": c["user"]["name"],
                "autor_url": c["user"]["links"]["html"],
                "md5": hashlib.md5(datos).hexdigest(),
                "transformacion": (f"rotación de {grados}° (antihoraria) y recorte al mayor rectángulo "
                                   f"inscrito con la misma proporción; original {c['width']}x{c['height']}, "
                                   f"MD5 {md5_original}") if grados else "",
                "ancho": ancho,
                "alto": alto,
                "megapixeles": round(ancho * alto / 1e6, 2),
                "fecha_descarga": json.loads(meta.read_text())["fecha"] if meta.exists() else "",
                "descartes_previos": elegida.get("descartes", ""),
            })
    except CuotaAgotada as e:
        print(f"Parada por cuota (quedan {e}). Vuelve a ejecutar dentro de una hora.")
    if filas_csv:
        filas_csv.sort(key=lambda r: int(r["nombre"][4:-4]))
        with FICHERO_MANIFIESTO.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas_csv[0]))
            w.writeheader()
            w.writerows(filas_csv)
        print(f"Manifiesto: {FICHERO_MANIFIESTO} ({len(filas_csv)} fotos)")


if __name__ == "__main__":
    {"buscar": buscar, "hojas": hojas, "descargar": descargar}[sys.argv[1] if len(sys.argv) > 1 else "buscar"]()
