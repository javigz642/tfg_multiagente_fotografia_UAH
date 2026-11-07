"""Manifiesto del conjunto de calibración: `experimentos/manifiesto_calibracion.csv`.

Una fila por foto de `data/` con nombre, MD5, dimensiones y, cuando se conoce, el origen en
Unsplash (ID, URL y autor). Fija qué 100 ficheros exactos se usaron para calibrar
(experimentos.md §3) y entra en la congelación (§9).

Las fotos de `data/` se renombraron a `imagen{N}.jpg`, así que el origen no está en el nombre.
Se recupera cruzándolas con las descargas originales de Unsplash, que se llaman
`{autor}-{ID}-unsplash.jpg`. Para cada foto se busca, en este orden:

    identica         mismo MD5 que un original.
    mismos_pixeles   MD5 distinto pero los mismos píxeles (solo cambian los metadatos).
    derivada_bn      la foto es un blanco y negro del original (luminancia casi idéntica,
                     correlación >= 0.999): es una foto DERIVADA y se declara.
    no_localizado    no hay original en las carpetas buscadas: URL y autor quedan vacíos.

El autor se guarda tal como aparece en el nombre del fichero (`autor_fichero`), sin adivinar
mayúsculas ni tildes. La URL `https://unsplash.com/photos/{ID}` lleva a la página de la foto.

Uso (desde la raíz del proyecto):
    uv run python experimentos/manifiesto_calibracion.py                     # busca en ~/Downloads
    uv run python experimentos/manifiesto_calibracion.py --originales DIR1 DIR2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
DIR_CALIBRACION = RAIZ / "data"
MANIFIESTO = RAIZ / "experimentos" / "manifiesto_calibracion.csv"

# `{autor}-{ID}-unsplash.jpg`, con un sufijo opcional si se descargó dos veces
# (`-unsplash-2.jpg`, `-unsplash_2.png`). Los ID de Unsplash tienen 11 caracteres.
PATRON_UNSPLASH = re.compile(
    r"^(?P<autor>.+)-(?P<id>[A-Za-z0-9_-]{11})-unsplash(?:[-_]\d+)?\.(?:jpe?g|png)$",
    re.IGNORECASE)
CORRELACION_MIN_BN = 0.999

CAMPOS = ["nombre", "md5", "ancho", "alto", "megapixeles", "origen", "licencia",
          "unsplash_id", "url", "autor_fichero", "fichero_original", "relacion_original",
          "transformacion", "notas"]


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def numero(ruta: Path) -> int:
    return int("".join(c for c in ruta.stem if c.isdigit()) or 0)


def rgb(ruta: Path) -> np.ndarray:
    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.int16)


def es_gris(pixeles: np.ndarray) -> bool:
    return bool((pixeles[..., 0] == pixeles[..., 1]).all()
                and (pixeles[..., 1] == pixeles[..., 2]).all())


def luminancia(ruta: Path) -> np.ndarray:
    return np.asarray(Image.open(ruta).convert("L"), dtype=np.float64).ravel()


def buscar_originales(carpetas: list[Path]) -> list[Path]:
    originales = []
    for carpeta in carpetas:
        if not carpeta.is_dir():
            print(f"AVISO: {carpeta} no existe, se ignora")
            continue
        originales += [p for p in carpeta.rglob("*")
                       if p.is_file() and PATRON_UNSPLASH.match(p.name)]
    return sorted(originales)


def emparejar(foto: Path, md5_foto: str, por_md5: dict, por_tamano: dict):
    """Devuelve (original, relación) o (None, "no_localizado")."""
    if md5_foto in por_md5:
        return por_md5[md5_foto], "identica"
    candidatos = por_tamano.get(Image.open(foto).size, [])
    if not candidatos:
        return None, "no_localizado"
    pixeles = rgb(foto)
    for original in candidatos:
        if np.array_equal(pixeles, rgb(original)):
            return original, "mismos_pixeles"
    if es_gris(pixeles):
        lum = luminancia(foto)
        for original in candidatos:
            if np.corrcoef(lum, luminancia(original))[0, 1] >= CORRELACION_MIN_BN:
                return original, "derivada_bn"
    return None, "no_localizado"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--originales", type=Path, nargs="+",
                        default=[Path.home() / "Downloads"],
                        help="carpetas donde buscar las descargas originales de Unsplash")
    args = parser.parse_args()

    fotos = sorted(DIR_CALIBRACION.glob("imagen*.jpg"), key=numero)
    originales = buscar_originales(args.originales)
    print(f"{len(fotos)} fotos en data/ · {len(originales)} descargas de Unsplash encontradas")

    por_md5, por_tamano = {}, {}
    for original in originales:
        por_md5.setdefault(md5(original), original)
        por_tamano.setdefault(Image.open(original).size, []).append(original)

    filas = []
    for foto in fotos:
        md5_foto = md5(foto)
        ancho, alto = Image.open(foto).size
        original, relacion = emparejar(foto, md5_foto, por_md5, por_tamano)
        fila = {
            "nombre": foto.name, "md5": md5_foto, "ancho": ancho, "alto": alto,
            "megapixeles": round(ancho * alto / 1e6, 2),
            "origen": "Unsplash", "licencia": "Unsplash License",
            "unsplash_id": "", "url": "", "autor_fichero": "", "fichero_original": "",
            "relacion_original": relacion, "transformacion": "", "notas": "",
        }
        if original is not None:
            m = PATRON_UNSPLASH.match(original.name)
            fila.update(unsplash_id=m["id"], url=f"https://unsplash.com/photos/{m['id']}",
                        autor_fichero=m["autor"], fichero_original=original.name)
        if relacion == "mismos_pixeles":
            fila["notas"] = "mismos píxeles que el original; solo cambian los metadatos"
        elif relacion == "derivada_bn":
            fila["transformacion"] = "convertida a blanco y negro"
            fila["notas"] = "foto DERIVADA del original en color"
        elif relacion == "no_localizado":
            fila["notas"] = ("original no localizado: origen según el manifiesto de la primera "
                             "campaña; URL y autor desconocidos")
        filas.append(fila)

    with MANIFIESTO.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)

    recuento = {}
    for fila in filas:
        recuento[fila["relacion_original"]] = recuento.get(fila["relacion_original"], 0) + 1
    print(f"Manifiesto: {MANIFIESTO}")
    print(f"Relación con el original: {recuento}")
    for fila in filas:
        if fila["relacion_original"] in ("derivada_bn", "no_localizado"):
            print(f"  {fila['nombre']:<14} {fila['relacion_original']}")


if __name__ == "__main__":
    main()
