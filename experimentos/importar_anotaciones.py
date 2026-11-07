"""Pasa la hoja `anotaciones/ANOTAR_AQUI.xlsx` a `A1_contexto.csv` y `A2_aplicabilidad.csv`.

La hoja de Excel es la fuente de las columnas de `comun.COLUMNAS_HOJA` (contexto y las 4 columnas
ciegas de A2): este script las sobrescribe en los CSV con lo que diga la hoja y no toca las demás
columnas (segmentación, protagonista, notas de A2). Las notas de la hoja van a `notas` de A1.

    .venv/Scripts/python.exe experimentos/importar_anotaciones.py
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
            if (p / "pyproject.toml").exists())
sys.path.insert(0, str(RAIZ / "experimentos"))

import pandas as pd  # noqa: E402

import comun  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    anot = comun.ruta_config("anotaciones")
    hoja = pd.read_excel(anot / "ANOTAR_AQUI.xlsx", sheet_name="Anotar", dtype=str).fillna("")
    hoja = hoja.set_index("imagen")
    esquemas = {"A1_contexto": comun.ESQUEMA_A1, "A2_aplicabilidad": comun.ESQUEMA_A2}

    problemas = []
    for nombre, columnas in comun.COLUMNAS_HOJA.items():
        ruta = anot / f"{nombre}.csv"
        csv = comun.leer_csv_robusto(ruta).set_index("imagen")
        assert sorted(csv.index) == sorted(hoja.index), f"{nombre}: las imágenes no coinciden con la hoja"
        cambios = 0
        for col in columnas:
            nuevos = hoja[col].map(comun.normalizar_valor).reindex(csv.index)
            invalidos = nuevos[(nuevos != "") & ~nuevos.isin(esquemas[nombre][col])]
            problemas += [f"{s}.{col} = {v!r}" for s, v in invalidos.items()]
            cambios += int((csv[col] != nuevos).sum())
            csv[col] = nuevos
        if nombre == "A1_contexto":
            csv["notas"] = hoja["notas"].reindex(csv.index).str.strip()
        csv.reset_index().to_csv(ruta, index=False, encoding="utf-8", lineterminator="\n")
        anotadas = {c: int((csv[c] != "").sum()) for c in columnas}
        print(f"{nombre}: {cambios} celdas actualizadas; anotadas {anotadas} de {len(csv)}")

    if problemas:
        print("Valores fuera de la lista (corrígelos en la hoja):\n  " + "\n  ".join(problemas))


if __name__ == "__main__":
    main()
