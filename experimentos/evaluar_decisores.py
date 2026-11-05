"""Evalúa los decisores de aplicabilidad frente a la anotación humana (A2).

Lee la hoja Detalle de uno o varios Excel de `aplicabilidad_vlm.py` (uno por versión
del prompt), los cruza con la anotación y calcula, por condición y decisor: FP, FN,
tasa de FP con IC de Wilson al 95 %, cobertura, precisión, exactitud y κ de Cohen.
Compara además las versiones entre sí con la prueba exacta de McNemar.

Decisores: la heurística, el VLM de cada versión y la variante conservadora de cada
versión (abre solo si heurística y VLM coinciden). «dudoso» en el VLM cuenta como no
abrir; las imágenes anotadas como «dudoso» quedan fuera de la evaluación.

Uso:
    uv run python experimentos/evaluar_decisores.py \\
        --anotaciones experimentos/resultados/aplicabilidad_vlm/anotacion_humana_A2.csv \\
        --resultado v1=experimentos/resultados/aplicabilidad_vlm/aplicabilidad_vlm_gemini-2.5-flash.xlsx \\
        --resultado v2=experimentos/resultados/aplicabilidad_vlm/aplicabilidad_vlm_gemini-2.5-flash_v2.xlsx \\
        --particion prueba
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aplicabilidad_vlm import leer_anotaciones, particion  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    mitad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # El acotado solo absorbe el error de coma flotante en los extremos (k = 0 o k = n).
    return max(0.0, centro - mitad), min(1.0, centro + mitad)


def kappa(pred, real):
    po = (pred == real).mean()
    pe = pred.mean() * real.mean() + (1 - pred.mean()) * (1 - real.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def cargar(resultados, anotaciones, part):
    """Tabla larga: imagen, condición, verdad y una columna 0/1 por decisor."""
    base = None
    for nombre, ruta in resultados:
        d = pd.read_excel(ruta, sheet_name="Detalle")
        d = d[["imagen", "condicion", "heur_abre", "vlm_abre"]].rename(
            columns={"vlm_abre": f"gemini_{nombre}"}
        )
        d[f"heur∧gemini_{nombre}"] = d.heur_abre * d[f"gemini_{nombre}"]
        if base is None:
            base = d
        else:
            # La heurística no depende del prompt: debe coincidir en todos los ficheros.
            unido = base.merge(d, on=["imagen", "condicion"], suffixes=("", "_otro"))
            if not (unido.heur_abre == unido.heur_abre_otro).all():
                raise SystemExit(f"La heurística de {ruta} no coincide con la de las demás")
            base = unido.drop(columns="heur_abre_otro")
    base["verdad"] = [
        anotaciones.get(i, {}).get(c) for i, c in zip(base.imagen, base.condicion)
    ]
    base = base[base.verdad.isin(["si", "no"])].copy()
    base["verdad"] = (base.verdad == "si").astype(int)
    if part != "todas":
        base = base[[particion(Path(i)) == part for i in base.imagen]]
    return base


def metricas(base, decisores):
    filas = []
    for condicion, g in base.groupby("condicion", sort=False):
        y = g.verdad.to_numpy()
        negativos, positivos = int((y == 0).sum()), int((y == 1).sum())
        for decisor in decisores:
            p = g[decisor].to_numpy()
            fp, fn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
            tp = int(((p == 1) & (y == 1)).sum())
            lo, hi = wilson(fp, negativos)
            filas.append({
                "condicion": condicion, "decisor": decisor, "n": len(y),
                "aplica": positivos, "abre": int(p.sum()), "FP": fp, "FN": fn,
                "tasa_FP": fp / negativos if negativos else np.nan,
                "IC95_inf": lo, "IC95_sup": hi,
                "cobertura": tp / positivos if positivos else np.nan,
                "precision": tp / p.sum() if p.sum() else np.nan,
                "exactitud": (p == y).mean(),
                "kappa": kappa(p, y),
            })
    return pd.DataFrame(filas)


def mcnemar(base, a, b):
    filas = []
    for condicion, g in base.groupby("condicion", sort=False):
        acierta_a, acierta_b = g[a] == g.verdad, g[b] == g.verdad
        solo_a = int((acierta_a & ~acierta_b).sum())
        solo_b = int((~acierta_a & acierta_b).sum())
        p = binomtest(min(solo_a, solo_b), solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
        filas.append({"condicion": condicion, "A": a, "B": b, "solo_A_acierta": solo_a,
                      "solo_B_acierta": solo_b, "p_valor": p})
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--anotaciones", type=Path, required=True)
    parser.add_argument("--resultado", action="append", required=True,
                        help="nombre=ruta.xlsx (repetible, uno por versión del prompt)")
    parser.add_argument("--particion", choices=["todas", "desarrollo", "prueba"],
                        default="todas")
    parser.add_argument("--salida", type=Path, default=None,
                        help="CSV con las métricas (por defecto no se guarda)")
    args = parser.parse_args()

    resultados = [tuple(r.split("=", 1)) for r in args.resultado]
    base = cargar(resultados, leer_anotaciones(args.anotaciones), args.particion)
    decisores = ["heur_abre"] + [c for c in base.columns if c.startswith(("gemini_", "heur∧"))]
    tabla = metricas(base, decisores)

    vista = tabla.copy()
    for c in ("tasa_FP", "IC95_inf", "IC95_sup", "cobertura", "precision", "exactitud"):
        vista[c] = vista[c].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    vista["kappa"] = vista.kappa.round(2)
    pd.set_option("display.width", 250)
    print(f"Partición: {args.particion} · {base.imagen.nunique()} imágenes\n")
    print(vista.to_string(index=False))

    print("\nMcNemar exacto (aciertos):")
    pares = [("heur_abre", f"gemini_{n}") for n, _ in resultados]
    pares += [(f"gemini_{a}", f"gemini_{b}") for (a, _), (b, _) in combinations(resultados, 2)]
    print(pd.concat([mcnemar(base, a, b) for a, b in pares]).round(3).to_string(index=False))

    if args.salida:
        tabla.to_csv(args.salida, index=False)
        print(f"\nMétricas: {args.salida}")


if __name__ == "__main__":
    main()
