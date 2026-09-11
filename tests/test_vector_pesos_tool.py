"""Prueba manual de VectorPesosTool."""
import json
from pathlib import Path

from tfg_multiagente_fotografia.tools.vector_pesos_tool import (
    PESOS_PATH,
    VectorPesosTool,
    vector_pesos_engine,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_TO_IDX_PATH = PROJECT_ROOT / "models" / "clasificador_contexto" / "class_to_idx.json"


COLUMNAS_ESPERADAS = [
    "composicion_espacial",
    "lineas_direccion",
    "espacio_aislamiento_sujeto",
    "luz_tono",
]

tool = VectorPesosTool()


def validar_claves():
    """Las claves de W = las 5 clases del clasificador + 'otro' (la de rechazo)."""
    with open(CLASS_TO_IDX_PATH, encoding="utf-8") as f:
        clases = set(json.load(f))

    esperadas = clases | {"otro"}
    reales = set(vector_pesos_engine.pesos)

    assert reales == esperadas, (
        f"Desincronización entre el clasificador y {PESOS_PATH.name}: "
        f"sobran {reales - esperadas or '{}'}, faltan {esperadas - reales or '{}'}"
    )
    print(f"[OK] claves sincronizadas con class_to_idx.json + 'otro': {sorted(reales)}")


def validar_columnas():
    """El orden de las columnas fija el significado de cada peso de la matriz."""
    assert vector_pesos_engine.columnas == COLUMNAS_ESPERADAS, (
        f"columnas={vector_pesos_engine.columnas}, se esperaba {COLUMNAS_ESPERADAS}"
    )
    print(f"[OK] columnas en el orden esperado: {COLUMNAS_ESPERADAS}")


def validar_filas():
    """Cada fila debe sumar 1 (lo impone VectorPesosW) y tener las 4 claves."""
    fallos = []
    for etiqueta in sorted(vector_pesos_engine.pesos):
        pesos = tool._run(etiqueta)
        try:
            assert set(pesos) == set(COLUMNAS_ESPERADAS), f"claves inesperadas: {set(pesos)}"
            total = sum(pesos.values())
            assert abs(total - 1.0) <= 1e-6, f"los pesos suman {total}, no 1"
            print(f"[OK] {etiqueta:<20} " + "  ".join(f"{k}={v}" for k, v in pesos.items()))
        except AssertionError as err:
            fallos.append((etiqueta, str(err)))
            print(f"[FAIL] {etiqueta}: {err}")
    return fallos


def validar_etiqueta_desconocida():
    """Una etiqueta que no existe debe fallar, no caer en 'otro' silenciosamente.

    Devolver pesos uniformes ante una etiqueta desconocida enmascararía justo el
    fallo de sincronización que este test busca.
    """
    try:
        tool._run("contexto_que_no_existe")
    except ValueError:
        print("[OK] una etiqueta desconocida lanza ValueError en vez de caer en 'otro'")
        return []
    return [("etiqueta_desconocida", "no lanzó ValueError con una etiqueta inexistente")]


def main():
    print(f"--- Vector de pesos W ({PESOS_PATH}) ---")
    validar_claves()
    validar_columnas()
    fallos = validar_filas()
    fallos += validar_etiqueta_desconocida()

    print("\n--- Resumen ---")
    print(f"{len(vector_pesos_engine.pesos) - len(fallos)}/{len(vector_pesos_engine.pesos)} filas OK")
    if fallos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
