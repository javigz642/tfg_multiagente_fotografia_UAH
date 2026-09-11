"""Prueba manual de SharedPerceptionTool."""
from pathlib import Path

from tfg_multiagente_fotografia.tools.shared_perception_tool import (
    SharedPerceptionTool,
    CONF_YOLO_THRESHOLD,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

tool = SharedPerceptionTool()

FUENTES_VALIDAS = ("yolo", "saliencia", "sin_sujeto_claro")


def assert_fichero_valido(ruta_str: str, descripcion: str):
    ruta = Path(ruta_str)
    assert ruta.exists(), f"{descripcion} no se guardó en disco: {ruta}"
    assert ruta.stat().st_size > 0, f"{descripcion} está vacío: {ruta}"


def validar_salida(image_path: Path, salida: dict):
    campos_esperados = {
        "bbox",
        "clase",
        "confianza_yolo",
        "fuente",
        "sujeto_discreto",
        "blob_area_ratio",
        "mapa_saliencia_path",
        "centroide_saliencia",
        "deteccion_yolo_path",
        "percepcion_bbox_path",
    }
    assert campos_esperados.issubset(salida.keys()), (
        f"Faltan campos en la salida: {campos_esperados - salida.keys()}"
    )

    assert salida["fuente"] in FUENTES_VALIDAS, f"fuente inesperada: {salida['fuente']}"
    assert isinstance(salida["bbox"], list) and len(salida["bbox"]) == 4, "bbox debe ser [cx, cy, w, h]"
    assert isinstance(salida["sujeto_discreto"], bool), "sujeto_discreto debe ser booleano"

    # Coherencia entre fuente y sujeto_discreto: es el contrato del que dependen
    # los agentes de Nivel 2 para decidir si anclan su análisis en el bbox.
    if salida["fuente"] == "yolo":
        assert salida["clase"] is not None, "fuente=yolo pero clase es None"
        assert salida["confianza_yolo"] is not None, "fuente=yolo pero confianza_yolo es None"
        assert salida["confianza_yolo"] > CONF_YOLO_THRESHOLD, (
            f"confianza_yolo {salida['confianza_yolo']} no supera el umbral {CONF_YOLO_THRESHOLD}"
        )
        assert salida["sujeto_discreto"] is True, "fuente=yolo debe implicar sujeto_discreto=True"
    else:
        assert salida["clase"] is None, f"fuente={salida['fuente']} pero clase no es None"
        assert salida["confianza_yolo"] is None, f"fuente={salida['fuente']} pero confianza_yolo no es None"
        esperado = salida["fuente"] == "saliencia"
        assert salida["sujeto_discreto"] is esperado, (
            f"fuente={salida['fuente']} incoherente con sujeto_discreto={salida['sujeto_discreto']}"
        )

    # El bbox debe caer dentro de la imagen (lo consumirá cv2.grabCut en el Agente 3)
    cx, cy, w, h = salida["bbox"]
    assert w > 0 and h > 0, f"bbox con dimensiones no positivas: {salida['bbox']}"
    assert cx - w / 2 >= -1 and cy - h / 2 >= -1, f"bbox se sale por arriba/izquierda: {salida['bbox']}"

    if salida["blob_area_ratio"] is not None:
        assert 0 < salida["blob_area_ratio"] <= 1, (
            f"blob_area_ratio fuera de [0, 1]: {salida['blob_area_ratio']}"
        )

    assert_fichero_valido(salida["mapa_saliencia_path"], "El mapa de saliencia")
    assert_fichero_valido(salida["deteccion_yolo_path"], "La imagen de detección YOLO")
    assert_fichero_valido(salida["percepcion_bbox_path"], "La imagen del bbox final")

    assert (
        isinstance(salida["centroide_saliencia"], list) and len(salida["centroide_saliencia"]) == 2
    ), "centroide_saliencia debe ser [x, y]"


def main():
    imagenes = sorted(DATA_DIR.glob("imagen*.jpg"))
    assert imagenes, f"No se encontraron imágenes de prueba en {DATA_DIR}"

    fallos = []
    conteo_fuentes = {f: 0 for f in FUENTES_VALIDAS}


    for image_path in imagenes:
        try:
            salida = tool._run(str(image_path))
            validar_salida(image_path, salida)
            conteo_fuentes[salida["fuente"]] += 1
            ratio = salida["blob_area_ratio"]
            ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
            print(
                f"[OK] {image_path.name}: fuente={salida['fuente']} "
                f"clase={salida['clase']} blob_ratio={ratio_str} "
                f"bbox={[round(v, 1) for v in salida['bbox']]}"
            )
        except AssertionError as e:
            fallos.append((image_path.name, str(e)))
            print(f"[FAIL] {image_path.name}: {e}")

    print("\n--- Resumen ---")
    print(f"{len(imagenes) - len(fallos)}/{len(imagenes)} imágenes OK")
    print("Distribución de fuentes: " + ", ".join(f"{k}={v}" for k, v in conteo_fuentes.items()))
    if fallos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
