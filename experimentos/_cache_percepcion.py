"""Construye la caché de percepción de C2. Se ejecuta en un proceso aparte (ver
`comun.construir_cache_percepcion`) para que el kernel del notebook no cargue YOLO.

Para cada imagen de C2 guarda, con el código REAL del sistema:
  - la percepción compartida (`SharedPerceptionEngine.detect`, Nivel 1);
  - todas las cajas de YOLO, no solo la elegida (para la calibración y E2);
  - la máscara del sujeto, con la misma bifurcación que `EspacioAislamientoEngine.analizar`.
Y en `meta` la firma de `comun.meta_cache("percepcion")` (MD5 de cada imagen, del código y las
constantes de los módulos de percepción y de los pesos de YOLO), para que `comun.vigencia_cache`
detecte cuándo queda obsoleta.

Deriva de la celda 4 del notebook 02. Uso: python _cache_percepcion.py <raiz> <destino>
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
RAIZ, DESTINO = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experimentos"))

import comun  # noqa: E402
from tfg_multiagente_fotografia.tools import espacio_aislamiento_tool as ea  # noqa: E402
from tfg_multiagente_fotografia.tools.imagen import cargar_imagen  # noqa: E402
from tfg_multiagente_fotografia.tools.shared_perception_tool import shared_perception_engine as spe  # noqa: E402


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    percepcion, mascaras = {}, {}
    imagenes = comun.imagenes_C2()
    t0 = time.time()
    for k, ruta in enumerate(imagenes, 1):
        d = spe.detect(str(ruta))                       # percepción compartida REAL
        original = cargar_imagen(ruta)
        res = spe.model(original, verbose=False)[0]     # todas las cajas
        d["cajas"] = [{"clase": spe.model.names[int(c)], "conf": float(f), "xywh": [float(v) for v in b]}
                      for c, f, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xywh)]
        trabajo = ea.reescalar(original)
        d["md5"] = comun.md5(ruta)
        d["shape_original"], d["shape_trabajo"] = list(original.shape[:2]), list(trabajo.shape[:2])
        if d["sujeto_discreto"]:                        # misma bifurcación que el engine
            d["rect"] = list(ea.rect_desde_bbox(d["bbox"], original.shape, trabajo.shape))
            mascara = ea.mascara_grabcut(trabajo, tuple(d["rect"]))
            d["fuente_mascara"] = "grabcut"
        else:
            d["rect"] = None
            mascara = ea.mascara_saliencia(d["mapa_saliencia_path"], trabajo.shape)
            d["fuente_mascara"] = "prior_saliencia"
        d["cobertura_mascara"] = float(mascara.mean())
        percepcion[ruta.stem], mascaras[ruta.stem] = d, mascara
        print(f"{k}/{len(imagenes)} {ruta.stem} ({time.time() - t0:.0f} s)", flush=True)

    meta = {**comun.meta_cache("percepcion"), "segundos": round(time.time() - t0, 1)}
    np.savez_compressed(DESTINO / "mascaras.npz", **mascaras)
    # percepcion.json se escribe el último: si existe, la caché está completa.
    comun.guardar_json({"meta": meta, "imagenes": percepcion}, DESTINO / "percepcion.json")
    print(f"Caché de {len(percepcion)} imágenes guardada en {DESTINO} ({meta['segundos']:.0f} s)")


if __name__ == "__main__":
    main()
