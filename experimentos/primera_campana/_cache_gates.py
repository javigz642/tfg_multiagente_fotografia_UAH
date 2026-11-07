"""Mide los informes del Nivel 2 y el arbitraje de cobertura sobre C2. Se ejecuta en un proceso
aparte (ver `comun.construir_cache("gates")`).

Para cada imagen de C2, con el código REAL del sistema y la percepción de la caché:
  - los cuatro informes (`analizar()` de cada engine), serializados como los devuelve cada tool;
  - el arbitraje de `PrioridadEngine` (métricas citables y cerradas, cobertura por dimensión),
    leyendo esos informes desde un directorio temporal igual que en una ejecución real.
No pasa por ningún LLM: son las `confianza` deterministas que E2 evalúa (experimentos.md §4.2-5).

Uso: python _cache_gates.py <raiz> <destino>
"""
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
RAIZ, DESTINO = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experimentos"))

import comun  # noqa: E402
from tfg_multiagente_fotografia.tools.composicion_espacial_tool import composicion_espacial_engine  # noqa: E402
from tfg_multiagente_fotografia.tools.espacio_aislamiento_tool import espacio_aislamiento_engine  # noqa: E402
from tfg_multiagente_fotografia.tools.lineas_direccion_tool import lineas_direccion_engine  # noqa: E402
from tfg_multiagente_fotografia.tools.luz_tono_tool import luz_tono_engine  # noqa: E402
from tfg_multiagente_fotografia.tools.prioridad_tool import INFORMES, PrioridadEngine  # noqa: E402


def informes_de(ruta, p):
    """Los cuatro informes de una imagen, con la percepción `p` de la caché."""
    ruta = str(ruta)
    return {
        "composicion_espacial": composicion_espacial_engine.analizar(
            ruta, p["bbox"], p["centroide_saliencia"], p["sujeto_discreto"], p["fuente"],
            p["mapa_saliencia_path"]),
        "lineas_direccion": lineas_direccion_engine.analizar(ruta),
        "espacio_aislamiento_sujeto": espacio_aislamiento_engine.analizar(
            ruta, p["bbox"], p["sujeto_discreto"], p["fuente"], p["mapa_saliencia_path"]),
        "luz_tono": luz_tono_engine.analizar(ruta),
    }


def main():
    percepcion = comun.cargar_cache("percepcion")["imagenes"]
    DESTINO.mkdir(parents=True, exist_ok=True)
    resultado = {}
    imagenes = comun.imagenes_C2()
    t0 = time.time()
    for k, ruta in enumerate(imagenes, 1):
        informes = {d: inf.model_dump(mode="json") for d, inf in informes_de(ruta, percepcion[ruta.stem]).items()}
        # Arbitraje con el código del crítico, leyendo los informes como en una ejecución real.
        # La etiqueta no afecta a la cobertura; "otro" solo fija la fila de W.
        with tempfile.TemporaryDirectory() as tmp:
            for dimension, fichero in INFORMES.items():
                (Path(tmp) / fichero).write_text(json.dumps({"informe": informes[dimension]}), encoding="utf-8")
            arbitraje = PrioridadEngine(raiz=Path(tmp)).arbitrar(ruta, "otro").model_dump(mode="json")
        resultado[ruta.stem] = {"informes": informes, "arbitraje": arbitraje}
        print(f"{k}/{len(imagenes)} {ruta.stem} ({time.time() - t0:.0f} s)", flush=True)

    meta = {**comun.meta_cache("gates"), "segundos": round(time.time() - t0, 1)}
    comun.guardar_json({"meta": meta, "imagenes": resultado}, DESTINO / "gates.json")
    print(f"Gates de {len(resultado)} imágenes guardados en {DESTINO} ({meta['segundos']:.0f} s)")


if __name__ == "__main__":
    main()
