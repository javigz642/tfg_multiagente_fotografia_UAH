"""Rutas del proyecto y directorios independientes por ejecución."""
from __future__ import annotations

from pathlib import Path

# `parents[2]` apunta a la raíz del proyecto.
PROYECT_PATH = Path(__file__).resolve().parents[2]
OUTPUTS_PATH = PROYECT_PATH / "outputs"

# Las ejecuciones son artefactos regenerables bajo `outputs/`.
EJECUCIONES_PATH = OUTPUTS_PATH / "ejecucion"


def dir_ejecucion(ruta_imagen: str | Path) -> Path:
    """Devuelve `outputs/ejecucion/{stem}` sin crear el directorio."""
    return EJECUCIONES_PATH / Path(ruta_imagen).stem


def ultima_ejecucion() -> Path | None:
    """Devuelve el directorio de ejecución más reciente, si existe."""
    if not EJECUCIONES_PATH.is_dir():
        return None
    candidatos = [d for d in EJECUCIONES_PATH.iterdir() if d.is_dir()]
    if not candidatos:
        return None
    return max(candidatos, key=lambda d: d.stat().st_mtime)
