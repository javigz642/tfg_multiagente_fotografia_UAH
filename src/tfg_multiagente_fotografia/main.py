#!/usr/bin/env python
"""Entrada de terminal para ejecutar el análisis de una fotografía."""
import warnings

from tfg_multiagente_fotografia.crew import TfgMultiagenteFotografia
from tfg_multiagente_fotografia.rutas import dir_ejecucion

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def construir_inputs(ruta_imagen: str) -> dict:
    """Construye los inputs del crew y el directorio de ejecución de la imagen."""
    return {
        "ruta_imagen": ruta_imagen,
        "dir_ejecucion": dir_ejecucion(ruta_imagen).as_posix(),
    }


def run():
    """Ejecuta el crew sobre la imagen configurada."""
    # Imagen de prueba para la ejecución desde terminal.
    inputs = construir_inputs("data\\imagen39.jpg") 
    
    TfgMultiagenteFotografia().crew().kickoff(inputs=inputs)
