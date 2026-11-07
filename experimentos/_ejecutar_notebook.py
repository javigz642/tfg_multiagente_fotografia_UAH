"""Ejecuta las celdas de código de un notebook como un script, para validarlo sin Jupyter.

`nbconvert` y `nbclient` no están instalados en el entorno, así que esta es la forma de comprobar
que un notebook de `experimentos/` corre de principio a fin (por ejemplo, tras editarlo). Las
figuras se generan con el backend Agg, sin mostrarse, y `display()` se sustituye por `print()`. Si
una celda falla, dice cuál.

Uso (desde la raíz del repositorio):
    .venv/Scripts/python.exe experimentos/_ejecutar_notebook.py experimentos/E2_condiciones_aplicabilidad.ipynb
    .venv/Scripts/python.exe experimentos/_ejecutar_notebook.py experimentos/00_preparacion.ipynb --hasta 3

`--hasta N` ejecuta solo las N primeras celdas de código. Ojo: el notebook escribe sus resultados
como en una ejecución normal (resultados/, anotaciones/ si faltan las plantillas, etc.).
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--hasta", type=int, default=None, help="número de celdas de código a ejecutar")
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    celdas = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    celdas = celdas[:args.hasta] if args.hasta else celdas

    import IPython.display as ipd

    def mostrar(*objetos, **_):
        for o in objetos:
            print(getattr(o, "data", o))

    ipd.display = mostrar
    # Los notebooks se ejecutan desde su carpeta, como en Jupyter.
    os.chdir(args.notebook.resolve().parent)
    espacio = {"__name__": "__main__"}
    for i, codigo in enumerate(celdas, 1):
        codigo = codigo.replace("from IPython.display import display", "from IPython.display import display as _d")
        codigo = codigo.replace("display(", "mostrar(") if "display(" in codigo else codigo
        espacio["mostrar"] = mostrar
        try:
            exec(compile(codigo, f"{args.notebook.name}[celda {i}]", "exec"), espacio)
        except Exception:
            traceback.print_exc()
            print(f"\nFALLA la celda de código {i} de {len(celdas)} de {args.notebook.name}")
            sys.exit(1)
    print(f"\nOK: {len(celdas)} celdas de código de {args.notebook.name} ejecutadas sin errores")


if __name__ == "__main__":
    main()
