"""Lanzador reanudable de las tandas con API (E1e, E3, E4). Ver experimentos.md §6.4 y §15.

Uso, desde la raíz del repositorio y con el Python de `.venv`:

    .venv/Scripts/python.exe experimentos/ejecutar_lote.py piloto
    .venv/Scripts/python.exe experimentos/ejecutar_lote.py T100 --tope-eur 20
    .venv/Scripts/python.exe experimentos/ejecutar_lote.py R --tope-eur 20
    .venv/Scripts/python.exe experimentos/ejecutar_lote.py T100 --estado      # solo informa

Las tandas se definen en `config_experimentos.yaml` (`tandas:`). Por cada repetición e imagen:

1. borra `outputs/ejecucion/{stem}` (y la lectura visual guardada si la tanda pide una nueva);
2. ejecuta el crew completo con los mismos inputs que `main.py`;
3. mide el tiempo total, la marca de fin de cada tarea (aproximada: el Nivel 2 va en paralelo),
   los tokens de cada LLM, los de la lectura visual y la VRAM pica;
4. copia la ejecución a `outputs/experimentos/{tanda}/rep{r}/{stem}/` con un `registro.json`.

Es reanudable: una imagen con `registro.json` en estado `ok` se salta; las que fallaron se
repiten, y el intento fallido se archiva en `{tanda}/fallidos/rep{r}/{stem}_intento{n}/`. Se detiene ante un error de cuota, tras 3 errores seguidos o si el gasto estimado de
todas las tandas supera `--tope-eur`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
            if (p / "pyproject.toml").exists())
for d in (RAIZ / "src", RAIZ / "experimentos"):
    sys.path.insert(0, str(d))
os.chdir(RAIZ)

import comun  # noqa: E402

SALIDAS = ["salida_orquestador.json", "salida_composicion_espacial.json", "salida_lineas_direccion.json",
           "salida_luz_tono.json", "salida_espacio_aislamiento.json", "salida_critica.json"]
ERRORES_CUOTA = ("RESOURCE_EXHAUSTED", "RateLimitError", "quota", "billing")
MAX_ERRORES_SEGUIDOS = 3


def imagenes_de(tanda: dict, cfg: dict) -> list[str]:
    sel = tanda["imagenes"]
    if sel == "C2":
        return [p.stem for p in comun.imagenes_C2()]
    if isinstance(sel, str):
        lista = (cfg.get("listas") or {}).get(sel)
        if not lista:
            sys.exit(f"La lista «{sel}» está vacía en config_experimentos.yaml (listas:). Fíjala antes.")
        return list(lista)
    return list(sel)


def leer_registro(ruta: Path) -> dict | None:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gasto_acumulado(base: Path, precios: dict) -> tuple[float, int, int]:
    """(USD estimados, ejecuciones con coste, ejecuciones sin precio) de todas las tandas."""
    total, n, sin_precio = 0.0, 0, 0
    for r in base.glob("**/registro.json"):   # incluye los intentos fallidos archivados
        reg = leer_registro(r)
        if not reg:
            continue
        c = comun.coste_usd(reg, precios)
        if c is None:
            sin_precio += 1
        else:
            total, n = total + c, n + 1
    return total, n, sin_precio


def estado(nombre: str, reps: int, stems: list[str], base: Path) -> None:
    for rep in range(1, reps + 1):
        cuenta = {"ok": 0, "error": 0, "incompleta": 0, "pendiente": 0}
        for s in stems:
            reg = leer_registro(base / nombre / f"rep{rep}" / s / "registro.json")
            cuenta[reg["estado"] if reg else "pendiente"] += 1
        print(f"{nombre} rep{rep}: " + ", ".join(f"{k} {v}" for k, v in cuenta.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tanda", help="nombre de la tanda en config_experimentos.yaml (piloto, T100, R)")
    ap.add_argument("--tope-eur", type=float, default=None,
                    help="deja de lanzar si el gasto estimado de TODAS las tandas llega a este valor")
    ap.add_argument("--estado", action="store_true", help="solo informa de lo hecho y lo pendiente")
    ap.add_argument("--forzar", action="store_true", help="ejecuta aunque el sistema no esté congelado")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    cfg = comun.cargar_config()
    tanda = (cfg.get("tandas") or {}).get(args.tanda)
    if tanda is None:
        sys.exit(f"No existe la tanda «{args.tanda}». Tandas: {list(cfg.get('tandas') or {})}")
    stems, reps = imagenes_de(tanda, cfg), int(tanda["repeticiones"])
    base = comun.ruta_config("tandas")
    precios = cfg.get("precios") or {}

    if args.estado:
        estado(args.tanda, reps, stems, base)
        usd, n, sin = gasto_acumulado(base, precios)
        print(f"Gasto estimado de todas las tandas: {usd:.2f} USD en {n} ejecuciones"
              + (f" ({sin} sin precio: rellena `precios:`)" if sin else ""))
        return

    congelado = comun.comprobar_congelacion()
    if tanda.get("requiere_congelacion") and not congelado and not args.forzar:
        sys.exit("El sistema no está congelado (00_preparacion §7). Esta tanda cuenta como resultado: "
                 "congélalo antes o usa --forzar para una prueba.")
    usd_por_eur = precios.get("usd_por_eur")
    modelos = (precios.get("modelos") or {}).values()
    precios_completos = usd_por_eur is not None and all(
        m.get("entrada") is not None and m.get("salida") is not None for m in modelos)
    if args.tope_eur is not None and not precios_completos:
        sys.exit("--tope-eur necesita los precios y el tipo de cambio en config_experimentos.yaml (precios:).")

    # Carga del sistema (tarda: CrewAI, YOLO y la ResNet se cargan con el primer uso).
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
    import torch
    import tfg_multiagente_fotografia.crew as crew_mod
    from tfg_multiagente_fotografia.main import construir_inputs
    from tfg_multiagente_fotografia.rutas import dir_ejecucion
    from tfg_multiagente_fotografia.tools import lectura_visual_tool as lv_mod
    from tfg_multiagente_fotografia.tools.capa_verificacion import OUTPUTS_PATH

    crew_mod.TRACING = False
    proc = comun.procedencia()
    info_sistema = {
        "commit": proc["commit"], "arbol_limpio": proc["arbol_limpio"],
        "congelado": congelado, "MODEL": os.environ.get("MODEL"), "MODELO_CRITICO": crew_mod.MODELO_CRITICO,
        "FLAG_CRITICO_VE_IMAGEN": crew_mod.FLAG_CRITICO_VE_IMAGEN, "TRACING": crew_mod.TRACING,
        "lectura_visual": {"modelo": lv_mod.MODELO, "temperatura": lv_mod.TEMPERATURA,
                           "version_prompt": lv_mod.VERSION_PROMPT},
    }
    dir_lecturas = OUTPUTS_PATH / lv_mod.SUBCARPETA
    primera, errores_seguidos = True, 0

    for rep in range(1, reps + 1):
        for i, stem in enumerate(stems, 1):
            destino = base / args.tanda / f"rep{rep}" / stem
            previo = leer_registro(destino / "registro.json")
            if previo and previo["estado"] == "ok":
                continue
            if args.tope_eur is not None:
                usd, n, _ = gasto_acumulado(base, precios)
                media = usd / n if n else 0.0
                if (usd + media) / usd_por_eur > args.tope_eur:
                    print(f"Tope alcanzado: {usd / usd_por_eur:.2f} EUR gastados (+{media / usd_por_eur:.2f} "
                          f"por imagen) frente a {args.tope_eur} EUR. Me detengo.")
                    return

            print(f"[{args.tanda} rep{rep} {i}/{len(stems)}] {stem} ...", end=" ", flush=True)
            if previo:   # el intento fallido se archiva: cuenta para la tasa de fallos y para el gasto
                archivo = base / args.tanda / "fallidos" / f"rep{rep}" / f"{stem}_intento{previo['intento']}"
                shutil.rmtree(archivo, ignore_errors=True)
                archivo.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destino), str(archivo))
            shutil.rmtree(destino, ignore_errors=True)
            destino.mkdir(parents=True)
            ruta = Path("data") / f"{stem}.jpg"
            shutil.rmtree(dir_ejecucion(ruta), ignore_errors=True)
            lectura = dir_lecturas / f"{stem}_lectura.json"
            uso_lectura = dir_lecturas / f"{stem}_lectura_uso.json"
            if tanda["lectura_visual"] == "nueva":
                lectura.unlink(missing_ok=True)
                uso_lectura.unlink(missing_ok=True)

            marcas = []
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            inicio = datetime.now(timezone.utc)
            t0 = time.perf_counter()

            def al_terminar_tarea(salida, t0=t0, marcas=marcas):
                marcas.append({"tarea": salida.name or salida.description[:60], "agente": salida.agent,
                               "t_fin_s": round(time.perf_counter() - t0, 2)})

            error, crew = None, None
            with open(destino / "log.txt", "w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                try:
                    crew = crew_mod.TfgMultiagenteFotografia().crew(task_callback=al_terminar_tarea)
                    crew.kickoff(inputs=construir_inputs(str(ruta)))
                except Exception:
                    error = traceback.format_exc()
                    print(error)
            segundos = time.perf_counter() - t0

            # Tokens por instancia de LLM (cada agente tiene la suya; se agrupan por si la comparten).
            tokens_llm = {}
            for agente in (crew.agents if crew else []):
                llm = agente.llm
                if llm is None or not hasattr(llm, "get_token_usage_summary"):
                    continue
                u = tokens_llm.setdefault(id(llm), {"modelo": llm.model, "agentes": [],
                                                    **llm.get_token_usage_summary().model_dump()})
                u["agentes"].append(agente.role.strip())
            tokens_lv, uso = None, leer_registro(uso_lectura)
            if uso and datetime.fromisoformat(uso["generado"]) >= inicio.replace(microsecond=0):
                tokens_lv = uso

            # Copia de la ejecución, la lectura visual y los PNG de verificación.
            if dir_ejecucion(ruta).is_dir():
                shutil.copytree(dir_ejecucion(ruta), destino, dirs_exist_ok=True)
            for f in (lectura, uso_lectura):
                if f.exists():
                    shutil.copy2(f, destino / f.name)
            for png in OUTPUTS_PATH.glob(f"*/{stem}_*.png"):
                if "experimentos" not in png.parts:
                    shutil.copy2(png, destino / png.name)

            faltan = [s for s in SALIDAS if not (destino / s).exists()]
            est = "error" if error else "incompleta" if faltan else "ok"
            registro = {
                "tanda": args.tanda, "rep": rep, "imagen": stem, "estado": est,
                "intento": (previo or {}).get("intento", 0) + 1,
                "inicio": inicio.isoformat(timespec="seconds"), "segundos": round(segundos, 2),
                "primera_del_proceso": primera, "tareas": marcas,
                "tokens_llm": list(tokens_llm.values()), "tokens_lectura_visual": tokens_lv,
                "lectura_visual_desde_cache": tokens_lv is None,
                "vram_pico_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
                "salidas_que_faltan": faltan, "error": error, "sistema": info_sistema,
            }
            coste = comun.coste_usd(registro, precios)
            registro["coste_usd_estimado"] = coste
            comun.guardar_json(registro, destino / "registro.json")
            primera = False

            tok = sum(u["total_tokens"] for u in registro["tokens_llm"])
            print(f"{est} en {segundos:.0f} s, {tok} tokens" + (f", ~{coste:.3f} USD" if coste is not None else ""))
            if est == "ok":
                errores_seguidos = 0
                continue
            errores_seguidos += 1
            if error and any(k.lower() in error.lower() for k in ERRORES_CUOTA):
                print("Error de cuota o de facturación de la API: me detengo. Revisa el saldo y reanuda.")
                return
            if errores_seguidos >= MAX_ERRORES_SEGUIDOS:
                print(f"{errores_seguidos} errores seguidos: me detengo. Mira {comun.relativa(destino / 'log.txt')}.")
                return

    estado(args.tanda, reps, stems, base)


if __name__ == "__main__":
    main()
