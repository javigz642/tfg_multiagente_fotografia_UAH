"""Lanzador reanudable de las tandas con API (experimentos.md §6.3, §6.4 y §7).

Uso, desde la raíz del proyecto y en la máquina Windows:

    uv run python experimentos/ejecutar_lote.py piloto
    uv run python experimentos/ejecutar_lote.py prueba --tope-eur 10
    uv run python experimentos/ejecutar_lote.py repeticiones --tope-eur 10
    uv run python experimentos/ejecutar_lote.py prueba --estado          # solo informa

Las tandas se definen en `config_experimentos.yaml` (`tandas:`). Antes de empezar comprueba que
cada foto tiene el MD5 de su manifiesto y, si la tanda lo exige, que el sistema coincide con la
congelación. Después, por cada repetición y foto:

1. borra `outputs/ejecucion/{foto}` (y la lectura visual guardada, si toca generarla de nuevo);
2. ejecuta el crew completo con los mismos inputs que `main.py` y con las trazas de CrewAI
   desactivadas (`crew.TRACING = False`);
3. mide el tiempo total, el fin de cada tarea (aproximado: el Nivel 2 va en paralelo), los
   tokens de cada LLM y de la lectura visual, la VRAM máxima y la RAM máxima del proceso
   (SAM corre en CPU y no aparece en la VRAM);
4. copia la ejecución a `outputs/experimentos/{tanda}/rep{r}/{foto}/` con un `registro.json`.

Es reanudable: una foto con `registro.json` en estado `ok` se salta. Las que fallaron se
repiten y el intento fallido se archiva en `{tanda}/fallidos/rep{r}/{foto}_intento{n}/` (cuenta
para la tasa de fallos y para el gasto). Se detiene ante un error de cuota, tras 3 errores
seguidos o si el gasto estimado de todas las tandas supera `--tope-eur`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

RAIZ = comun.preparar_sys_path()
os.chdir(RAIZ)

SALIDAS = ["salida_orquestador.json", "salida_composicion_espacial.json",
           "salida_lineas_direccion.json", "salida_luz_tono.json",
           "salida_espacio_aislamiento.json", "salida_critica.json"]
ERRORES_CUOTA = ("RESOURCE_EXHAUSTED", "RateLimitError", "quota", "billing")
MAX_ERRORES_SEGUIDOS = 3
MODOS_LECTURA = ("nueva", "cache", "nueva_en_primera")


class MonitorRAM:
    """RAM residente máxima del proceso mientras está activo (muestrea cada 0,2 s)."""

    def __init__(self, intervalo: float = 0.2):
        import psutil
        self._proceso, self._intervalo = psutil.Process(), intervalo
        self.maximo, self._parar = 0, threading.Event()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)

    def _muestrear(self):
        while not self._parar.is_set():
            self.maximo = max(self.maximo, self._proceso.memory_info().rss)
            self._parar.wait(self._intervalo)

    def __enter__(self):
        self.maximo = self._proceso.memory_info().rss
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        self._hilo.join()


def fotos_de(tanda: dict, cfg: dict) -> dict[str, Path]:
    """{foto: ruta} de la tanda, comprobadas contra el manifiesto de su conjunto."""
    todas = comun.fotos_conjunto(tanda["conjunto"])
    sel = tanda["imagenes"]
    if sel == "todas":
        return todas
    nombres = cfg[sel] if isinstance(sel, str) else sel
    faltan = [n for n in nombres if n not in todas]
    if faltan:
        sys.exit(f"Fotos que no están en el conjunto «{tanda['conjunto']}»: {faltan}")
    return {n: todas[n] for n in nombres}


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


def estado(nombre: str, reps: int, fotos: list[str], base: Path) -> None:
    for rep in range(1, reps + 1):
        cuenta = {"ok": 0, "error": 0, "incompleta": 0, "pendiente": 0}
        for f in fotos:
            reg = leer_registro(base / nombre / f"rep{rep}" / f / "registro.json")
            cuenta[reg["estado"] if reg else "pendiente"] += 1
        print(f"{nombre} rep{rep}: " + ", ".join(f"{k} {v}" for k, v in cuenta.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tanda", help="nombre de la tanda en config_experimentos.yaml")
    ap.add_argument("--tope-eur", type=float, default=None,
                    help="deja de lanzar si el gasto estimado de TODAS las tandas llega a este valor")
    ap.add_argument("--estado", action="store_true", help="solo informa de lo hecho y lo pendiente")
    ap.add_argument("--forzar", action="store_true",
                    help="ejecuta aunque el sistema no esté congelado (solo para pruebas)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    cfg = comun.cargar_config()
    tanda = (cfg.get("tandas") or {}).get(args.tanda)
    if tanda is None:
        sys.exit(f"No existe la tanda «{args.tanda}». Tandas: {list(cfg.get('tandas') or {})}")
    if tanda["lectura_visual"] not in MODOS_LECTURA:
        sys.exit(f"lectura_visual debe ser uno de {MODOS_LECTURA}")
    fotos, reps = fotos_de(tanda, cfg), int(tanda["repeticiones"])
    base = comun.ruta_config("tandas")
    precios = cfg.get("precios") or {}

    if args.estado:
        estado(args.tanda, reps, list(fotos), base)
        usd, n, sin = gasto_acumulado(base, precios)
        print(f"Gasto estimado de todas las tandas: {usd:.2f} USD en {n} ejecuciones"
              + (f" ({sin} sin precio: rellena `precios:`)" if sin else ""))
        return

    congelado = comun.comprobar_congelacion()
    if tanda.get("requiere_congelacion") and not congelado and not args.forzar:
        sys.exit("El sistema no está congelado (experimentos.md §9). Esta tanda cuenta como "
                 "resultado: congélalo antes o usa --forzar para una prueba.")
    usd_por_eur = precios.get("usd_por_eur")
    precios_completos = usd_por_eur is not None and all(
        m.get("entrada") is not None and m.get("salida") is not None
        for m in (precios.get("modelos") or {}).values())
    if args.tope_eur is not None and not precios_completos:
        sys.exit("--tope-eur necesita las tarifas en config_experimentos.yaml (precios:).")

    # Carga del sistema (tarda: CrewAI, YOLO, SAM y la ResNet se cargan con el primer uso).
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
    import torch

    import tfg_multiagente_fotografia.crew as crew_mod
    from tfg_multiagente_fotografia.main import construir_inputs
    from tfg_multiagente_fotografia.rutas import OUTPUTS_PATH, dir_ejecucion
    from tfg_multiagente_fotografia.tools import lectura_visual_tool as lv_mod

    crew_mod.TRACING = False
    proc = comun.procedencia()
    info_sistema = {
        "commit": proc["commit"], "arbol_limpio": proc["arbol_limpio"], "congelado": congelado,
        "maquina": proc["maquina"], "gpu": proc["gpu"],
        "MODEL": os.environ.get("MODEL"), "MODELO_CRITICO": crew_mod.MODELO_CRITICO,
        "FLAG_CRITICO_VE_IMAGEN": crew_mod.FLAG_CRITICO_VE_IMAGEN, "TRACING": crew_mod.TRACING,
        "lectura_visual": {"modelo": lv_mod.MODELO, "temperatura": lv_mod.TEMPERATURA,
                           "version_prompt": lv_mod.VERSION_PROMPT,
                           "modo": tanda["lectura_visual"]},
    }
    dir_lecturas = OUTPUTS_PATH / lv_mod.SUBCARPETA
    primera, errores_seguidos = True, 0

    for rep in range(1, reps + 1):
        for i, (foto, ruta) in enumerate(fotos.items(), 1):
            destino = base / args.tanda / f"rep{rep}" / foto
            previo = leer_registro(destino / "registro.json")
            if previo and previo["estado"] == "ok":
                continue
            if args.tope_eur is not None:
                usd, n, _ = gasto_acumulado(base, precios)
                media = usd / n if n else 0.0
                if (usd + media) / usd_por_eur > args.tope_eur:
                    print(f"Tope alcanzado: {usd / usd_por_eur:.2f} EUR gastados "
                          f"(+{media / usd_por_eur:.2f} por foto) frente a {args.tope_eur} EUR.")
                    return

            print(f"[{args.tanda} rep{rep} {i}/{len(fotos)}] {foto} ...", end=" ", flush=True)
            if previo:
                archivo = (base / args.tanda / "fallidos" / f"rep{rep}"
                           / f"{foto}_intento{previo['intento']}")
                shutil.rmtree(archivo, ignore_errors=True)
                archivo.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destino), str(archivo))
            shutil.rmtree(destino, ignore_errors=True)
            destino.mkdir(parents=True)
            ruta_rel = Path(comun.relativa(ruta))
            shutil.rmtree(dir_ejecucion(ruta_rel), ignore_errors=True)
            lectura = dir_lecturas / f"{foto}_lectura.json"
            uso_lectura = dir_lecturas / f"{foto}_lectura_uso.json"
            nueva = (tanda["lectura_visual"] == "nueva"
                     or (tanda["lectura_visual"] == "nueva_en_primera" and rep == 1))
            if nueva:
                lectura.unlink(missing_ok=True)
                uso_lectura.unlink(missing_ok=True)

            marcas = []
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            inicio = datetime.now(timezone.utc)
            t0 = time.perf_counter()

            def al_terminar_tarea(salida, t0=t0, marcas=marcas):
                marcas.append({"tarea": salida.name or salida.description[:60],
                               "agente": salida.agent,
                               "t_fin_s": round(time.perf_counter() - t0, 2)})

            error, crew = None, None
            with (open(destino / "log.txt", "w", encoding="utf-8") as log,
                  redirect_stdout(log), redirect_stderr(log), MonitorRAM() as ram):
                try:
                    crew = crew_mod.TfgMultiagenteFotografia().crew(task_callback=al_terminar_tarea)
                    crew.kickoff(inputs=construir_inputs(str(ruta_rel)))
                except Exception:
                    error = traceback.format_exc()
                    print(error)
            segundos = time.perf_counter() - t0

            # Tokens por instancia de LLM (cada agente tiene la suya; se agrupan si la comparten).
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

            # Copia de la ejecución, de la lectura visual y de los PNG de verificación.
            if dir_ejecucion(ruta_rel).is_dir():
                shutil.copytree(dir_ejecucion(ruta_rel), destino, dirs_exist_ok=True)
            for f in (lectura, uso_lectura):
                if f.exists():
                    shutil.copy2(f, destino / f.name)
            for png in OUTPUTS_PATH.glob(f"*/{foto}_*.png"):
                if "experimentos" not in png.parts:
                    shutil.copy2(png, destino / png.name)

            faltan = [s for s in SALIDAS if not (destino / s).exists()]
            est = "error" if error else "incompleta" if faltan else "ok"
            registro = {
                "tanda": args.tanda, "rep": rep, "imagen": foto, "conjunto": tanda["conjunto"],
                "md5_imagen": comun.md5(ruta), "estado": est,
                "intento": (previo or {}).get("intento", 0) + 1,
                "inicio": inicio.isoformat(timespec="seconds"), "segundos": round(segundos, 2),
                "primera_del_proceso": primera, "tareas": marcas,
                "tokens_llm": list(tokens_llm.values()), "tokens_lectura_visual": tokens_lv,
                "lectura_visual_desde_cache": tokens_lv is None,
                "vram_pico_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                 if torch.cuda.is_available() else None),
                "ram_pico_mb": round(ram.maximo / 2**20, 1),
                "salidas_que_faltan": faltan, "error": error, "sistema": info_sistema,
            }
            coste = comun.coste_usd(registro, precios)
            registro["coste_usd_estimado"] = coste
            comun.guardar_json(registro, destino / "registro.json")
            primera = False

            tok = sum(u["total_tokens"] for u in registro["tokens_llm"])
            print(f"{est} en {segundos:.0f} s, {tok} tokens"
                  + (f", ~{coste:.3f} USD" if coste is not None else ""))
            if est == "ok":
                errores_seguidos = 0
                continue
            errores_seguidos += 1
            if error and any(k.lower() in error.lower() for k in ERRORES_CUOTA):
                print("Error de cuota o de facturación de la API: me detengo. Revisa el saldo "
                      "y reanuda.")
                return
            if errores_seguidos >= MAX_ERRORES_SEGUIDOS:
                print(f"{errores_seguidos} errores seguidos: me detengo. Mira "
                      f"{comun.relativa(destino / 'log.txt')}.")
                return

    estado(args.tanda, reps, list(fotos), base)


if __name__ == "__main__":
    main()
