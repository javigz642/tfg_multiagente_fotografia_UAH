"""Análisis de huecos de la calibración (paso 2.4 de experimentos/ESTADO.md).

Aplica a cada umbral la regla del §4.1 de experimentos.md, fijada el 2026-11-07 ANTES de
medir, y propone un valor. No toca el código del sistema: los valores se presentan al
usuario (paso 2.5) y solo después se aplican (paso 2.6).

Entradas (todas versionadas):
    resultados/calibracion/medidas_calibracion.csv        variables de decisión por foto
    anotaciones/segmentacion_calibracion.csv              segmentación y caja = protagonista
    resultados/aplicabilidad_vlm/anotacion_humana_A2.csv  aplicabilidad de las 100 fotos

Salidas (en resultados/calibracion/):
    propuesta_umbrales.csv     una fila por umbral: valor actual, propuesto, regla, VP/FP/FN/VN
    barrido_yolo.csv           identidad del sujeto según CONF_YOLO_THRESHOLD (solo se documenta)
    escalas_composicion.csv    percentiles de margen_patron y d_equilibrio
    huecos_*.png               una gráfica por umbral

La regla, en resumen (el texto completo está en el §4.1):
    1. Banda vacía: el corte va dentro de ella, redondeado y lo más cerca posible del extremo
       más exigente para abrir.
    2. Sin banda vacía: entre los cortes que dejan abrir como mucho el 10 % de las negativas,
       el de mayor sensibilidad; si empatan, el más estricto. Sensibilidad < 60 % = punto débil.
    3. Menos de 3 positivas o de 3 negativas: «comprobado, no calibrado».
    4. Umbrales que actúan juntos: el candidato es la pareja y se aplican 1-3.

Concreciones que la regla no dice y que este script fija (se declaran en la memoria):
    - «Redondeado»: con 2 cifras significativas; si ese valor se sale de la banda, con 3, y así
      sucesivamente. Es el formato de las constantes actuales (0.09, 0.13, 35, 50, 0.90).
    - «El más estricto» en una pareja: el que deja abrir menos negativas; si aún empatan, la
      mayor longitud y luego el menor ángulo.
    - Las fotos `dudoso` se excluyen (como dice el §4.1) y se cuentan aparte.

Uso (desde la raíz del proyecto; no necesita GPU ni las fotos):
    uv run python experimentos/analisis_calibracion.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibracion import (  # noqa: E402
    ANOTACION_APLICABILIDAD,
    DIR_RESULTADOS,
    MEDIDAS_CSV,
    SEGMENTACION_CSV,
    leer_csv,
)

PROPUESTA_CSV = DIR_RESULTADOS / "propuesta_umbrales.csv"
BARRIDO_YOLO_CSV = DIR_RESULTADOS / "barrido_yolo.csv"
ESCALAS_CSV = DIR_RESULTADOS / "escalas_composicion.csv"

# Regla del §4.1.
MAX_TASA_FP = 0.10          # se descartan los cortes que abren más de 1 de cada 10 negativas
MIN_SENSIBILIDAD = 0.60     # por debajo, el corte se declara punto débil (pero no se cambia)
MIN_CASOS = 3               # menos positivas o negativas que esto: «comprobado, no calibrado»

# Valores con los que se midió (entorno_calibracion.json).
ACTUAL = {
    "RATIO_CROMATICO_MIN": 0.09,
    "LONGITUD_MIN_HORIZONTE": 0.13,
    "ANGULO_MAX_HORIZONTE": 10.0,
    "APERTURA_MIN_GRADOS": 35.0,
    "SAM_SCORE_MIN": 0.90,
    "RELLENO_MIN": 0.20,
    "RELLENO_MAX": 0.95,
    "VAR_LAPLACIANO_MIN": 50.0,
    "CONF_YOLO_THRESHOLD": 0.75,
    "BLOB_AREA_MIN_RATIO": 0.005,
    "BLOB_AREA_MAX_RATIO": 0.60,
}
# Escalas de lectura de agents.yaml (Agente 1).
ESCALAS_ACTUALES = {"margen_patron": (0.01, 0.15), "d_equilibrio": (0.03, 0.12)}

# Gráficas: paleta categórica validada (azul / naranja) y marcador distinto por clase, para
# que la clase no dependa solo del color.
AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#8a8a85"
TINTA, TINTA_SUAVE = "#1f1f1e", "#6b6b66"

NUMERICAS = [
    "conf_yolo", "conf_yolo_max", "blob_area_ratio", "ratio_cromatico", "longitud_horizonte",
    "angulo_horizonte", "n_lineas_horizonte", "score_convergencia", "score_nulo",
    "apertura_haz", "n_lineas_fuga", "score_sam", "relleno_bbox", "segmentacion_fiable",
    "var_figura", "var_fondo", "var_max", "gate_color", "gate_horizonte", "gate_convergencia",
    "gate_espacio", "gate_nitidez", "gate_composicion", "margen_patron", "d_equilibrio",
]


# =============================================================================
# Datos
# =============================================================================

def cargar() -> pd.DataFrame:
    medidas = pd.DataFrame(leer_csv(MEDIDAS_CSV))
    for col in NUMERICAS:
        medidas[col] = pd.to_numeric(medidas[col], errors="coerce")

    aplicabilidad = pd.DataFrame(leer_csv(ANOTACION_APLICABILIDAD))
    aplicabilidad = aplicabilidad.rename(columns={"comentario": "comentario_aplicabilidad"})
    for col in aplicabilidad.columns:
        aplicabilidad[col] = aplicabilidad[col].str.strip()

    segmentacion = pd.DataFrame(leer_csv(SEGMENTACION_CSV))
    segmentacion = segmentacion[["imagen", "segmentacion", "caja_protagonista", "comentario"]]
    segmentacion = segmentacion.rename(columns={"comentario": "comentario_segmentacion"})

    datos = (medidas.merge(aplicabilidad, on="imagen", how="left", validate="1:1")
             .merge(segmentacion, on="imagen", how="left", validate="1:1"))
    assert len(datos) == 100 and datos["color"].notna().all(), "faltan anotaciones"
    yolo = datos["fuente"] == "yolo"
    assert datos.loc[yolo, "segmentacion"].notna().all(), "falta revisar alguna foto yolo"

    # Horizonte con la definición amplia (§6.2): `si` si lo es el natural o el estructural.
    natural, estructural = datos["horizonte_natural"], datos["horizontal_estructural"]
    datos["horizonte_amplio"] = np.select(
        [(natural == "si") | (estructural == "si"),
         (natural == "dudoso") | (estructural == "dudoso")],
        ["si", "dudoso"], default="no")
    datos["angulo_abs"] = datos["angulo_horizonte"].abs()
    return datos


# =============================================================================
# Regla del §4.1
# =============================================================================

def redondear_corte(estricto: float, otro: float | None, sentido: str) -> float:
    """Valor redondeado dentro de la banda, lo más cerca posible del extremo estricto.

    sentido = "min": la condición abre si valor >= corte; la banda es (otro, estricto] y se
                     redondea hacia abajo.
    sentido = "max": abre si valor <= corte; la banda es [estricto, otro) y se redondea
                     hacia arriba.
    Se prueba con 2 cifras significativas y se añade una cada vez que el valor redondeado
    se sale de la banda.
    """
    if estricto == 0:
        return 0.0
    exponente = math.floor(math.log10(abs(estricto)))
    for cifras in range(2, 9):
        paso = 10.0 ** (exponente - cifras + 1)
        if sentido == "min":
            valor = math.floor(estricto / paso + 1e-9) * paso
            dentro = otro is None or valor > otro
        else:
            valor = math.ceil(estricto / paso - 1e-9) * paso
            dentro = otro is None or valor < otro
        valor = float(f"{valor:.{cifras}g}")
        if dentro:
            return valor
    return estricto


def matriz(abre: pd.Series, positiva: pd.Series, nombres: pd.Series) -> dict:
    vp, fp = abre & positiva, abre & ~positiva
    fn, vn = ~abre & positiva, ~abre & ~positiva
    n_pos, n_neg = int(positiva.sum()), int((~positiva).sum())
    return {
        "vp": int(vp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "vn": int(vn.sum()),
        "tasa_fp": fp.sum() / n_neg if n_neg else float("nan"),
        "sensibilidad": vp.sum() / n_pos if n_pos else float("nan"),
        "precision": vp.sum() / abre.sum() if abre.sum() else float("nan"),
        "fotos_fp": " ".join(nombres[fp]), "fotos_fn": " ".join(nombres[fn]),
    }


def numero_de_regla(n_pos: int, n_neg: int, matriz_propuesta: dict) -> str:
    if n_pos < MIN_CASOS or n_neg < MIN_CASOS:
        return "3"
    if matriz_propuesta["fp"] == 0 and matriz_propuesta["fn"] == 0:
        return "1"
    return "2"


def calibrar_minimo(poblacion: pd.DataFrame, variable: str, positiva: pd.Series,
                    actual: float) -> dict:
    """Umbral de una variable que abre si valor >= corte (reglas 1 a 3)."""
    valores, nombres = poblacion[variable], poblacion["imagen"]
    n_pos, n_neg = int(positiva.sum()), int((~positiva).sum())

    if n_pos < MIN_CASOS or n_neg < MIN_CASOS:
        negativas_abiertas = valores[(valores >= actual) & ~positiva]
        if negativas_abiertas.empty:
            propuesto = actual
        else:
            # Se endurece lo justo para cerrarlas: el valor redondeado más bajo por encima.
            peor = negativas_abiertas.max()
            exponente = math.floor(math.log10(abs(peor)))
            propuesto = (math.floor(peor / 10.0 ** (exponente - 1)) + 1) * 10.0 ** (exponente - 1)
    else:
        # Los cortes que importan son los valores de las positivas (y «no abrir nada»).
        candidatos = sorted(set(valores[positiva]), reverse=True) + [math.inf]
        mejor = None
        for corte in candidatos:
            abre = valores >= corte
            vp, fp = int((abre & positiva).sum()), int((abre & ~positiva).sum())
            if fp > MAX_TASA_FP * n_neg:
                continue
            clave = (vp, corte)  # mayor sensibilidad; si empata, el más estricto
            if mejor is None or clave > mejor:
                mejor = clave
        estricto = mejor[1]
        por_debajo = valores[valores < estricto]
        propuesto = redondear_corte(estricto, por_debajo.max() if len(por_debajo) else None,
                                    "min")

    final = matriz(valores >= propuesto, positiva, nombres)
    return {"propuesto": propuesto, "regla": numero_de_regla(n_pos, n_neg, final),
            "n_pos": n_pos, "n_neg": n_neg, "final": final,
            "con_actual": matriz(valores >= actual, positiva, nombres)}


def calibrar_horizonte(poblacion: pd.DataFrame, positiva: pd.Series,
                       angulos_fijos: list[float] | None = None,
                       angulo_minimo: float = 0.0) -> dict:
    """Pareja (LONGITUD_MIN_HORIZONTE, ANGULO_MAX_HORIZONTE) con la regla 4.

    Abre si hay candidato, longitud >= L y |ángulo| <= A. `angulos_fijos` deja el ángulo fijo
    y calibra solo la longitud; `angulo_minimo` descarta los ángulos por debajo de ese valor.
    Las dos opciones son ALTERNATIVAS a la regla, no parte de ella.
    """
    nombres = poblacion["imagen"]
    candidato = poblacion["n_lineas_horizonte"] > 0
    longitud, angulo = poblacion["longitud_horizonte"], poblacion["angulo_abs"]
    n_pos, n_neg = int(positiva.sum()), int((~positiva).sum())

    def abre(l_min, a_max):
        return candidato & (longitud >= l_min) & (angulo <= a_max)

    longitudes = sorted(set(longitud[positiva & candidato])) + [math.inf]
    angulos = angulos_fijos or sorted(a for a in set(angulo[positiva & candidato])
                                      if a >= angulo_minimo)
    mejor = None
    for l_min in longitudes:
        for a_max in angulos:
            zona = abre(l_min, a_max)
            vp, fp = int((zona & positiva).sum()), int((zona & ~positiva).sum())
            if fp > MAX_TASA_FP * n_neg:
                continue
            # Mayor sensibilidad; si empata, el más estricto: menos FP, más longitud, menos ángulo.
            clave = (vp, -fp, l_min, -a_max)
            if mejor is None or clave > mejor[0]:
                mejor = (clave, l_min, a_max)
    _, l_exacta, a_exacto = mejor

    # Redondeo de cada extremo sin cambiar las fotos que abren.
    debajo = longitud[candidato & (angulo <= a_exacto) & (longitud < l_exacta)]
    l_prop = redondear_corte(l_exacta, debajo.max() if len(debajo) else None, "min")
    if angulos_fijos:
        a_prop = a_exacto
    else:
        encima = angulo[candidato & (longitud >= l_exacta) & (angulo > a_exacto)]
        a_prop = redondear_corte(a_exacto, encima.min() if len(encima) else None, "max")
    assert (abre(l_prop, a_prop) == abre(l_exacta, a_exacto)).all(), "el redondeo cambia la zona"

    final = matriz(abre(l_prop, a_prop), positiva, nombres)
    con_actual = matriz(abre(ACTUAL["LONGITUD_MIN_HORIZONTE"], ACTUAL["ANGULO_MAX_HORIZONTE"]),
                        positiva, nombres)
    regla = numero_de_regla(n_pos, n_neg, final)
    return {"propuesto": (l_prop, a_prop), "regla": f"4→{regla}", "n_pos": n_pos,
            "n_neg": n_neg, "final": final, "con_actual": con_actual}


# =============================================================================
# Análisis por umbral
# =============================================================================

def sin_dudosas(datos: pd.DataFrame, columna: str) -> tuple[pd.DataFrame, int]:
    dudosas = int((datos[columna] == "dudoso").sum())
    return datos[datos[columna] != "dudoso"].reset_index(drop=True), dudosas


def fila(umbral, actual, propuesto, regla, poblacion, resultado, dudosas, nota="",
         tipo="regla §4.1") -> dict:
    final, con_actual = resultado["final"], resultado["con_actual"]
    debil = (regla.endswith("2") and final["sensibilidad"] < MIN_SENSIBILIDAD)
    if debil:
        nota = (f"PUNTO DÉBIL: sensibilidad {final['sensibilidad']:.0%} < 60 %. " + nota).strip()
    return {
        "umbral": umbral, "tipo": tipo, "valor_actual": actual, "valor_propuesto": propuesto,
        "regla": regla, "poblacion": poblacion, "positivas": resultado["n_pos"],
        "negativas": resultado["n_neg"], "dudosas_excluidas": dudosas,
        "vp": final["vp"], "fp": final["fp"], "fn": final["fn"], "vn": final["vn"],
        "tasa_fp": round(final["tasa_fp"], 3), "sensibilidad": round(final["sensibilidad"], 3),
        "precision_apertura": round(final["precision"], 3),
        "fotos_fp": final["fotos_fp"], "fotos_fn": final["fotos_fn"],
        "fp_con_actual": con_actual["fp"], "fn_con_actual": con_actual["fn"],
        "tasa_fp_con_actual": round(con_actual["tasa_fp"], 3),
        "sensibilidad_con_actual": round(con_actual["sensibilidad"], 3),
        "fotos_fp_con_actual": con_actual["fotos_fp"], "nota": nota,
    }


def analizar_cromatico(datos):
    poblacion, dudosas = sin_dudosas(datos, "color")
    positiva = poblacion["color"] == "si"
    r = calibrar_minimo(poblacion, "ratio_cromatico", positiva, ACTUAL["RATIO_CROMATICO_MIN"])
    grafica_ordenada(poblacion, "ratio_cromatico", positiva, ACTUAL["RATIO_CROMATICO_MIN"],
                     r["propuesto"], "RATIO_CROMATICO_MIN", "ratio de píxeles cromáticos",
                     "huecos_ratio_cromatico.png", "color = si", "color = no")
    return [fila("RATIO_CROMATICO_MIN", ACTUAL["RATIO_CROMATICO_MIN"], r["propuesto"],
                 r["regla"], "las 100", r, dudosas)]


def analizar_horizonte(datos):
    poblacion, dudosas = sin_dudosas(datos, "horizonte_amplio")
    positiva = poblacion["horizonte_amplio"] == "si"
    filas = []
    variantes = [
        ("regla §4.1", {}, ""),
        ("alternativa A: ángulo fijo en 10°", {"angulos_fijos": [ACTUAL["ANGULO_MAX_HORIZONTE"]]},
         "No es la regla: se calibra solo la longitud y el ángulo se queda en su valor de diseño."),
        ("alternativa B: ángulo ≥ 10°", {"angulo_minimo": ACTUAL["ANGULO_MAX_HORIZONTE"]},
         "No es la regla: la pareja solo puede tener ángulos de 10° o más."),
    ]
    resultados = {}
    for tipo, opciones, nota in variantes:
        r = calibrar_horizonte(poblacion, positiva, **opciones)
        resultados[tipo] = r
        l_prop, a_prop = r["propuesto"]
        if a_prop == 0:
            nota = ("DEGENERADO: con ángulo máximo 0° solo abren las líneas exactamente "
                    "horizontales, así que angulo_horizonte valdría siempre 0 al abrir y la "
                    "métrica no podría señalar un horizonte torcido. " + nota).strip()
        for umbral, actual, propuesto in (
                ("LONGITUD_MIN_HORIZONTE", ACTUAL["LONGITUD_MIN_HORIZONTE"], l_prop),
                ("ANGULO_MAX_HORIZONTE", ACTUAL["ANGULO_MAX_HORIZONTE"], a_prop)):
            filas.append(fila(umbral, actual, propuesto, r["regla"],
                              "las 100 (definición amplia)", r, dudosas, nota, tipo))
    grafica_horizonte(poblacion, positiva, resultados)
    return filas


def analizar_convergencia(datos):
    supera_nulo = ((datos["n_lineas_fuga"] >= 2)
                   & (datos["score_convergencia"] > datos["score_nulo"]))
    poblacion, dudosas = sin_dudosas(datos[supera_nulo], "perspectiva")
    positiva = poblacion["perspectiva"] == "si"
    r = calibrar_minimo(poblacion, "apertura_haz", positiva, ACTUAL["APERTURA_MIN_GRADOS"])
    sin_nulo = datos[~supera_nulo & (datos["perspectiva"] == "si")]["imagen"]
    nota = (f"Perspectivas que no superan el modelo nulo (FN que este umbral no puede "
            f"arreglar): {' '.join(sin_nulo) or 'ninguna'}.")
    grafica_ordenada(poblacion, "apertura_haz", positiva, ACTUAL["APERTURA_MIN_GRADOS"],
                     r["propuesto"], "APERTURA_MIN_GRADOS", "apertura del haz (grados)",
                     "huecos_apertura_haz.png", "perspectiva = si", "perspectiva = no")
    return [fila("APERTURA_MIN_GRADOS", ACTUAL["APERTURA_MIN_GRADOS"], r["propuesto"],
                 r["regla"], "las que superan el modelo nulo", r, dudosas, nota)]


def analizar_sam(datos):
    poblacion = datos[datos["fuente"] == "yolo"].reset_index(drop=True)
    positiva = poblacion["segmentacion"].isin(["bien", "parcial"])
    score, relleno = poblacion["score_sam"], poblacion["relleno_bbox"]
    abre = ((score >= ACTUAL["SAM_SCORE_MIN"]) & (relleno >= ACTUAL["RELLENO_MIN"])
            & (relleno <= ACTUAL["RELLENO_MAX"]))
    m = matriz(abre, positiva, poblacion["imagen"])
    r = {"n_pos": int(positiva.sum()), "n_neg": int((~positiva).sum()),
         "final": m, "con_actual": m}
    assert r["n_neg"] < MIN_CASOS and m["fp"] == 0, "hay segmentaciones `mal`: revisar la regla 3"
    nota = (f"Comprobado, no calibrado: {r['n_neg']} segmentaciones `mal` (hacen falta 3). "
            f"Observado: score [{score.min():.3f}, {score.max():.3f}], "
            f"relleno [{relleno.min():.3f}, {relleno.max():.3f}]; ninguna foto queda fuera, "
            f"así que la condición de segmentación fiable nunca cierra en calibración.")
    grafica_sam(poblacion)
    return [fila(u, ACTUAL[u], ACTUAL[u], "4→3", "las yolo", r, 0, nota)
            for u in ("SAM_SCORE_MIN", "RELLENO_MIN", "RELLENO_MAX")]


def analizar_nitidez(datos):
    fiable = (datos["fuente"] == "yolo") & (datos["segmentacion_fiable"] == 1)
    poblacion, dudosas = sin_dudosas(datos[fiable], "nitidez")
    positiva = poblacion["nitidez"] == "si"
    r = calibrar_minimo(poblacion, "var_max", positiva, ACTUAL["VAR_LAPLACIANO_MIN"])
    caja_no = poblacion[~positiva & (poblacion["caja_protagonista"] == "no")]["imagen"]
    nota = (f"{len(caja_no)} de las {r['n_neg']} negativas lo son porque la caja de YOLO no es "
            f"el protagonista ({' '.join(caja_no)}), no por falta de textura.")
    filas = [fila("VAR_LAPLACIANO_MIN", ACTUAL["VAR_LAPLACIANO_MIN"], r["propuesto"],
                  r["regla"], "las yolo con segmentación fiable", r, dudosas, nota)]

    # Alternativa: solo las fotos en las que la caja es el protagonista, para aislar la
    # pregunta que responde este umbral (¿hay textura que comparar?).
    solo_caja = poblacion[poblacion["caja_protagonista"] == "si"].reset_index(drop=True)
    positiva_caja = solo_caja["nitidez"] == "si"
    r_caja = calibrar_minimo(solo_caja, "var_max", positiva_caja, ACTUAL["VAR_LAPLACIANO_MIN"])
    filas.append(fila("VAR_LAPLACIANO_MIN", ACTUAL["VAR_LAPLACIANO_MIN"], r_caja["propuesto"],
                      r_caja["regla"], "las yolo fiables con caja = protagonista", r_caja, 0,
                      "No es la regla: quita las negativas que este umbral no puede cerrar.",
                      "alternativa: solo caja = protagonista"))
    grafica_ordenada(poblacion, "var_max", positiva, ACTUAL["VAR_LAPLACIANO_MIN"],
                     r["propuesto"], "VAR_LAPLACIANO_MIN",
                     "max(var. Laplaciano figura, fondo) — escala log", "huecos_var_laplaciano.png",
                     "nitidez = si", "nitidez = no", log=True,
                     marcar=set(caja_no), texto_marca="caja ≠ protagonista",
                     extra=(r_caja["propuesto"], "alternativa (caja = protagonista)"))
    return filas


# =============================================================================
# Lo que no se calibra por huecos
# =============================================================================

def barrido_yolo(datos) -> list[dict]:
    """Cuántas fotos tendrían fuente = yolo con cada umbral y cuántas cajas son protagonista.

    Solo se documenta (§4). La caja elegida es la de mayor confianza, así que el umbral no
    cambia QUÉ caja se elige, solo si se acepta. Por debajo de 0.75 entrarían fotos cuya caja
    no se ha revisado; YOLO no devuelve cajas con confianza < 0.25.
    """
    conf = datos["conf_yolo_max"]
    caja = datos["caja_protagonista"]
    filas = []
    for umbral in np.round(np.arange(0.25, 0.951, 0.05), 2):
        acepta = conf >= umbral
        filas.append({
            "conf_yolo_threshold": umbral, "fotos_yolo": int(acepta.sum()),
            "caja_protagonista_si": int((acepta & (caja == "si")).sum()),
            "caja_protagonista_no": int((acepta & (caja == "no")).sum()),
            "sin_revisar": int((acepta & caja.isna()).sum()),
        })
    with BARRIDO_YOLO_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    grafica_yolo(datos)
    return filas


def escalas_composicion(datos) -> list[dict]:
    """Percentiles de margen_patron (fotos yolo: solo ahí se cita) y d_equilibrio (las 100)."""
    poblaciones = {
        "margen_patron": datos.loc[datos["fuente"] == "yolo", "margen_patron"],
        "d_equilibrio": datos["d_equilibrio"],
    }
    filas = []
    for variable, valores in poblaciones.items():
        bajo, alto = ESCALAS_ACTUALES[variable]
        registro = {"variable": variable, "n": len(valores),
                    "minimo": round(valores.min(), 4), "maximo": round(valores.max(), 4)}
        for p in (5, 10, 25, 50, 75, 90, 95):
            registro[f"p{p}"] = round(float(np.percentile(valores, p)), 4)
        registro["ref_baja_actual"] = bajo
        registro["percentil_de_ref_baja"] = round(100 * float((valores <= bajo).mean()), 1)
        registro["ref_alta_actual"] = alto
        registro["percentil_de_ref_alta"] = round(100 * float((valores <= alto).mean()), 1)
        filas.append(registro)
    with ESCALAS_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    return filas


def comprobar_blob(datos) -> str:
    saliencia = datos[datos["fuente"] != "yolo"]
    ratio = saliencia["blob_area_ratio"]
    fuera = saliencia[(ratio < ACTUAL["BLOB_AREA_MIN_RATIO"])
                      | (ratio > ACTUAL["BLOB_AREA_MAX_RATIO"])]
    fuentes = datos["fuente"].value_counts().to_dict()
    return (f"BLOB_*: {len(saliencia)} fotos sin YOLO, blob_area_ratio en "
            f"[{ratio.min():.3f}, {ratio.max():.3f}]; fuera de "
            f"[{ACTUAL['BLOB_AREA_MIN_RATIO']}, {ACTUAL['BLOB_AREA_MAX_RATIO']}]: {len(fuera)} "
            f"{' '.join(fuera['imagen'])}. Fuentes: {fuentes}.")


def composicion_con_caja_equivocada(datos) -> str:
    caja_no = datos[datos["caja_protagonista"] == "no"]
    abren = caja_no[caja_no["gate_composicion"] == 1]
    return (f"Composición: {len(abren)} de las {len(caja_no)} fotos con caja ≠ protagonista "
            f"abren la condición (falsos positivos): {' '.join(abren['imagen'])}.")


# =============================================================================
# Gráficas
# =============================================================================

def _estilo(ax):
    ax.grid(True, color="#e6e6e2", linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c9c4")
    ax.tick_params(colors=TINTA_SUAVE, labelsize=8)


def _puntos(ax, x, y, positiva, etiqueta_pos, etiqueta_neg):
    ax.scatter(x[positiva], y[positiva], s=34, marker="o", color=AZUL, edgecolor="white",
               linewidth=0.8, label=etiqueta_pos, zorder=3)
    ax.scatter(x[~positiva], y[~positiva], s=46, marker="X", color=NARANJA, edgecolor="white",
               linewidth=0.6, label=etiqueta_neg, zorder=4)


def grafica_ordenada(poblacion, variable, positiva, actual, propuesto, umbral, eje_y, nombre,
                     etiqueta_pos, etiqueta_neg, log=False, marcar=frozenset(),
                     texto_marca="", extra=None):
    """Fotos ordenadas por su variable, positivas y negativas, corte actual y propuesto."""
    orden = poblacion[variable].sort_values(kind="stable").index
    tabla = poblacion.loc[orden].reset_index(drop=True)
    pos = positiva.loc[orden].reset_index(drop=True)
    x = pd.Series(np.arange(1, len(tabla) + 1))
    y = tabla[variable]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    _estilo(ax)
    if log:
        ax.set_yscale("log")
    _puntos(ax, x, y, pos, etiqueta_pos, etiqueta_neg)
    if marcar:
        m = tabla["imagen"].isin(marcar)
        ax.scatter(x[m], y[m], s=150, facecolor="none", edgecolor=TINTA_SUAVE, linewidth=0.9,
                   label=texto_marca, zorder=2)
    ax.axhline(actual, color=GRIS, linestyle="--", linewidth=1.2,
               label=f"actual = {actual:g}", zorder=1)
    ax.axhline(propuesto, color=TINTA, linewidth=1.6,
               label=f"propuesto (regla §4.1) = {propuesto:g}", zorder=1)
    if extra is not None:
        ax.axhline(extra[0], color=TINTA, linestyle=":", linewidth=1.4,
                   label=f"{extra[1]} = {extra[0]:g}", zorder=1)

    # Nombre de las negativas que abren con el corte actual o el propuesto.
    # Se alternan arriba y abajo para que no se pisen las de fotos vecinas.
    abre = (y >= min(actual, propuesto)) & ~pos
    for k, i in enumerate(np.flatnonzero(abre.to_numpy())):
        ax.annotate(tabla["imagen"][i].replace("imagen", "img"), (x[i], y[i]),
                    xytext=(0, -12 if k % 2 == 0 else 7), textcoords="offset points",
                    ha="center", fontsize=6.5, color=TINTA_SUAVE)
    ax.set_xlabel("fotos ordenadas por la variable", fontsize=9, color=TINTA_SUAVE)
    ax.set_ylabel(eje_y, fontsize=9, color=TINTA_SUAVE)
    ax.set_title(f"{umbral} · {int(pos.sum())} positivas, {int((~pos).sum())} negativas",
                 fontsize=11, color=TINTA, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right" if log else "upper left")
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / nombre)
    plt.close(fig)


def grafica_horizonte(poblacion, positiva, resultados):
    con = poblacion["n_lineas_horizonte"] > 0
    x, y = poblacion["longitud_horizonte"][con], poblacion["angulo_abs"][con]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    _estilo(ax)
    _puntos(ax, x, y, positiva[con], "horizonte amplio = si", "horizonte amplio = no")
    x_max = poblacion["longitud_horizonte"].max() * 1.03

    def zona(l_min, a_max, **estilo):
        ax.plot([l_min, l_min, x_max], [0, a_max, a_max], **estilo)

    zona(ACTUAL["LONGITUD_MIN_HORIZONTE"], ACTUAL["ANGULO_MAX_HORIZONTE"], color=GRIS,
         linestyle="--", linewidth=1.2, label="actual (0.13, 10°)")
    estilos = {"regla §4.1": dict(color=TINTA, linewidth=1.6),
               "alternativa A: ángulo fijo en 10°": dict(color=TINTA, linestyle=":",
                                                          linewidth=1.4),
               "alternativa B: ángulo ≥ 10°": dict(color=TINTA, linestyle="-.", linewidth=1.2)}
    for tipo, r in resultados.items():
        l_prop, a_prop = r["propuesto"]
        zona(l_prop, max(a_prop, 0.15), label=f"{tipo} ({l_prop:g}, {a_prop:g}°)",
             **estilos[tipo])
    sin = int((~con).sum())
    sin_pos = int((~con & positiva).sum())
    ax.set_xlabel("longitud del candidato (fracción de la diagonal)", fontsize=9,
                  color=TINTA_SUAVE)
    ax.set_ylabel("|ángulo| del candidato (grados)", fontsize=9, color=TINTA_SUAVE)
    ax.set_title(f"Horizonte (definición amplia) · abre a la derecha y por debajo de cada "
                 f"línea\n{sin} fotos sin candidato ({sin_pos} positiva) no se dibujan: no "
                 f"abren con ningún umbral", fontsize=10, color=TINTA, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "huecos_horizonte.png")
    plt.close(fig)


def grafica_sam(poblacion):
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    _estilo(ax)
    parcial = poblacion["segmentacion"] == "parcial"
    ax.scatter(poblacion["relleno_bbox"][~parcial], poblacion["score_sam"][~parcial], s=34,
               color=AZUL, edgecolor="white", linewidth=0.8, label="bien", zorder=3)
    ax.scatter(poblacion["relleno_bbox"][parcial], poblacion["score_sam"][parcial], s=60,
               marker="D", color=AZUL, edgecolor=TINTA, linewidth=0.8,
               label="parcial (cuenta como positiva)", zorder=4)
    for i in np.flatnonzero(parcial.to_numpy()):
        ax.annotate(poblacion["imagen"][i], (poblacion["relleno_bbox"][i],
                                             poblacion["score_sam"][i]),
                    xytext=(6, -3), textcoords="offset points", fontsize=7, color=TINTA_SUAVE)
    ax.plot([ACTUAL["RELLENO_MIN"], ACTUAL["RELLENO_MIN"], ACTUAL["RELLENO_MAX"],
             ACTUAL["RELLENO_MAX"]], [1.0, ACTUAL["SAM_SCORE_MIN"], ACTUAL["SAM_SCORE_MIN"], 1.0],
            color=TINTA, linewidth=1.4, label="zona fiable (0.90; 0.20–0.95), sin cambios")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.88, 1.0)
    ax.set_xlabel("relleno de la caja (máscara / caja)", fontsize=9, color=TINTA_SUAVE)
    ax.set_ylabel("score de SAM", fontsize=9, color=TINTA_SUAVE)
    ax.set_title("Segmentación (fotos yolo) · 0 segmentaciones «mal»: regla 3", fontsize=11,
                 color=TINTA, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "huecos_sam.png")
    plt.close(fig)


def grafica_yolo(datos):
    revisadas = datos[datos["caja_protagonista"].notna()]
    resto = datos[datos["caja_protagonista"].isna() & datos["conf_yolo_max"].notna()]
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    _estilo(ax)
    es = revisadas["caja_protagonista"] == "si"
    jitter = lambda n, semilla: np.random.default_rng(semilla).uniform(-0.25, 0.25, n)  # noqa
    ax.scatter(revisadas["conf_yolo_max"][es], 1 + jitter(es.sum(), 1), s=34, color=AZUL,
               edgecolor="white", linewidth=0.8, label="caja = protagonista", zorder=3)
    ax.scatter(revisadas["conf_yolo_max"][~es], 1 + jitter((~es).sum(), 2), s=46, marker="X",
               color=NARANJA, edgecolor="white", linewidth=0.6, label="caja ≠ protagonista",
               zorder=4)
    ax.scatter(resto["conf_yolo_max"], jitter(len(resto), 3), s=30, color=GRIS,
               edgecolor="white", linewidth=0.8, label="sin revisar (hoy van a saliencia)",
               zorder=3)
    ax.axvline(ACTUAL["CONF_YOLO_THRESHOLD"], color=TINTA, linewidth=1.4,
               label="CONF_YOLO_THRESHOLD = 0.75")
    ax.set_yticks([0, 1], ["< 0.75", "≥ 0.75"])
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("confianza de la mejor caja de YOLO", fontsize=9, color=TINTA_SUAVE)
    ax.set_title(f"Barrido de CONF_YOLO_THRESHOLD · {datos['conf_yolo_max'].isna().sum()} "
                 f"fotos sin ninguna caja", fontsize=11, color=TINTA, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(DIR_RESULTADOS / "huecos_conf_yolo.png")
    plt.close(fig)


# =============================================================================
# Principal
# =============================================================================

def main() -> None:
    datos = cargar()
    filas = (analizar_cromatico(datos) + analizar_horizonte(datos)
             + analizar_convergencia(datos) + analizar_sam(datos) + analizar_nitidez(datos))
    with PROPUESTA_CSV.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"Propuesta: {PROPUESTA_CSV}\n")
    for f_ in filas:
        print(f"{f_['umbral']:<23} {f_['tipo']:<40} {f_['valor_actual']:>7g} → "
              f"{f_['valor_propuesto']:<8g} regla {f_['regla']:<4} "
              f"VP {f_['vp']:>2} FP {f_['fp']:>2} FN {f_['fn']:>2} VN {f_['vn']:>2} · "
              f"tasa FP {f_['tasa_fp']:.0%} (actual {f_['tasa_fp_con_actual']:.0%}) · "
              f"sens. {f_['sensibilidad']:.0%} (actual {f_['sensibilidad_con_actual']:.0%})")
        print(f"    FP: {f_['fotos_fp'] or '—'}   |   FP con el actual: "
              f"{f_['fotos_fp_con_actual'] or '—'}")
        if f_["nota"]:
            print(f"    {f_['nota']}")

    print(f"\nBarrido de YOLO: {BARRIDO_YOLO_CSV}")
    for b in barrido_yolo(datos):
        print(f"  {b['conf_yolo_threshold']:.2f}: {b['fotos_yolo']:>3} yolo · caja si "
              f"{b['caja_protagonista_si']:>2} · no {b['caja_protagonista_no']:>2} · "
              f"sin revisar {b['sin_revisar']:>2}")
    print(f"\nEscalas de composición: {ESCALAS_CSV}")
    for e in escalas_composicion(datos):
        print(f"  {e}")
    print()
    print(comprobar_blob(datos))
    print(composicion_con_caja_equivocada(datos))


if __name__ == "__main__":
    main()
