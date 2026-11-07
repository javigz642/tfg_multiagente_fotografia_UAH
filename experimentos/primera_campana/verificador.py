"""Verificador de E3: fidelidad de las citas `[campo = valor]` y disciplina del crítico.

Unifica los auditores de los notebooks 01-05 con las correcciones de experimentos.md §6.3:

- extrae varias citas por corchete, con valores escalares, de texto o tupla `(a, b)`;
- resuelve `metrica.campo`, `campo` y, en el crítico, `dimension.metrica.campo`, además de las citas
  del arbitraje (`W`, `W(dim)`, `vector_pesos.dim`, `cobertura_por_dimension.dim`);
- clasifica cada cita: `exacta` (redondeo correcto a los decimales escritos), `truncada` (coincide
  con cortar los dígitos, no con redondear), `alterada`, `texto_distinto`, `null_citado`,
  `no_numerica` o `campo_inexistente`;
- marca la cita a una métrica cerrada solo si cita su `valor`/`valor_norm` (o la métrica a secas) o
  un campo de evidencia condicionado por ella (`CAMPOS_CONDICIONADOS`). Citar `.confianza` no lo es;
- controla el crítico: arbitraje transcrito frente al recomputado, `afirmaciones` en el orden del
  arbitraje, una salvedad por métrica cerrada y `campos_citados` frente a lo citado;
- compara el informe transcrito por el LLM con el de la herramienta (fidelidad de transcripción).

Las citas se comprueban contra el `informe` del propio `salida_*.json` (lo que el autor del texto tenía
delante); la transcripción del informe se comprueba aparte contra la herramienta.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

DIMENSIONES = ["composicion_espacial", "lineas_direccion", "espacio_aislamiento_sujeto", "luz_tono"]
FICHERO = {
    "composicion_espacial": "salida_composicion_espacial.json",
    "lineas_direccion": "salida_lineas_direccion.json",
    "espacio_aislamiento_sujeto": "salida_espacio_aislamiento.json",
    "luz_tono": "salida_luz_tono.json",
}

# Campos planos que describen lo que mide una métrica con gate: si la métrica está cerrada, tampoco
# se pueden citar (agents.yaml: «los campos planos de una dimensión cerrada [...] no se citan»).
# `longitud_horizonte`, `n_lineas_fuga` y `score_convergencia_nulo` no están: explican el cierre.
CAMPOS_CONDICIONADOS = {
    "esquema_cromatico": {"spread_cromatico", "matices_dominantes", "n_matices_dominantes"},
    "angulo_horizonte": set(),
    "score_convergencia": {"coord_punto_fuga", "apertura_haz", "n_lineas_inliers"},
    "ratio_espacio_negativo": set(),
    "ratio_nitidez": {"var_laplaciano_figura", "var_laplaciano_fondo"},
    "d_tercios": {"coord_centroide", "coord_p_cercano", "margen_patron"},
    "d_centro": {"coord_centroide", "coord_p_cercano", "margen_patron"},
    "patron_dominante": {"coord_centroide", "coord_p_cercano", "margen_patron"},
}
CAMPOS_DE_METRICA_PROHIBIDOS = {"", "valor", "valor_norm"}   # `.confianza` y `.fuente_confianza` sí valen

RE_CORCHETE = re.compile(r"\[([^\[\]]*(?:\[\d+\][^\[\]]*)*)\]")
# Una cita dentro de un corchete: ruta (admite «W luz_tono» y «luz_tono: W») = valor (tupla, texto
# entre comillas o un token sin espacios ni comas).
RE_CITA = re.compile(r"([A-Za-z_][\w\.\[\]\(\)]*(?:\s*:\s*W\b|\s+[a-z_]+)?)\s*=\s*"
                     r"(\([^)]*\)|'[^']*'|\"[^\"]*\"|«[^»]*»|[^,\s\]]+)")
RE_NUMERO = re.compile(r"-?\d+(?:\.\d+)?")
# Forma canónica de las citas al arbitraje (experimentos.md §6.3).
RE_ARBITRAJE = re.compile(r"^(?:W|w)(?:\((\w+)\))?$|^(vector_pesos|cobertura_por_dimension)\.(\w+)$")
RE_W = re.compile(r"(?:^|[\s\.\(:_])W(?:$|[\s\.\(\):_])|vector_pesos")


def es_metrica(campo) -> bool:
    return isinstance(campo, dict) and "valor" in campo and "confianza" in campo


# =============================================================================
# Extracción y resolución
# =============================================================================

def extraer_citas(texto: str) -> list[tuple[str, str]]:
    """Citas `(ruta, valor escrito)` de un texto; admite varias por corchete (`[a = 1, b = 2]`)."""
    return [(ruta.strip(), escrito.strip()) for grupo in RE_CORCHETE.findall(texto or "")
            for ruta, escrito in RE_CITA.findall(grupo)]


def _dimension_en(texto: str) -> str | None:
    """La dimensión nombrada en una ruta libre (admite prefijos: «luz», «espacio_aislamiento»)."""
    for token in re.findall(r"[a-z][a-z_]{2,}", texto):
        token = token.strip("_")
        dims = [d for d in DIMENSIONES if d.startswith(token)]
        if len(dims) == 1:
            return dims[0]
    return None


def _resolver_arbitraje(ruta: str, arbitraje: dict) -> dict | None:
    """Cita al arbitraje, canónica o libre («lineas_direccion.W», «W_luz», «luz_tono: W»...)."""
    canonica = RE_ARBITRAJE.match(ruta)
    if "etiqueta_contexto" in ruta:
        clave, dim = "etiqueta_contexto", None
    elif "cobertura" in ruta:
        clave, dim = "cobertura_por_dimension", _dimension_en(ruta.replace("cobertura_por_dimension", ""))
    elif RE_W.search(ruta) or ruta in ("W", "w"):
        clave, dim = "vector_pesos", _dimension_en(ruta.replace("vector_pesos", ""))
    else:
        return None
    if clave == "etiqueta_contexto":
        valor = arbitraje[clave]
    elif dim is not None:
        valor = arbitraje[clave][dim]
    elif clave == "vector_pesos":   # «W» a secas: vale si coincide con algún peso
        valor = tuple(arbitraje[clave].values())
    else:
        return None
    return {"valor": valor, "dimension": "arbitraje", "metrica": None,
            "campo": f"{clave}.{dim}" if dim else clave, "ruta": f"arbitraje.{clave}" + (f".{dim}" if dim else ""),
            "forma_canonica": bool(canonica)}


def _bajar(obj, partes):
    for parte in partes:
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", parte)
        if not m or not isinstance(obj, dict) or m.group(1) not in obj:
            raise KeyError(parte)
        obj = obj[m.group(1)]
        if m.group(2) is not None:
            if not isinstance(obj, list) or int(m.group(2)) >= len(obj):
                raise KeyError(parte)
            obj = obj[int(m.group(2))]
    return obj


def resolver_ruta(ruta: str, informes: dict, arbitraje: dict | None = None) -> dict:
    """Resuelve una ruta citada. Devuelve {valor, dimension, metrica, campo, ruta} o lanza KeyError.

    `informes` es {dimension: informe}. Con una sola dimensión (especialista), la ruta no lleva la
    dimensión delante. `metrica` es la métrica con gate a la que pertenece el campo (o None) y
    `campo` lo que se cita dentro de ella ('' si se cita la métrica a secas).
    """
    if arbitraje is not None and (r := _resolver_arbitraje(ruta, arbitraje)) is not None:
        return r

    partes = ruta.split(".")
    canonica = True
    if len(partes) > 1 and partes[-1] in ("x", "y"):   # «coord_centro_masa.x» -> componente 0
        partes = partes[:-2] + [f"{partes[-2]}[{'xy'.index(partes[-1])}]"]
        canonica = False
    candidatas = [partes[0]] if partes[0] in informes else list(informes)
    resto = partes[1:] if partes[0] in informes else partes
    if not resto or not all(resto):
        raise KeyError(ruta)
    for dim in candidatas:
        try:
            valor = _bajar(informes[dim], resto)
        except KeyError:
            continue
        cabeza = re.sub(r"\[\d+\]$", "", resto[0])
        if es_metrica(informes[dim].get(cabeza)):
            metrica, campo = cabeza, ".".join(resto[1:])
            if campo == "":
                valor = informes[dim][cabeza]["valor"]   # [metrica = x] se lee como su valor
        else:
            metrica = next((m for m, campos in CAMPOS_CONDICIONADOS.items()
                            if cabeza in campos and m in informes[dim]), None)
            campo = cabeza
        return {"valor": valor, "dimension": dim, "metrica": metrica, "campo": campo,
                "ruta": f"{dim}.{'.'.join(resto)}", "forma_canonica": canonica}
    raise KeyError(ruta)


# =============================================================================
# Clasificación de una cita
# =============================================================================

def _decimales(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def _comparar_numero(real: float, escrito: str) -> str:
    try:
        e = float(escrito)
    except ValueError:
        return "no_numerica"
    d = _decimales(escrito)
    paso = 10.0 ** -d
    if abs(real - e) <= paso / 2 + 1e-9 * max(1.0, abs(real)):
        return "exacta"
    truncado = math.trunc(real / paso) * paso
    if abs(truncado - e) <= 1e-9 * max(1.0, abs(real)):
        return "truncada"
    return "alterada"


def clasificar_cita(valor, escrito: str) -> str:
    """exacta / truncada / alterada / texto_distinto / null_citado / no_numerica."""
    limpio = escrito.strip().strip("'\"«»").strip()
    if valor is None:
        return "null_citado"
    if isinstance(valor, bool):
        return "exacta" if limpio.lower() in (str(valor).lower(), "si" if valor else "no") else "texto_distinto"
    if isinstance(valor, str):
        return "exacta" if limpio.lower() == valor.lower() else "texto_distinto"
    if isinstance(valor, tuple):   # «W» a secas: cualquier peso del vector, o el vector entero
        if len(RE_NUMERO.findall(limpio)) > 1:
            return clasificar_cita(list(valor), limpio)
        clases = [_comparar_numero(v, limpio) for v in valor]
        return "exacta" if "exacta" in clases else "truncada" if "truncada" in clases else clases[0]
    if isinstance(valor, list):
        nums = RE_NUMERO.findall(limpio)
        if not all(isinstance(v, (int, float)) for v in valor):
            return "no_numerica"
        if len(nums) == 1 and len(valor) > 1:   # un componente suelto de una tupla
            return clasificar_cita(tuple(valor), limpio)
        if len(nums) != len(valor):
            return "no_numerica"
        clases = [_comparar_numero(v, n) for v, n in zip(valor, nums)]
        return next((c for c in ("alterada", "no_numerica", "truncada") if c in clases), "exacta")
    if isinstance(valor, (int, float)):
        return _comparar_numero(float(valor), limpio.rstrip("%°º").strip())
    return "no_numerica"


def auditar_texto(texto: str, informes: dict, arbitraje: dict | None, cerradas: set) -> list[dict]:
    """Una fila por cita del texto, con su clase y si cita una métrica cerrada.
    `cerradas` son pares (dimension, metrica)."""
    filas = []
    for ruta, escrito in extraer_citas(texto):
        try:
            r = resolver_ruta(ruta, informes, arbitraje)
        except KeyError:
            filas.append({"ruta": ruta, "escrito": escrito, "real": None, "clase": "campo_inexistente",
                          "a_cerrada": False, "ruta_completa": None, "forma_canonica": None})
            continue
        clase = clasificar_cita(r["valor"], escrito)
        a_cerrada = (r["metrica"] is not None and (r["dimension"], r["metrica"]) in cerradas
                     and (r["campo"] in CAMPOS_DE_METRICA_PROHIBIDOS
                          or r["campo"] in CAMPOS_CONDICIONADOS.get(r["metrica"], set())))
        real = r["valor"] if not isinstance(r["valor"], tuple) else list(r["valor"])
        filas.append({"ruta": ruta, "escrito": escrito, "real": json.dumps(real, ensure_ascii=False),
                      "clase": clase, "a_cerrada": a_cerrada, "ruta_completa": r["ruta"],
                      "forma_canonica": r["forma_canonica"]})
    return filas


def cifras_sin_cita(texto: str) -> list[str]:
    """Cifras escritas en la prosa fuera de los corchetes (heurística: decimales o 2+ dígitos)."""
    fuera = RE_CORCHETE.sub(" ", texto or "")
    return re.findall(r"(?<![\w.])-?\d+\.\d+|(?<![\w.])\d{2,}(?![\w.])", fuera)


def metricas_cerradas(informes: dict) -> set:
    return {(d, m) for d, inf in informes.items() for m, v in inf.items() if es_metrica(v) and v["confianza"] == 0}


# =============================================================================
# Auditorías de una ejecución
# =============================================================================

def cargar_salidas(dir_ejec: Path) -> dict:
    return {d: json.loads((Path(dir_ejec) / f).read_text(encoding="utf-8")) for d, f in FICHERO.items()}


def auditar_especialista(salida: dict, dimension: str) -> list[dict]:
    informes = {dimension: salida["informe"]}
    filas = auditar_texto(salida["diagnostico"], informes, None, metricas_cerradas(informes))
    for f in filas:
        f.update(autor=dimension, bloque="diagnostico")
    return filas


def auditar_critica(critica: dict, salidas: dict, arbitraje_recomputado: dict | None = None) -> tuple[list[dict], dict]:
    """(citas, controles) de la crítica. Los informes son los transcritos de la misma ejecución."""
    informes = {d: s["informe"] for d, s in salidas.items()}
    arb = critica["arbitraje"]
    cerradas = metricas_cerradas(informes)
    citas = []
    bloques = ([("afirmacion", i, a["texto"], a) for i, a in enumerate(critica["afirmaciones"])]
               + [("tension", i, t["texto"], t) for i, t in enumerate(critica["tensiones"])]
               + [("tension_resolucion", i, t.get("resolucion_por_w", ""), None) for i, t in enumerate(critica["tensiones"])]
               + [("salvedad", i, s if isinstance(s, str) else json.dumps(s, ensure_ascii=False), None)
                  for i, s in enumerate(critica["salvedades"])]
               + [("sintesis", 0, critica["sintesis"], None)])
    campos_mal = []
    for tipo, i, texto, obj in bloques:
        filas = auditar_texto(texto, informes, arb, cerradas)
        for f in filas:
            f.update(autor="critico", bloque=tipo)
        citas += filas
        if obj is not None and "campos_citados" in obj:
            # Solo campos de los informes (las citas a W o a la cobertura no cuentan) y sin índices:
            # declarar `matices_dominantes` y citar `matices_dominantes[0]` es coherente.
            def normal(r):   # sin índices y con `metrica.valor` igual a `metrica`
                return re.sub(r"\.valor$", "", re.sub(r"\[\d+\]", "", r))
            citados = {normal(f["ruta_completa"]) for f in filas
                       if f["ruta_completa"] and f["ruta_completa"].split(".")[0] in DIMENSIONES}
            declarados = set()
            for c in obj["campos_citados"]:
                try:
                    r = resolver_ruta(c, informes, None)["ruta"]
                except KeyError:
                    r = c
                if r.split(".")[0] in DIMENSIONES:
                    declarados.add(normal(r))
            if citados != declarados:
                campos_mal.append({"bloque": f"{tipo}[{i}]", "no_declarados": sorted(citados - declarados),
                                   "declarados_sin_citar": sorted(declarados - citados)})

    orden = arb["orden_dimensiones"]
    dims_afirm = [a["dimension"] for a in critica["afirmaciones"]]
    indices = [orden.index(d) for d in dims_afirm if d in orden]
    # Las cerradas de verdad salen de los informes; el crítico suele escribir una salvedad por gate
    # (las 3 métricas de composición comparten uno), así que se exige un número entre las
    # dimensiones con algo cerrado y las métricas cerradas.
    n_cerradas = len(cerradas)
    n_dims_cerradas = len({d for d, _ in cerradas})
    n_salv = len(critica["salvedades"])

    def iguales(k):   # las listas de métricas no tienen orden con significado; el orden de dimensiones sí
        a, b = arbitraje_recomputado.get(k), arb.get(k)
        return sorted(a) == sorted(b or []) if k in ("metricas_citables", "metricas_cerradas") else a == b

    diferencias = [] if arbitraje_recomputado is None else [k for k in arbitraje_recomputado if not iguales(k)]
    controles = {
        "arbitraje_igual_recomputado": None if arbitraje_recomputado is None else not diferencias,
        "arbitraje_diferencias": diferencias,
        "afirmaciones_en_orden": indices == sorted(indices) and len(indices) == len(dims_afirm),
        "dimensiones_afirmaciones": dims_afirm,
        "n_metricas_cerradas": n_cerradas,
        "n_dimensiones_con_cerradas": n_dims_cerradas,
        "n_salvedades": n_salv,
        "salvedades_suficientes": n_salv >= n_dims_cerradas and (n_salv <= n_cerradas or n_cerradas == 0),
        "campos_citados_coherentes": not campos_mal,
        "campos_citados_incidencias": campos_mal,
        "cifras_sin_cita_sintesis": cifras_sin_cita(critica["sintesis"]),
        "n_tensiones": len(critica["tensiones"]),
        "vio_imagen": critica.get("vio_imagen"),
    }
    return citas, controles


def fidelidad_transcripcion(transcrito: dict, herramienta: dict, tol_rel: float = 1e-9) -> list[dict]:
    """Campo a campo: informe copiado por el LLM frente al de la herramienta.

    `igual`: idéntico (números con tolerancia relativa `tol_rel`). `ruido`: difiere menos de 1e-6
    relativo (variación numérica de la herramienta, no del LLM). `distinto`, `falta`, `sobra`.
    Se omite `verificacion_path` (ruta local).
    """
    filas = []

    def comparar(a, b, ruta):
        if isinstance(b, dict) and isinstance(a, dict):
            for k in sorted(set(a) | set(b)):
                if k == "verificacion_path":
                    continue
                if k not in a:
                    filas.append({"campo": f"{ruta}{k}", "estado": "falta"})
                elif k not in b:
                    filas.append({"campo": f"{ruta}{k}", "estado": "sobra"})
                else:
                    comparar(a[k], b[k], f"{ruta}{k}.")
            return
        campo = ruta.rstrip(".")
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            estados = []
            for x, y in zip(a, b):
                sub = []
                filas_previas = len(filas)
                comparar(x, y, campo + "[].")
                sub = filas[filas_previas:]
                del filas[filas_previas:]
                estados += [f["estado"] for f in sub]
            estado = next((e for e in ("distinto", "falta", "sobra", "ruido") if e in estados), "igual")
            filas.append({"campo": campo, "estado": estado, "transcrito": a, "herramienta": b})
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            dif = abs(a - b) / max(1.0, abs(b))
            estado = "igual" if dif <= tol_rel else "ruido" if dif < 1e-6 else "distinto"
        else:
            estado = "igual" if a == b else "distinto"
        filas.append({"campo": campo, "estado": estado, "transcrito": a, "herramienta": b})

    comparar(transcrito, herramienta, "")
    return filas
