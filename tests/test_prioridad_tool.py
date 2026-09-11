"""Prueba manual de PrioridadTool."""
import json
import tempfile
from pathlib import Path

from tfg_multiagente_fotografia.schemas.orquestador import VectorPesosW
from tfg_multiagente_fotografia.tools.prioridad_tool import (
    INFORMES,
    PrioridadEngine,
    PrioridadTool,
    es_metrica,
)
from tfg_multiagente_fotografia.rutas import ultima_ejecucion
from tfg_multiagente_fotografia.tools.vector_pesos_tool import vector_pesos_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Nombres reales de las metricas de cada especialista. Se usan los de verdad y no `m1`/`m2`
# para que el orden alfabetico dentro de cada dimension —que es el que publica el engine—
# se lea igual aqui que en una salida real.
CORPUS = {
    "composicion_espacial": ["d_tercios", "d_centro", "d_equilibrio", "patron_dominante"],
    "lineas_direccion": ["angulo_horizonte", "score_convergencia"],
    "espacio_aislamiento_sujeto": ["ratio_espacio_negativo", "ratio_nitidez"],
    "luz_tono": ["media_L", "esquema_cromatico"],
}

STEM = "imagen30"


# =============================================================================
# Utilidades: fabricar los cuatro informes en un directorio temporal
# =============================================================================

def _metrica(confianza: float, valor: float = 0.5) -> dict:
    """Un campo con la forma exacta de `MetricaConfianza`."""
    return {
        "valor": valor,
        "valor_norm": 0.5,
        "confianza": confianza,
        "fuente_confianza": "caso fabricado",
    }


def _informe(stem: str, confianzas: dict, extras: dict | None = None, envolver: bool = True):
    """Un `salida_*.json` fabricado.

    `envolver` reproduce la forma que emiten los cuatro especialistas
    (`{informe, diagnostico}`); con False se escribe el informe pelado, que es la otra forma
    que el engine acepta.
    """
    informe = {
        "dimensiones_imagen": [1920, 1440],
        # El engine deduce de aqui a que imagen pertenece el informe, asi que el nombre no
        # es decorativo: es la unica defensa contra arbitrar con los ficheros de la
        # ejecucion anterior.
        "verificacion_path": str(Path("outputs") / "fabricado" / f"{stem}_x.png"),
    }
    informe.update({nombre: _metrica(c) for nombre, c in confianzas.items()})
    if extras:
        informe.update(extras)
    return {"informe": informe, "diagnostico": "texto fabricado"} if envolver else informe


def _escribir(raiz: Path, confianzas: dict, stem: str = STEM, extras: dict | None = None,
              envolver: bool = True, omitir: str | None = None) -> PrioridadEngine:
    """Escribe los cuatro informes en `raiz` y devuelve un engine que apunta ahi."""
    extras = extras or {}
    for dimension, fichero in INFORMES.items():
        if dimension == omitir:
            continue
        datos = _informe(stem, confianzas[dimension], extras.get(dimension), envolver)
        (raiz / fichero).write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    return PrioridadEngine(raiz=raiz)


def _confianzas(por_dimension: dict | None = None, defecto: float = 1.0) -> dict:
    """Confianzas para las cuatro dimensiones; `por_dimension` sobreescribe metricas sueltas."""
    por_dimension = por_dimension or {}
    return {
        dimension: {
            nombre: por_dimension.get(dimension, {}).get(nombre, defecto)
            for nombre in metricas
        }
        for dimension, metricas in CORPUS.items()
    }


# =============================================================================
# A) Casos analiticos con informes fabricados
# =============================================================================

def test_todas_citables():
    """Cobertura 1.0 en las cuatro, `metricas_cerradas` vacia.

    Es el extremo que ninguna ejecucion real ha dado todavia, y el que fija el orden en que
    se publican las rutas: las dimensiones en el orden CANONICO de `INFORMES` (no en el de
    W, que viaja aparte en `orden_dimensiones`) y las metricas alfabeticamente dentro de cada
    una. Ese orden es contrato: el prompt del critico recorre estas listas.
    """
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas(defecto=1.0))
        arb = engine.arbitrar(f"{STEM}.jpg", "paisaje")

        esperadas = [
            "composicion_espacial.d_centro",
            "composicion_espacial.d_equilibrio",
            "composicion_espacial.d_tercios",
            "composicion_espacial.patron_dominante",
            "lineas_direccion.angulo_horizonte",
            "lineas_direccion.score_convergencia",
            "espacio_aislamiento_sujeto.ratio_espacio_negativo",
            "espacio_aislamiento_sujeto.ratio_nitidez",
            "luz_tono.esquema_cromatico",
            "luz_tono.media_L",
        ]
        try:
            assert arb.metricas_citables == esperadas, f"citables={arb.metricas_citables}"
            assert arb.metricas_cerradas == [], f"cerradas={arb.metricas_cerradas}"
            assert all(v == 1.0 for v in arb.cobertura_por_dimension.values()), (
                f"cobertura={arb.cobertura_por_dimension}"
            )
            print(f"[OK] todas citables: 10/10 rutas, cobertura 1.0 en las cuatro dimensiones")
        except AssertionError as err:
            fallos.append(("todas citables", str(err)))
            print(f"[FAIL] todas citables: {err}")
    return fallos


def test_dimension_entera_cerrada():
    """Una dimension con cobertura 0.0 SIGUE siendo la primera si es la de mayor peso.

    Es el caso que justifica la decision de NO ordenar por `W x cobertura`: en paisaje
    `luz_tono` pesa 0.35, el maximo de la fila, y aunque no se haya podido medir nada de ella
    tiene que encabezar la critica. Multiplicar por la cobertura la mandaria al final y
    enterraria el hallazgo —la dimension que mas importa en este contexto es justo la que no
    se pudo medir—, que es lo unico que este sistema puede decir y un LLM generalista no.
    """
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        confianzas = _confianzas({"luz_tono": {"media_L": 0.0, "esquema_cromatico": 0.0}})
        engine = _escribir(Path(tmp), confianzas)
        arb = engine.arbitrar(f"{STEM}.jpg", "paisaje")

        try:
            assert arb.cobertura_por_dimension["luz_tono"] == 0.0, (
                f"cobertura de luz_tono={arb.cobertura_por_dimension['luz_tono']}"
            )
            assert arb.orden_dimensiones[0] == "luz_tono", (
                f"la dimension de mayor peso no encabeza el orden: {arb.orden_dimensiones}"
            )
            assert arb.metricas_cerradas == ["luz_tono.esquema_cromatico", "luz_tono.media_L"], (
                f"cerradas={arb.metricas_cerradas}"
            )
            assert not any(r.startswith("luz_tono.") for r in arb.metricas_citables), (
                "una metrica de una dimension cerrada aparece como citable"
            )
            # Y W no se renormaliza: el peso de lo no medido se declara, no se reparte.
            assert arb.vector_pesos.luz_tono == 0.35, (
                f"W se renormalizo: luz_tono={arb.vector_pesos.luz_tono}, se esperaba 0.35"
            )
            print("[OK] dimension entera cerrada: luz_tono cobertura 0.0 y AUN ASI primera "
                  "(W=0.35 sin renormalizar)")
        except AssertionError as err:
            fallos.append(("dimension entera cerrada", str(err)))
            print(f"[FAIL] dimension entera cerrada: {err}")
    return fallos


def test_confianzas_intermedias():
    """El corte es `> 0`, no `== 1`.

    Hoy los cuatro especialistas emiten la confianza binaria, asi que este camino no lo
    recorre ninguna ejecucion real. Pero el contrato admite confianzas graduadas —el
    refinamiento por concentracion de saliencia que el Agente 1 dejo anotado como
    post-MVP— y si alguien cambiara la comparacion a `== 1` esas metricas pasarian a
    CERRADAS en silencio: el critico dejaria de citar mediciones perfectamente validas y
    nada en la salida lo delataria.
    """
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        confianzas = _confianzas({
            "composicion_espacial": {
                "d_tercios": 0.5,      # intermedia -> citable
                "d_centro": 0.01,      # casi nula pero > 0 -> citable
                "d_equilibrio": 1.0,   # citable
                "patron_dominante": 0.0,  # la unica cerrada
            }
        })
        engine = _escribir(Path(tmp), confianzas)
        arb = engine.arbitrar(f"{STEM}.jpg", "paisaje")

        try:
            assert arb.cobertura_por_dimension["composicion_espacial"] == 0.75, (
                f"cobertura={arb.cobertura_por_dimension['composicion_espacial']}, se esperaba 0.75"
            )
            assert "composicion_espacial.d_tercios" in arb.metricas_citables, (
                "una confianza 0.5 no se conto como citable"
            )
            assert "composicion_espacial.d_centro" in arb.metricas_citables, (
                "una confianza 0.01 no se conto como citable"
            )
            assert arb.metricas_cerradas == ["composicion_espacial.patron_dominante"], (
                f"cerradas={arb.metricas_cerradas}"
            )
            print("[OK] confianzas intermedias: 0.5 y 0.01 son CITABLES, solo 0.0 cierra "
                  "(cobertura 0.75)")
        except AssertionError as err:
            fallos.append(("confianzas intermedias", str(err)))
            print(f"[FAIL] confianzas intermedias: {err}")
    return fallos


def test_cobertura_redondeo():
    """La cobertura se publica redondeada a 4 decimales, no como fraccion cruda."""
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        # 3 de 4 -> 0.75 exacto; se busca un caso periodico: 1 de 3 no existe con este
        # corpus, asi que se comprueba sobre 1 de 4 y 1 de 2, mas el tipo publicado.
        confianzas = _confianzas({
            "composicion_espacial": {"d_centro": 0.0, "d_equilibrio": 0.0, "d_tercios": 0.0},
            "lineas_direccion": {"angulo_horizonte": 0.0},
        })
        engine = _escribir(Path(tmp), confianzas)
        arb = engine.arbitrar(f"{STEM}.jpg", "paisaje")
        try:
            assert arb.cobertura_por_dimension["composicion_espacial"] == 0.25, (
                f"1 de 4 dio {arb.cobertura_por_dimension['composicion_espacial']}"
            )
            assert arb.cobertura_por_dimension["lineas_direccion"] == 0.5, (
                f"1 de 2 dio {arb.cobertura_por_dimension['lineas_direccion']}"
            )
            assert set(arb.cobertura_por_dimension) == set(INFORMES), (
                f"faltan dimensiones en la cobertura: {set(arb.cobertura_por_dimension)}"
            )
            print("[OK] cobertura: 1/4 -> 0.25, 1/2 -> 0.5, y las cuatro dimensiones presentes")
        except AssertionError as err:
            fallos.append(("cobertura", str(err)))
            print(f"[FAIL] cobertura: {err}")
    return fallos


def test_deteccion_por_forma():
    """Las metricas se detectan por FORMA, no por una lista de nombres.

    Es lo que permite anadir o renombrar una metrica en cualquier especialista sin tocar
    este modulo. La comprobacion importante es la simetrica: que los campos PLANOS —los
    recuentos, las varianzas, las rutas, las listas— NO se cuelen como metricas, porque
    inflarian el denominador de la cobertura y el critico leeria una aplicabilidad menor de
    la real.
    """
    fallos = []
    extras = {
        "lineas_direccion": {
            # Metrica nueva con un nombre que este modulo no conoce: debe entrar igual.
            "metrica_inventada_manana": _metrica(1.0),
            # Campos planos reales del Agente 2: no son metricas.
            "n_lineas_total": 65,
            "longitud_horizonte": 0.166,
            "coord_punto_fuga": [0.68, 0.36],
            "score_convergencia_nulo": 0.111,
            # Un dict que tiene `valor` pero NO `confianza`: tampoco lo es.
            "casi_metrica": {"valor": 0.4, "valor_norm": 0.4},
            "sin_valor": {"confianza": 1.0},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas(), extras=extras)
        arb = engine.arbitrar(f"{STEM}.jpg", "paisaje")

        rutas_lineas = [r for r in arb.metricas_citables + arb.metricas_cerradas
                        if r.startswith("lineas_direccion.")]
        try:
            assert "lineas_direccion.metrica_inventada_manana" in rutas_lineas, (
                "una metrica con nombre desconocido quedo FUERA del arbitraje"
            )
            assert len(rutas_lineas) == 3, f"se detectaron {len(rutas_lineas)} metricas: {rutas_lineas}"
            assert arb.cobertura_por_dimension["lineas_direccion"] == 1.0, (
                "un campo plano se colo en el denominador de la cobertura"
            )
            # Y la funcion suelta, que es la que fija el criterio.
            assert es_metrica({"valor": 1, "confianza": 1}) is True
            assert es_metrica({"valor": 1}) is False
            assert es_metrica(0.5) is False and es_metrica([1, 2]) is False
            print("[OK] deteccion por forma: la metrica desconocida entra, los 6 campos planos no")
        except AssertionError as err:
            fallos.append(("deteccion por forma", str(err)))
            print(f"[FAIL] deteccion por forma: {err}")
    return fallos


def test_informe_pelado():
    """El engine acepta tanto `{informe, diagnostico}` como el informe pelado."""
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas(), envolver=False)
        try:
            arb = engine.arbitrar(f"{STEM}.jpg", "paisaje")
            assert len(arb.metricas_citables) == 10, f"citables={len(arb.metricas_citables)}"
            print("[OK] informe pelado (sin envolver en {informe, diagnostico}) tambien se lee")
        except AssertionError as err:
            fallos.append(("informe pelado", str(err)))
            print(f"[FAIL] informe pelado: {err}")
    return fallos


# =============================================================================
# A) Casos analiticos — las cuatro formas de fallar RUIDOSAMENTE
# =============================================================================

def _debe_lanzar(nombre: str, fn) -> list:
    """Ejecuta `fn` y exige un ValueError. Fallar en silencio es el fallo caro."""
    try:
        fn()
    except ValueError:
        print(f"[OK] {nombre}")
        return []
    print(f"[FAIL] {nombre}: no lanzo ValueError")
    return [(nombre, "no lanzo ValueError")]


def test_fallos_ruidosos():
    """Las cuatro situaciones en las que arbitrar produciria una critica coherente y FALSA.

    Las cuatro tienen el mismo perfil y por eso van juntas: son fallos que, si se
    enmascararan, darian una salida perfectamente bien formada describiendo otra cosa. Es la
    clase de fallo que este proyecto ya ha pagado varias veces (el modelo nulo desincronizado,
    la fila 'otro' como red de seguridad silenciosa, las claves duplicadas de los YAML).
    """
    fallos = []

    # 1) Falta un informe: el critico no puede arbitrar sobre tres dimensiones.
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas(), omitir="luz_tono")
        fallos += _debe_lanzar(
            "falta un informe -> lanza (no arbitra sobre tres dimensiones)",
            lambda: engine.arbitrar(f"{STEM}.jpg", "paisaje"),
        )

    # 2) Un informe sin ninguna metrica reconocible: o el fichero esta corrupto o el
    #    contrato de MetricaConfianza ha cambiado y este modulo se quedo atras.
    with tempfile.TemporaryDirectory() as tmp:
        confianzas = _confianzas()
        confianzas["luz_tono"] = {}
        engine = _escribir(Path(tmp), confianzas, extras={"luz_tono": {"std_L": 23.5}})
        fallos += _debe_lanzar(
            "informe sin ninguna metrica {valor, confianza} -> lanza",
            lambda: engine.arbitrar(f"{STEM}.jpg", "paisaje"),
        )

    # 3) Informes de OTRA imagen. El caso fino es `imagen3` contra informes de `imagen30`:
    #    sin el guion bajo del prefijo, uno casaria con el otro y el critico escribiria
    #    sobre imagen3 citando las metricas de imagen30.
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas(), stem="imagen30")
        fallos += _debe_lanzar(
            "informes de imagen30 arbitrados como imagen3 -> lanza (el guion bajo del prefijo)",
            lambda: engine.arbitrar("imagen3.jpg", "paisaje"),
        )
        # Control: con la imagen correcta NO puede lanzar, o la guarda seria inutilizable.
        try:
            engine.arbitrar("imagen30.jpg", "paisaje")
            print("[OK] control: los informes de imagen30 SI arbitran como imagen30")
        except ValueError as err:
            fallos.append(("control imagen30", str(err)))
            print(f"[FAIL] control imagen30: {err}")

    # 4) Etiqueta de contexto desconocida: no se cae en la fila uniforme 'otro'.
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas())
        fallos += _debe_lanzar(
            "etiqueta de contexto inexistente -> lanza (no cae en la fila 'otro')",
            lambda: engine.arbitrar(f"{STEM}.jpg", "contexto_que_no_existe"),
        )

    return fallos


def test_hueco_conocido_sin_verificacion_path():
    """LIMITACION DECLARADA, no un fallo: sin `verificacion_path` la guarda no se aplica.

    `_cargar_informe` solo comprueba la imagen `if verificacion:`. Un informe que no traiga
    ese campo pasa sin verificar. Hoy no es explotable —los cuatro especialistas lo emiten
    siempre— y la tolerancia es lo que permite fabricar casos, pero conviene que quede
    escrito: la guarda protege a los informes que LLEVAN la ruta, no a todos. Si algun dia se
    quita ese campo de un schema, esta comprobacion deja de proteger nada y el test lo dira
    al empezar a lanzar aqui.
    """
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        for dimension, fichero in INFORMES.items():
            informe = {nombre: _metrica(1.0) for nombre in CORPUS[dimension]}
            (raiz / fichero).write_text(json.dumps(informe), encoding="utf-8")
        try:
            PrioridadEngine(raiz=raiz).arbitrar("imagen_cualquiera.jpg", "paisaje")
            print("[OK] hueco conocido: un informe sin verificacion_path NO se verifica "
                  "(documentado, no explotable hoy)")
        except ValueError as err:
            fallos.append(("hueco conocido", f"el comportamiento documentado cambio: {err}"))
            print(f"[FAIL] hueco conocido: el comportamiento documentado cambio: {err}")
    return fallos


# =============================================================================
# B) Orden por W: ground truth escrito a mano
# =============================================================================

# Derivado a mano de la matriz de parametros/pesos_contextuales.json, NO recalculado con el
# mismo algoritmo que se esta probando (eso seria circular). Orden canonico de desempate:
# composicion_espacial, lineas_direccion, espacio_aislamiento_sujeto, luz_tono.
ORDEN_ESPERADO = {
    # animal              0.24, 0.14, 0.32, 0.30
    "animal": ["espacio_aislamiento_sujeto", "luz_tono", "composicion_espacial", "lineas_direccion"],
    # arquitectura        0.22, 0.37, 0.17, 0.24
    "arquitectura": ["lineas_direccion", "luz_tono", "composicion_espacial", "espacio_aislamiento_sujeto"],
    # retrato-humano      0.22, 0.10, 0.31, 0.37
    "retrato-humano": ["luz_tono", "espacio_aislamiento_sujeto", "composicion_espacial", "lineas_direccion"],
    # paisaje             0.24, 0.19, 0.22, 0.35
    "paisaje": ["luz_tono", "composicion_espacial", "espacio_aislamiento_sujeto", "lineas_direccion"],
    # producto-still_life 0.26, 0.11, 0.23, 0.40
    "producto-still_life": ["luz_tono", "composicion_espacial", "espacio_aislamiento_sujeto", "lineas_direccion"],
    # otro                0.25 x 4 -> empate a cuatro bandas: manda el orden canonico
    "otro": ["composicion_espacial", "lineas_direccion", "espacio_aislamiento_sujeto", "luz_tono"],
}


def test_orden_por_w():
    """El orden de la critica es el de W, y la fila aplicada se transcribe sin alterar."""
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas())
        for etiqueta, esperado in ORDEN_ESPERADO.items():
            try:
                arb = engine.arbitrar(f"{STEM}.jpg", etiqueta)
                assert arb.orden_dimensiones == esperado, (
                    f"orden={arb.orden_dimensiones}, esperado={esperado}"
                )
                # La fila publicada es exactamente la del fichero: el arbitraje no la toca.
                fila = VectorPesosW(**vector_pesos_engine.obtener(etiqueta))
                assert arb.vector_pesos == fila, f"W alterado: {arb.vector_pesos} vs {fila}"
                assert arb.etiqueta_contexto == etiqueta
                pesos = " ".join(f"{d.split('_')[0][:4]}={getattr(fila, d)}" for d in esperado)
                print(f"[OK] {etiqueta:<20} {pesos}")
            except AssertionError as err:
                fallos.append((etiqueta, str(err)))
                print(f"[FAIL] {etiqueta}: {err}")
    return fallos


def test_desempate_estable():
    """La fila uniforme 'otro' no puede producir dos criticas ordenadas distinto.

    Es el motivo de que el desempate exista. Con 0.25 en las cuatro dimensiones, un
    ordenamiento sin segundo criterio depende de detalles de implementacion (el orden de
    iteracion de un dict, la estabilidad del sort), y dos ejecuciones identicas del sistema
    darian criticas con las dimensiones en distinto orden. Se comprueba repitiendo el
    arbitraje completo y exigiendo `model_dump()` identico, igual que el test de determinismo
    del Agente 4.
    """
    fallos = []
    with tempfile.TemporaryDirectory() as tmp:
        engine = _escribir(Path(tmp), _confianzas({"luz_tono": {"media_L": 0.0}}))
        primera = engine.arbitrar(f"{STEM}.jpg", "otro").model_dump()
        for i in range(4):
            otra = engine.arbitrar(f"{STEM}.jpg", "otro").model_dump()
            if otra != primera:
                fallos.append(("desempate estable", f"la repeticion {i + 1} difiere"))
                print(f"[FAIL] desempate estable: la repeticion {i + 1} difiere")
                return fallos
        print(f"[OK] desempate estable: 5 arbitrajes identicos con la fila uniforme 'otro' "
              f"-> {primera['orden_dimensiones']}")
    return fallos


# =============================================================================
# C) Integracion sobre los informes REALES, si los hay
# =============================================================================

def _corpus_real() -> tuple[str, str] | None:
    """Devuelve (stem, etiqueta_contexto) si los cinco ficheros reales son coherentes.

    Se mira la ejecución más reciente de `outputs/ejecucion/` (antes era la raíz del
    proyecto, que es donde caían los informes cuando el directorio de salida era único).
    """
    base = ultima_ejecucion()
    if base is None:
        return None

    orquestador = base / "salida_orquestador.json"
    if not orquestador.exists():
        return None
    etiqueta = json.loads(orquestador.read_text(encoding="utf-8")).get("etiqueta_contexto")

    stems = set()
    for fichero in INFORMES.values():
        destino = base / fichero
        if not destino.exists():
            return None
        datos = json.loads(destino.read_text(encoding="utf-8"))
        ruta = datos.get("informe", datos).get("verificacion_path")
        if not ruta:
            return None
        stems.add(Path(ruta).stem.rsplit("_", 1)[0])

    if len(stems) != 1 or not etiqueta:
        return None
    return stems.pop(), etiqueta


def test_integracion_real(stem: str, etiqueta: str):
    """Recomputa el arbitraje desde los ficheros y lo compara con el publicado.

    Es el mismo invariante que sostiene los tests de los cuatro especialistas: un tercero
    reconstruye la salida leyendo solo lo que el sistema publica. Aqui verifica ademas el
    camino que los casos fabricados no tocan —desenvolver `{informe, diagnostico}` real,
    convivir con los campos planos de verdad— y que ninguna metrica de los informes se queda
    fuera del arbitraje EN SILENCIO, que es el modo de fallo que la deteccion por forma
    existe para evitar.
    """
    fallos = []
    # Sin `raiz`, el engine deriva el directorio del propio stem: es justo el camino que
    # recorre en producción, así que este test lo ejercita tal cual.
    arb = PrioridadEngine().arbitrar(f"{stem}.jpg", etiqueta)

    base = ultima_ejecucion()
    citables, cerradas, cobertura = [], [], {}
    for dimension, fichero in INFORMES.items():
        datos = json.loads((base / fichero).read_text(encoding="utf-8"))
        informe = datos.get("informe", datos)
        metricas = {k: v for k, v in informe.items() if es_metrica(v)}
        abiertas = [n for n, m in metricas.items() if m["confianza"] > 0]
        citables += [f"{dimension}.{n}" for n in sorted(abiertas)]
        cerradas += [f"{dimension}.{n}" for n in sorted(set(metricas) - set(abiertas))]
        cobertura[dimension] = round(len(abiertas) / len(metricas), 4)

    try:
        assert sorted(arb.metricas_citables) == sorted(citables), (
            f"citables publicadas {arb.metricas_citables} vs recomputadas {citables}"
        )
        assert sorted(arb.metricas_cerradas) == sorted(cerradas), (
            f"cerradas publicadas {arb.metricas_cerradas} vs recomputadas {cerradas}"
        )
        assert arb.cobertura_por_dimension == cobertura, (
            f"cobertura publicada {arb.cobertura_por_dimension} vs recomputada {cobertura}"
        )
        print(f"[OK] {stem} ({etiqueta}): {len(citables)} citables, {len(cerradas)} cerradas, "
              f"recomputadas desde los JSON")
        for dimension in arb.orden_dimensiones:
            peso = getattr(arb.vector_pesos, dimension)
            print(f"       {dimension:<28} W={peso:<5} cobertura={arb.cobertura_por_dimension[dimension]}")
    except AssertionError as err:
        fallos.append((stem, str(err)))
        print(f"[FAIL] {stem}: {err}")

    # La tool devuelve un dict serializable: el LLM tiene que ver JSON limpio, igual que en
    # los cuatro especialistas.
    salida = PrioridadTool()._run(ruta_imagen=f"{stem}.jpg", etiqueta_contexto=etiqueta)
    try:
        assert isinstance(salida, dict), f"la tool devolvio {type(salida).__name__}, no dict"
        json.dumps(salida, ensure_ascii=False)
        assert set(salida) == set(arb.model_dump()), f"claves de la tool: {sorted(salida)}"
        print(f"[OK] la tool devuelve un dict serializable con {len(salida)} claves")
    except (AssertionError, TypeError) as err:
        fallos.append(("tool", str(err)))
        print(f"[FAIL] tool: {err}")

    return fallos


# =============================================================================

def main():
    print("=" * 78)
    print("A) CASOS ANALITICOS — informes FABRICADOS")
    print("=" * 78)
    fallos = test_todas_citables()
    fallos += test_dimension_entera_cerrada()
    fallos += test_confianzas_intermedias()
    fallos += test_cobertura_redondeo()
    fallos += test_deteccion_por_forma()
    fallos += test_informe_pelado()

    print("\n" + "=" * 78)
    print("A) CASOS ANALITICOS — fallos que deben ser RUIDOSOS")
    print("=" * 78)
    fallos += test_fallos_ruidosos()
    fallos += test_hueco_conocido_sin_verificacion_path()

    print("\n" + "=" * 78)
    print("B) ORDEN POR W — ground truth de las 6 filas + desempate")
    print("=" * 78)
    fallos += test_orden_por_w()
    fallos += test_desempate_estable()

    print("\n" + "=" * 78)
    print("C) INTEGRACION — informes reales en disco")
    print("=" * 78)
    real = _corpus_real()
    if real is None:
        print("[SKIP] no hay cuatro informes coherentes de la misma imagen en la raiz del "
              "proyecto. Ejecuta `crewai run` para generarlos; este bloque no cuenta como fallo.")
    else:
        fallos += test_integracion_real(*real)

    print("\n--- Resumen ---")
    if fallos:
        print(f"{len(fallos)} FALLOS:")
        for nombre, err in fallos:
            print(f"  - {nombre}: {err}")
        raise SystemExit(1)
    print("Todo OK.")


if __name__ == "__main__":
    main()
