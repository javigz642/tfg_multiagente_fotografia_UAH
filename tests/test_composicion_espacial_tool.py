"""Prueba manual de ComposicionEspacialTool."""
from math import isclose, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory


import cv2
import numpy as np

from tfg_multiagente_fotografia.tools.shared_perception_tool import SharedPerceptionTool
from tfg_multiagente_fotografia.tools.composicion_espacial_tool import (
    composicion_espacial_engine,
    COTA_D_TERCIOS,
    COTA_D_CENTRO,
    FUENTE_CON_IDENTIDAD,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Se prueba contra el ENGINE, no contra la tool: el engine devuelve el modelo Pydantic
# (el núcleo determinista que interesa validar), mientras que la tool es solo el
# adaptador que lo serializa a dict para CrewAI.
percepcion_tool = SharedPerceptionTool()

# Los 4 puntos de fuerza en coordenadas normalizadas: coord_p_cercano debe ser
# siempre uno de ellos, sea cual sea la imagen.
PUNTOS_FUERZA_NORM = {(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)}

# Tolerancia para comparar reconstrucciones en píxeles (solo absorbe el ida y
# vuelta de normalizar y desnormalizar en coma flotante).
TOL = 1e-6


def assert_fichero_valido(ruta_str: str, descripcion: str):
    ruta = Path(ruta_str)
    assert ruta.exists(), f"{descripcion} no se guardó en disco: {ruta}"
    assert ruta.stat().st_size > 0, f"{descripcion} está vacío: {ruta}"


def validar_informe(informe, sujeto_discreto: bool, percepcion: dict):
    w, h = informe.dimensiones_imagen
    diagonal = sqrt(w**2 + h**2)
    t, c, e = informe.d_tercios, informe.d_centro, informe.d_equilibrio

    # --- Cotas exactas (ver docstring de InformeComposicionEspacial) ---
    assert 0 <= t.valor <= COTA_D_TERCIOS, f"d_tercios={t.valor} fuera de [0, 1/3]"
    assert 0 <= c.valor <= COTA_D_CENTRO, f"d_centro={c.valor} fuera de [0, 1/2]"
    assert 0 <= e.valor <= 0.5, f"d_equilibrio={e.valor} fuera de [0, 1/2]"

    # --- Coordenadas normalizadas ---
    for nombre, (x, y) in (
        ("coord_centroide", informe.coord_centroide),
        ("coord_p_cercano", informe.coord_p_cercano),
        ("coord_centro_masa", informe.coord_centro_masa),
    ):
        assert 0 <= x <= 1 and 0 <= y <= 1, f"{nombre}={(x, y)} fuera de [0,1]"

    assert any(
        isclose(informe.coord_p_cercano[0], px, abs_tol=1e-9)
        and isclose(informe.coord_p_cercano[1], py, abs_tol=1e-9)
        for px, py in PUNTOS_FUERZA_NORM
    ), f"coord_p_cercano={informe.coord_p_cercano} no es uno de los 4 puntos de fuerza"

    # --- Consistencia coordenadas <-> distancias ---
    # Si el crítico cita "el sujeto está en (0.31, 0.66) a 0.08 del punto de fuerza",
    # ambas cifras tienen que describir la MISMA geometría. Aquí se recalcula la
    # distancia a partir de las coordenadas publicadas y se compara con la publicada.
    def dist_norm(p1, p2):
        return sqrt(((p2[0] - p1[0]) * w) ** 2 + ((p2[1] - p1[1]) * h) ** 2) / diagonal

    assert isclose(dist_norm(informe.coord_centroide, informe.coord_p_cercano), t.valor, abs_tol=TOL), (
        "d_tercios no coincide con la distancia entre coord_centroide y coord_p_cercano"
    )
    assert isclose(dist_norm(informe.coord_centroide, (0.5, 0.5)), c.valor, abs_tol=TOL), (
        "d_centro no coincide con la distancia entre coord_centroide y el centro geométrico"
    )
    assert isclose(dist_norm(informe.coord_centro_masa, (0.5, 0.5)), e.valor, abs_tol=TOL), (
        "d_equilibrio no coincide con la distancia entre coord_centro_masa y el centro geométrico"
    )

    # --- Regla de anclaje: bbox si hay sujeto, centroide de saliencia si no ---
    esperado = percepcion["bbox"][:2] if sujeto_discreto else percepcion["centroide_saliencia"][:2]
    anclaje_px = (informe.coord_centroide[0] * w, informe.coord_centroide[1] * h)
    assert isclose(anclaje_px[0], esperado[0], abs_tol=1e-3) and isclose(
        anclaje_px[1], esperado[1], abs_tol=1e-3
    ), (
        f"anclaje {anclaje_px} no corresponde a {'bbox' if sujeto_discreto else 'centroide_saliencia'} {esperado}"
    )

    # --- Arbitraje entre hipótesis rivales ---
    # Se recalcula aquí el veredicto de forma independiente y se compara con el
    # publicado: es la operación que el crítico ya NO tiene que hacer, así que debe
    # quedar blindada. La comparación va sobre los `valor` (comparten diagonal),
    # nunca sobre los `valor_norm` (escalados por cotas distintas, no comparables).
    p = informe.patron_dominante
    esperado_patron = "centrada" if c.valor < t.valor else "tercios"
    assert p.valor == esperado_patron, (
        f"patron_dominante={p.valor} pero d_tercios={t.valor:.6f} y d_centro={c.valor:.6f} "
        f"implican '{esperado_patron}'"
    )
    assert isclose(informe.margen_patron, abs(t.valor - c.valor), abs_tol=TOL), (
        f"margen_patron={informe.margen_patron} no es |d_tercios - d_centro|"
    )
    assert p.valor_norm is None, "patron_dominante es una categoría, no una adherencia: valor_norm=None"

    # --- Confianza binaria derivada de `fuente` (contrato del Nivel 2) ---
    # Se recomputa desde el campo publicado por la percepción compartida, igual que hacen
    # los tests de los Agentes 2, 3 y 4: un tercero debe poder reconstruir POR QUÉ el
    # agente se calló sin leer el código. OJO a lo que NO es: el gate mira `fuente`, no
    # `sujeto_discreto`, así que una imagen puede anclarse en un bbox perfectamente
    # definido (sujeto_discreto=True) y salir con confianza 0.
    esperada = 1.0 if percepcion["fuente"] == FUENTE_CON_IDENTIDAD else 0.0
    assert t.confianza == esperada, f"d_tercios.confianza={t.confianza}, se esperaba {esperada}"
    assert c.confianza == t.confianza, "d_centro comparte anclaje con d_tercios: debe compartir confianza"
    assert p.confianza == t.confianza, (
        "patron_dominante arbitra entre dos distancias ancladas al sujeto: debe compartir su confianza"
    )
    assert e.confianza == 1.0, "d_equilibrio es global: su confianza siempre es plena"
    for nombre, m in (("d_tercios", t), ("d_centro", c), ("patron_dominante", p), ("d_equilibrio", e)):
        assert m.fuente_confianza, f"{nombre} no explica su confianza en fuente_confianza"

    # --- La explicación distingue los DOS motivos de cierre, que no son el mismo ---
    # El gate se cierra por dos escenas distintas —hay una región destacada pero sin
    # identidad que la confirme como sujeto, o no hay ningún objeto focal— y el prompt
    # pide redactarlas de forma diferente. Si `fuente_confianza` no las separa, el
    # diagnóstico no puede hacerlo.
    if percepcion["fuente"] == FUENTE_CON_IDENTIDAD:
        assert "yolo" in t.fuente_confianza.lower(), (
            f"gate abierto pero fuente_confianza no dice por qué: {t.fuente_confianza!r}"
        )
    else:
        # Con el gate cerrado el mensaje tiene que nombrar la fuente real, que es lo
        # único que separa "hay una región destacada sin identidad" de "no hay ningún
        # objeto focal": son dos escenas distintas y el prompt las redacta distinto.
        assert percepcion["fuente"] in t.fuente_confianza, (
            f"gate cerrado por fuente={percepcion['fuente']} pero fuente_confianza no la "
            f"nombra: {t.fuente_confianza!r}"
        )
    for nombre, m in (("d_centro", c), ("patron_dominante", p)):
        assert m.fuente_confianza == t.fuente_confianza, (
            f"{nombre} comparte anclaje con d_tercios: debe compartir también su explicación"
        )

    # --- valor_norm: convención única "1 = adherencia máxima" ---
    for nombre, m, cota in (("d_tercios", t, COTA_D_TERCIOS), ("d_centro", c, COTA_D_CENTRO)):
        assert m.valor_norm is not None, f"{nombre} debe emitir valor_norm"
        esperado_norm = min(1.0, max(0.0, 1.0 - m.valor / cota))
        assert isclose(m.valor_norm, esperado_norm, abs_tol=TOL), (
            f"{nombre}.valor_norm={m.valor_norm} no es 1 - valor/{cota:.4f} = {esperado_norm}"
        )
    assert e.valor_norm is None, (
        "d_equilibrio debe emitir valor_norm=None: su cota teórica no refleja el rango real "
        "y está pendiente de calibración empírica (validación §5)"
    )

    assert_fichero_valido(informe.verificacion_path, "La imagen de verificación visual")


def casos_analiticos():
    """Ground truth exacto sobre imágenes sintéticas de 900x600.

    Puntos de fuerza en (300,200), (600,200), (300,400), (600,400); centro en
    (450,300). El mapa de saliencia es uniforme, luego su centro de masa es el
    centro de la imagen y d_equilibrio ~ 0 en los tres casos.
    """
    w, h = 900, 600
    casos = [
        # (nombre, anclaje px, sujeto_discreto, fuente, d_tercios esp., d_centro esp., patrón esp.)
        ("sobre_punto_fuerza", (300, 200), True, "yolo", 0.0, 1 / 6, "tercios"),
        ("centrado", (450, 300), True, "yolo", 1 / 6, 0.0, "centrada"),
        # EL CASO QUE BLINDA LA ORTOGONALIDAD VÍA/GATE, y el que hay que copiar si se
        # vuelve a tocar el gate: MISMO anclaje y MISMAS cifras que "sobre_punto_fuerza",
        # pero con el bbox procedente del blob de saliencia. Comprueba que el gate mira la
        # PROCEDENCIA y no la medida, y que el valor publicado NO depende del gate —que es
        # lo que exige la regla "ningún especialista silencia una métrica"—. Es el
        # equivalente del caso A7 del Agente 3.
        ("punto_fuerza_saliencia", (300, 200), True, "saliencia", 0.0, 1 / 6, "tercios"),
        # En la esquina AMBAS distancias son máximas, pero el punto de fuerza queda
        # más cerca que el centro (1/3 < 1/2), así que el veredicto es "tercios".
        # Es contraintuitivo y por eso conviene fijarlo: el patrón dominante dice a
        # qué referencia está más próximo el anclaje, no que la foto esté lograda.
        # Aquí llega además con confianza 0 y margen amplio (1/6).
        ("esquina_sin_sujeto", (0, 0), False, "sin_sujeto_claro",
         COTA_D_TERCIOS, COTA_D_CENTRO, "tercios"),
    ]

    fallos = []
    generados = []
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_path = tmp / "analitico.png"
        sal_path = tmp / "analitico_saliency.png"
        cv2.imwrite(str(img_path), np.full((h, w, 3), 200, dtype=np.uint8))
        cv2.imwrite(str(sal_path), np.full((h, w), 128, dtype=np.uint8))

        for nombre, anclaje, sujeto_discreto, fuente, d_t_esp, d_c_esp, patron_esp in casos:
            # El anclaje se inyecta por el campo que corresponda según el flag:
            # con sujeto se lee bbox[:2], sin sujeto se lee centroide_saliencia.
            bbox = [float(anclaje[0]), float(anclaje[1]), 10.0, 10.0]
            centroide = [int(anclaje[0]), int(anclaje[1])]
            informe = composicion_espacial_engine.analizar(
                ruta_imagen=str(img_path),
                bbox=bbox,
                centroide_saliencia=centroide,
                sujeto_discreto=sujeto_discreto,
                fuente=fuente,
                mapa_saliencia_path=str(sal_path),
            )
            generados.append(Path(informe.verificacion_path))
            try:
                assert informe.dimensiones_imagen == (w, h), (
                    f"dimensiones_imagen={informe.dimensiones_imagen}, se esperaba {(w, h)}"
                )
                assert isclose(informe.d_tercios.valor, d_t_esp, abs_tol=1e-9), (
                    f"d_tercios={informe.d_tercios.valor}, se esperaba {d_t_esp}"
                )
                assert isclose(informe.d_centro.valor, d_c_esp, abs_tol=1e-9), (
                    f"d_centro={informe.d_centro.valor}, se esperaba {d_c_esp}"
                )
                assert informe.patron_dominante.valor == patron_esp, (
                    f"patron_dominante={informe.patron_dominante.valor}, se esperaba {patron_esp}"
                )
                assert isclose(informe.margen_patron, abs(d_t_esp - d_c_esp), abs_tol=1e-9), (
                    f"margen_patron={informe.margen_patron}, se esperaba {abs(d_t_esp - d_c_esp)}"
                )
                # Mapa uniforme -> centro de masa en el centro geométrico. No es 0
                # exacto porque el centro de masa de una rejilla de w píxeles cae en
                # (w-1)/2, medio píxel a la izquierda del centro geométrico w/2.
                assert informe.d_equilibrio.valor < 1e-3, (
                    f"mapa uniforme pero d_equilibrio={informe.d_equilibrio.valor}"
                )
                # El gate mira la PROCEDENCIA del bbox, no la medida ni el flag de vía.
                # Es lo que separa a "punto_fuerza_saliencia" de "sobre_punto_fuerza",
                # que traen exactamente las mismas cifras.
                conf_esp = 1.0 if fuente == FUENTE_CON_IDENTIDAD else 0.0
                assert informe.d_tercios.confianza == conf_esp, (
                    f"fuente={fuente} pero d_tercios.confianza={informe.d_tercios.confianza}, "
                    f"se esperaba {conf_esp}"
                )
                print(
                    f"[OK] analitico/{nombre}: d_tercios={informe.d_tercios.valor:.4f} "
                    f"(adh={informe.d_tercios.valor_norm:.2f}) "
                    f"d_centro={informe.d_centro.valor:.4f} "
                    f"(adh={informe.d_centro.valor_norm:.2f}) "
                    f"conf={informe.d_tercios.confianza:.0f} -> {informe.patron_dominante.valor} "
                    f"(margen={informe.margen_patron:.4f})"
                )
            except AssertionError as err:
                fallos.append((f"analitico/{nombre}", str(err)))
                print(f"[FAIL] analitico/{nombre}: {err}")

    # El engine guarda su PNG de verificación siempre; el de las imágenes
    # sintéticas no aporta nada en outputs/, que es material de revisión real.
    for ruta in generados:
        ruta.unlink(missing_ok=True)

    return len(casos), fallos


def integracion_sobre_data():
    """Encadena percepción compartida -> composición espacial sobre data/."""
    imagenes = sorted(DATA_DIR.glob("imagen*.jpg"))
    assert imagenes, f"No se encontraron imágenes de prueba en {DATA_DIR}"

    fallos = []
    patrones = {"tercios": 0, "centrada": 0}
    # Reparto por fuente y recuento del gate. Es la cifra que va a la memoria y la que
    # hay que rehacer cuando crezca data/: dice sobre cuántas imágenes se midió el efecto
    # del gate, y separa las dos poblaciones que la revisión visual tuvo que distinguir.
    por_fuente = {}
    gate_cerrado = 0
    # Se recogen los márgenes observados para tener base empírica sobre lo separadas
    # que están en la práctica las dos hipótesis rivales (insumo para decidir en §5
    # si algún día procede un umbral de "ambigua", que hoy NO se inventa).
    margenes = []

    for image_path in imagenes:
        percepcion = percepcion_tool._run(str(image_path))
        informe = composicion_espacial_engine.analizar(
            ruta_imagen=str(image_path),
            bbox=percepcion["bbox"],
            centroide_saliencia=percepcion["centroide_saliencia"],
            sujeto_discreto=percepcion["sujeto_discreto"],
            mapa_saliencia_path=percepcion["mapa_saliencia_path"],
            fuente=percepcion["fuente"],
        )
        try:
            validar_informe(informe, percepcion["sujeto_discreto"], percepcion)

            # El veredicto ya viene resuelto por el engine; validar_informe ha
            # comprobado que coincide con recalcularlo desde los `valor`.
            patrones[informe.patron_dominante.valor] += 1
            margenes.append(informe.margen_patron)
            por_fuente[percepcion["fuente"]] = por_fuente.get(percepcion["fuente"], 0) + 1
            if informe.d_tercios.confianza == 0.0:
                gate_cerrado += 1

            print(
                f"[OK] {image_path.name}: fuente={percepcion['fuente']} "
                f"d_tercios={informe.d_tercios.valor:.4f} (adh={informe.d_tercios.valor_norm:.2f}) "
                f"d_centro={informe.d_centro.valor:.4f} (adh={informe.d_centro.valor_norm:.2f}) "
                f"d_equilibrio={informe.d_equilibrio.valor:.4f} "
                f"conf={informe.d_tercios.confianza:.0f} -> {informe.patron_dominante.valor} "
                f"(margen={informe.margen_patron:.4f})"
            )
        except AssertionError as err:
            fallos.append((image_path.name, str(err)))
            print(f"[FAIL] {image_path.name}: {err}")

    return len(imagenes), fallos, patrones, por_fuente, gate_cerrado, margenes


def main():
    print("--- Casos analíticos (ground truth exacto) ---")
    n_analiticos, fallos_analiticos = casos_analiticos()

    print("\n--- Integración sobre data/ (percepción compartida -> composición espacial) ---")
    n_imagenes, fallos_data, patrones, por_fuente, gate_cerrado, margenes = integracion_sobre_data()

    fallos = fallos_analiticos + fallos_data
    print("\n--- Resumen ---")
    print(f"{n_analiticos - len(fallos_analiticos)}/{n_analiticos} casos analíticos OK")
    print(f"{n_imagenes - len(fallos_data)}/{n_imagenes} imágenes de data/ OK")
    print("Patrón dominante: " + ", ".join(f"{k}={v}" for k, v in patrones.items()))
    if margenes:
        print(
            f"Margen entre hipótesis: min={min(margenes):.4f} "
            f"max={max(margenes):.4f} media={sum(margenes) / len(margenes):.4f}"
        )
    print("Fuente del bbox: " + ", ".join(f"{k}={v}" for k, v in sorted(por_fuente.items())))
    print(
        f"Gate de d_tercios/d_centro/patron_dominante CERRADO en {gate_cerrado}/{n_imagenes} "
        f"(d_equilibrio sigue abierta en las {n_imagenes})"
    )
    if fallos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
