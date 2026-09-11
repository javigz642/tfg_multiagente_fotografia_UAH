"""Prueba manual de LuzTonoTool."""
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from tfg_multiagente_fotografia.tools.luz_tono_tool import (
    DECIMALES_MATIZ,
    ESQUEMA_SIN_COLOR,
    L_CENTRO,
    L_MAX,
    PESO_MIN_MATIZ,
    RATIO_CROMATICO_MIN,
    SPREAD_ANALOGO,
    SPREAD_MONOCROMATICO,
    UMBRAL_FUSION_MATIZ,
    clasificar_esquema,
    dif_matiz,
    luz_tono_engine,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Lienzo pequeño: aquí no hay ningún umbral en píxeles ni detección de bordes, y
# todas las métricas son estadísticos de la distribución, así que el tamaño solo
# afecta al tiempo de cómputo. Se mantiene por debajo del lado de trabajo para que el
# engine no reescale y lo pintado sea exactamente lo que se mide.
ANCHO, ALTO = 600, 400

ETIQUETAS_VALIDAS = {"monocromático", "análogo", "complementario", "otro"}

# Tolerancia del matiz: OpenCV guarda H como H/2 en 8 bits, así que cuantiza a 2°, y
# el centroide de un grupo se desplaza otro poco. 3° cubre las dos cosas de sobra.
TOL_MATIZ = 3.0
TOL = 1e-9


def assert_fichero_valido(ruta_str, descripcion):
    ruta = Path(ruta_str)
    assert ruta.exists(), f"{descripcion} no se guardó en disco: {ruta}"
    assert ruta.stat().st_size > 0, f"{descripcion} está vacío: {ruta}"


def lienzo_hsv(bandas):
    """Lienzo dividido en bandas verticales de matiz EXACTO conocido.

    `bandas` es una lista de matices en grados [0,360). Cada uno se pinta puro
    (S=V=máximo) para que sobreviva al filtro acromático sin margen de duda, y todas
    las bandas tienen el mismo ancho, de modo que la masa de cada matiz es 1/n y
    supera PESO_MIN_MATIZ mientras n <= 6.
    """
    img = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    ancho_banda = ANCHO // len(bandas)
    for i, matiz in enumerate(bandas):
        x0 = i * ancho_banda
        x1 = ANCHO if i == len(bandas) - 1 else x0 + ancho_banda
        # OpenCV codifica H como grados/2 para que quepa en 8 bits.
        parche = np.full((ALTO, x1 - x0, 3), (int(round(matiz / 2)), 255, 255), dtype=np.uint8)
        img[:, x0:x1] = cv2.cvtColor(parche, cv2.COLOR_HSV2BGR)
    return img


def lienzo_gris(nivel):
    """Lienzo acromático uniforme (S = 0): sin matiz que medir, gate cerrado."""
    return np.full((ALTO, ANCHO, 3), nivel, dtype=np.uint8)


def lienzo_medio_negro_medio_blanco():
    """Mitad negra y mitad blanca: exposición con ground truth exacto.

    Es el único caso cuya exposición se conoce sin depender de la conversión de
    color: media_L = 50 y std_L = 50 por construcción, con el 50% de los píxeles
    recortados por cada extremo. Sirve además para comprobar que la adherencia da 1.0
    en una imagen que NO es "correcta" en ningún sentido fotográfico — la adherencia
    mide conformidad con la exposición centrada, no calidad.
    """
    img = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    img[:, ANCHO // 2:] = 255
    return img


def validar_informe(informe):
    """Invariantes que deben cumplirse en CUALQUIER imagen, sintética o real."""
    m = informe.media_L
    e = informe.esquema_cromatico

    # --- Exposición: dominio, evidencia y adherencia recomputable ---
    assert 0.0 <= m.valor <= L_MAX, f"media_L={m.valor} fuera de [0,{L_MAX}]"
    assert informe.std_L >= 0.0, f"std_L={informe.std_L} negativa"
    for nombre, pct in (("sombras", informe.pct_clipping_sombras),
                        ("luces", informe.pct_clipping_luces)):
        assert 0.0 <= pct <= 100.0, f"pct_clipping_{nombre}={pct} fuera de [0,100]"
    assert informe.pct_clipping_sombras + informe.pct_clipping_luces <= 100.0 + TOL, (
        "un píxel no puede estar recortado por los dos extremos a la vez"
    )
    esperado_norm = min(1.0, max(0.0, 1.0 - abs(m.valor - L_CENTRO) / L_CENTRO))
    assert m.valor_norm is not None, "media_L siempre es medible, luego siempre tiene adherencia"
    assert isclose(m.valor_norm, esperado_norm, abs_tol=TOL), (
        f"media_L.valor_norm={m.valor_norm} no es 1 - |valor-{L_CENTRO}|/{L_CENTRO}"
    )

    # --- La ASIMETRÍA del agente, convertida en aserción ---
    # La exposición NO tiene gate: no existe el caso "no medible", así que una
    # confianza distinta de 1 aquí significaría que alguien le ha inventado una
    # condición de aplicabilidad, que es justo lo que el contrato prohíbe.
    assert m.confianza == 1.0, (
        f"media_L.confianza={m.confianza}: la exposición es medible en cualquier imagen "
        "y su confianza es 1.0 por construcción, no por ser esta foto favorable"
    )

    # --- El gate cromático es REPRODUCIBLE desde el campo publicado ---
    # Una sola condición, y el campo con el que se decide viaja en crudo en el
    # informe: cualquiera puede recomputar por qué el agente afirmó o se calló.
    assert 0.0 <= informe.ratio_pixeles_cromaticos <= 1.0
    abre = informe.ratio_pixeles_cromaticos >= RATIO_CROMATICO_MIN
    assert e.confianza == (1.0 if abre else 0.0), (
        f"esquema_cromatico.confianza={e.confianza} no coincide con recomputar su condición "
        f"(ratio={informe.ratio_pixeles_cromaticos:.4f}, mínimo {RATIO_CROMATICO_MIN})"
    )

    # --- `valor_norm` es None SIEMPRE: es una categoría, no una adherencia ---
    assert e.valor_norm is None, (
        f"esquema_cromatico.valor_norm={e.valor_norm}: puntuar una categoría mezclaría "
        "adherencia con aplicabilidad, que es lo que MetricaConfianza separa"
    )

    # --- Coherencia interna de la evidencia cromática ---
    assert informe.n_matices_dominantes == len(informe.matices_dominantes), (
        "n_matices_dominantes debe ser la longitud de matices_dominantes"
    )
    for matiz in informe.matices_dominantes:
        assert 0.0 <= matiz < 360.0, f"matiz {matiz} fuera de [0,360)"
        assert isclose(matiz, round(matiz, DECIMALES_MATIZ), abs_tol=TOL), (
            f"matiz {matiz} sin redondear a {DECIMALES_MATIZ} decimal(es): un campo citable "
            "tiene que ser bit-reproducible por un tercero"
        )

    if not abre:
        # Con el gate cerrado se suprime la evidencia que no significaría nada, igual
        # que `coord_punto_fuga = None` en el Agente 2. Los valores neutros tienen que
        # ser COHERENTES con sus propias definiciones: lista vacía, luego ningún par
        # que medir, luego spread 0 y n 0. (Lección del `valor_norm` del Agente 2: el
        # valor neutro de una métrica no medida no puede ser un número que su propia
        # fórmula contradiga.)
        assert e.valor == ESQUEMA_SIN_COLOR, (
            f"con el gate cerrado el esquema debe ser '{ESQUEMA_SIN_COLOR}', es '{e.valor}'"
        )
        assert informe.matices_dominantes == [], "con el gate cerrado no se publican matices"
        assert informe.n_matices_dominantes == 0
        assert informe.spread_cromatico == 0.0
    else:
        assert e.valor in ETIQUETAS_VALIDAS, (
            f"'{e.valor}' no es una etiqueta de la taxonomía; '{ESQUEMA_SIN_COLOR}' está "
            "reservada al gate cerrado y no es una quinta categoría"
        )
        assert informe.n_matices_dominantes >= 1, (
            "con el gate abierto siempre hay al menos un matiz dominante: las masas suman 1 "
            f"sobre 5 grupos como mucho, así que la mayor no puede bajar de {PESO_MIN_MATIZ}"
        )
        # `spread_cromatico` se recomputa desde los matices publicados: es la evidencia
        # sobre la que se decide la etiqueta, y tiene que cuadrar con ella.
        centros = np.array(informe.matices_dominantes)
        esperado_spread = (
            float(dif_matiz(centros[:, None], centros[None, :]).max()) if len(centros) > 1 else 0.0
        )
        assert isclose(informe.spread_cromatico, esperado_spread, abs_tol=0.05), (
            f"spread_cromatico={informe.spread_cromatico} no es la mayor distancia entre los "
            f"matices publicados ({esperado_spread})"
        )
        # Y la ETIQUETA se recomputa desde el spread: la taxonomía es determinista.
        assert e.valor == clasificar_esquema(centros, informe.spread_cromatico), (
            f"la etiqueta '{e.valor}' no se deduce de spread={informe.spread_cromatico:.1f} "
            "aplicando las plantillas de la taxonomía"
        )
        # Dos matices distintos están al menos a UMBRAL_FUSION_MATIZ: por debajo se
        # habrían fusionado, así que publicarlos por separado sería partir un matiz.
        if len(centros) > 1:
            pares = dif_matiz(centros[:, None], centros[None, :])
            np.fill_diagonal(pares, 999.0)
            assert pares.min() >= UMBRAL_FUSION_MATIZ - 0.05, (
                f"hay dos matices publicados a {pares.min():.1f}°, por debajo del umbral de "
                f"fusión ({UMBRAL_FUSION_MATIZ}°): son rodajas del mismo matiz"
            )
        assert (informe.spread_cromatico > 0.0) == (informe.n_matices_dominantes > 1), (
            "spread solo puede ser 0 cuando hay un único matiz dominante"
        )

    # --- Contrato uniforme de confianza ---
    for nombre, metrica in (("media_L", m), ("esquema_cromatico", e)):
        assert metrica.fuente_confianza, f"{nombre} no explica su confianza en fuente_confianza"

    assert informe.dimensiones_imagen[0] > 0 and informe.dimensiones_imagen[1] > 0
    assert_fichero_valido(informe.verificacion_path, "La imagen de verificación visual")


def casos_analiticos():
    """Ground truth exacto sobre lienzos sintéticos."""
    fallos = []
    generados = []
    n_casos = 0

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def correr(nombre, img):
            ruta = tmp / f"{nombre}.png"
            cv2.imwrite(str(ruta), img)
            informe = luz_tono_engine.analizar(str(ruta))
            generados.append(Path(informe.verificacion_path))
            return informe

        # --- Casos 1-4: las cuatro etiquetas de la taxonomía, con matices exactos ---
        # (nombre, matices dibujados, etiqueta esperada, spread esperado, por qué)
        casos_paleta = [
            ("monocromatico", [30.0], "monocromático", 0.0,
             "un solo matiz: no hay par que medir"),
            ("analogo", [20.0, 60.0], "análogo", 40.0,
             f"dos matices vecinos, spread entre {SPREAD_MONOCROMATICO} y {SPREAD_ANALOGO}"),
            ("complementario", [30.0, 210.0], "complementario", 180.0,
             "par opuesto exacto en el círculo de color"),
            ("otro", [0.0, 100.0], "otro", 100.0,
             "spread amplio pero sin ningún par a 180°+-30"),
        ]
        for nombre, matices, etiqueta, spread, motivo in casos_paleta:
            informe = correr(f"analitico_{nombre}", lienzo_hsv(matices))
            n_casos += 1
            try:
                validar_informe(informe)
                e = informe.esquema_cromatico
                assert e.confianza == 1.0, (
                    f"un lienzo de matices puros debe abrir el gate: {e.fuente_confianza}"
                )
                assert e.valor == etiqueta, (
                    f"etiqueta '{e.valor}', se esperaba '{etiqueta}' ({motivo})"
                )
                assert informe.n_matices_dominantes == len(matices), (
                    f"se pintaron {len(matices)} matices y se publicaron "
                    f"{informe.n_matices_dominantes}: {informe.matices_dominantes}"
                )
                # Cada matiz dibujado tiene que aparecer entre los medidos.
                for esperado in matices:
                    assert any(
                        dif_matiz(np.array(medido), np.array(esperado)) <= TOL_MATIZ
                        for medido in informe.matices_dominantes
                    ), f"el matiz {esperado}° no aparece en {informe.matices_dominantes}"
                assert isclose(informe.spread_cromatico, spread, abs_tol=2 * TOL_MATIZ), (
                    f"spread={informe.spread_cromatico:.1f}, se dibujó {spread}"
                )
                # Un matiz puro está saturado al máximo: no debe filtrarse nada.
                assert informe.ratio_pixeles_cromaticos > 0.99, (
                    f"ratio={informe.ratio_pixeles_cromaticos:.4f}: matices puros no deberían "
                    "caer en el filtro acromático"
                )
                print(
                    f"[OK] analitico/{nombre}: {informe.matices_dominantes} "
                    f"spread={informe.spread_cromatico:.1f} -> {e.valor}"
                )
            except AssertionError as err:
                fallos.append((f"analitico/{nombre}", str(err)))
                print(f"[FAIL] analitico/{nombre}: {err}")

        # --- Caso 5: gris puro, el lado CERRADO del gate ---
        # Es la distinción que esta métrica existe para preservar: un acromático no
        # es un "monocromático", es una imagen sin esquema cromático medible.
        informe = correr("analitico_gris", lienzo_gris(128))
        n_casos += 1
        try:
            validar_informe(informe)
            e = informe.esquema_cromatico
            assert informe.ratio_pixeles_cromaticos == 0.0, (
                f"un gris puro no tiene ningún píxel cromático, ratio={informe.ratio_pixeles_cromaticos}"
            )
            assert e.confianza == 0.0 and e.valor == ESQUEMA_SIN_COLOR, (
                f"el gate debe cerrarse sobre un acromático: conf={e.confianza}, valor='{e.valor}'"
            )
            # Y sin embargo la exposición SÍ se mide: es la asimetría del agente.
            assert informe.media_L.confianza == 1.0
            # Tolerancia y no igualdad exacta: `np.std` sobre un lienzo uniforme deja
            # un residuo del orden de 1e-15 al restar la media en coma flotante. Es
            # ruido numérico, no contraste.
            assert isclose(informe.std_L, 0.0, abs_tol=1e-9), (
                f"un lienzo uniforme no tiene contraste, std_L={informe.std_L}"
            )
            print(
                f"[OK] analitico/gris: ratio=0.0000 gate cerrado, pero media_L="
                f"{informe.media_L.valor:.2f} con confianza 1.0 (asimetría del agente)"
            )
        except AssertionError as err:
            fallos.append(("analitico/gris", str(err)))
            print(f"[FAIL] analitico/gris: {err}")

        # --- Casos 6-8: exposición con ground truth que no depende de la conversión ---
        # (nombre, lienzo, media esperada, std esperada, clip sombras, clip luces)
        casos_expo = [
            ("negro", lienzo_gris(0), 0.0, 0.0, 100.0, 0.0),
            ("blanco", lienzo_gris(255), 100.0, 0.0, 0.0, 100.0),
            ("mitades", lienzo_medio_negro_medio_blanco(), 50.0, 50.0, 50.0, 50.0),
        ]
        for nombre, img, media, std, clip_s, clip_l in casos_expo:
            informe = correr(f"analitico_expo_{nombre}", img)
            n_casos += 1
            try:
                validar_informe(informe)
                m = informe.media_L
                assert isclose(m.valor, media, abs_tol=0.01), (
                    f"media_L={m.valor:.4f}, se esperaba {media} por construcción"
                )
                assert isclose(informe.std_L, std, abs_tol=0.01), (
                    f"std_L={informe.std_L:.4f}, se esperaba {std}"
                )
                assert isclose(informe.pct_clipping_sombras, clip_s, abs_tol=0.01)
                assert isclose(informe.pct_clipping_luces, clip_l, abs_tol=0.01)
                print(
                    f"[OK] analitico/expo_{nombre}: media_L={m.valor:.2f} std={informe.std_L:.2f} "
                    f"clip {informe.pct_clipping_sombras:.0f}/{informe.pct_clipping_luces:.0f}% "
                    f"adh={m.valor_norm:.2f}"
                )
            except AssertionError as err:
                fallos.append((f"analitico/expo_{nombre}", str(err)))
                print(f"[FAIL] analitico/expo_{nombre}: {err}")

    # Los PNG de verificación de las sintéticas no aportan nada en outputs/, que es
    # material de revisión real.
    for ruta in generados:
        ruta.unlink(missing_ok=True)

    return n_casos, fallos


def integracion_sobre_data():
    """El engine completo sobre el corpus real, con la comprobación de determinismo."""
    imagenes = sorted(DATA_DIR.glob("imagen*.jpg"))
    assert imagenes, f"No se encontraron imágenes de prueba en {DATA_DIR}"

    fallos = []
    abren = 0
    reparto = {}
    medias_l = []

    for image_path in imagenes:
        informe = luz_tono_engine.analizar(str(image_path))
        try:
            validar_informe(informe)

            # DETERMINISMO: analizar dos veces tiene que dar exactamente lo mismo.
            # No es una formalidad ni una comprobación de "que no crashee": k-means no
            # es bit-exacto entre ejecuciones aunque se fije la semilla, y por eso el
            # matiz se publica redondeado. Si esto falla, `matices_dominantes` no es
            # recomputable por un tercero y la premisa del sistema se cae aquí.
            repetido = luz_tono_engine.analizar(str(image_path))
            assert repetido.model_dump() == informe.model_dump(), (
                "dos análisis de la misma imagen dan informes distintos: el redondeo del "
                "matiz no está absorbiendo el no-determinismo de k-means"
            )

            e = informe.esquema_cromatico
            if e.confianza >= 1.0:
                abren += 1
            reparto[e.valor] = reparto.get(e.valor, 0) + 1
            medias_l.append(informe.media_L.valor)
            print(
                f"[OK] {image_path.name}: media_L={informe.media_L.valor:6.2f} "
                f"adh={informe.media_L.valor_norm:.2f} std={informe.std_L:5.2f} | "
                f"ratio={informe.ratio_pixeles_cromaticos:.3f} n={informe.n_matices_dominantes} "
                f"spread={informe.spread_cromatico:6.1f} conf={e.confianza:.0f} -> {e.valor}"
            )
        except AssertionError as err:
            fallos.append((image_path.name, str(err)))
            print(f"[FAIL] {image_path.name}: {err}")

    return len(imagenes), fallos, abren, reparto, medias_l


def main():
    print("--- Casos analíticos (ground truth exacto) ---")
    n_analiticos, fallos_analiticos = casos_analiticos()

    print("\n--- Integración sobre data/ (incluye determinismo) ---")
    n_imagenes, fallos_data, abren, reparto, medias_l = integracion_sobre_data()

    fallos = fallos_analiticos + fallos_data
    print("\n--- Resumen ---")
    print(f"{n_analiticos - len(fallos_analiticos)}/{n_analiticos} casos analíticos OK")
    print(f"{n_imagenes - len(fallos_data)}/{n_imagenes} imágenes de data/ OK")
    print(f"Gate cromático ABIERTO en {abren}/{n_imagenes}")
    print(f"Reparto de esquemas: {dict(sorted(reparto.items(), key=lambda kv: -kv[1]))}")
    if medias_l:
        print(
            f"media_L sobre el corpus: min={min(medias_l):.2f} max={max(medias_l):.2f} "
            f"media={sum(medias_l) / len(medias_l):.2f}"
        )
    if fallos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
