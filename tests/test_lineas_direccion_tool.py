"""Prueba manual de LineasDireccionTool."""
from math import isclose, hypot
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from tfg_multiagente_fotografia.tools.lineas_direccion_tool import (
    ANGULO_MAX_HORIZONTE,
    APERTURA_MIN_GRADOS,
    LADO_MAYOR_TRABAJO,
    LONGITUD_MIN_HORIZONTE,
    MIN_LINEAS_FUGA,
    UMBRAL_HORIZONTE_GRADOS,
    lineas_direccion_engine,
    score_nulo,
    tabla_nulo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Lienzo sintético con el lado mayor ya en el espacio de trabajo, para que el engine
# NO reescale (escala = 1.0) y las coordenadas que se dibujan sean las mismas que
# procesa Hough. Sin esto el ground truth se desplazaría por el redondeo del resize.
ANCHO, ALTO = LADO_MAYOR_TRABAJO, 683
DIAGONAL = hypot(ANCHO, ALTO)

# Tolerancias. La angular es generosa a propósito: HoughLinesP cuantiza theta a 1°,
# así que el ángulo medido no puede ser más preciso que eso. La tolerancia de
# posición del punto de fuga se deriva de la
# angular: un error de 1° en un rayo de ~400 px desplaza el cruce unos 7 px.
TOL_GRADOS = 1.0
TOL_PF_NORM = 0.03
TOL = 1e-9


def assert_fichero_valido(ruta_str, descripcion):
    ruta = Path(ruta_str)
    assert ruta.exists(), f"{descripcion} no se guardó en disco: {ruta}"
    assert ruta.stat().st_size > 0, f"{descripcion} está vacío: {ruta}"


def lienzo():
    """Fondo negro uniforme: sin bordes, luego sin segmentos espurios."""
    return np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)


def trazar(img, p1, p2, grosor=3):
    """Línea blanca. El grosor da dos bordes paralelos (Canny detecta cada lado),
    lo cual es realista y no estorba: ambos tienen la misma orientación."""
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255), grosor)


def rayo_desde(punto, grados, r_min, r_max):
    """Segmento contenido en la recta que pasa por `punto` con esa inclinación.

    Se empieza a `r_min` del punto y no en él, para que los rayos no se toquen
    entre sí: si convergieran en un cruce dibujado, Canny lo vería como una mancha
    y el consenso quedaría ayudado por un artefacto del dibujo en vez de por la
    geometría, que es justo lo que se quiere comprobar.
    """
    rad = np.radians(grados)
    dx, dy = np.cos(rad), np.sin(rad)
    return (
        (punto[0] + dx * r_min, punto[1] + dy * r_min),
        (punto[0] + dx * r_max, punto[1] + dy * r_max),
    )


def validar_informe(informe):
    """Invariantes que deben cumplirse en CUALQUIER imagen, sintética o real."""
    a = informe.angulo_horizonte
    s = informe.score_convergencia

    # --- Recuentos y partición angular ---
    assert informe.n_lineas_total >= informe.n_lineas_horizonte + informe.n_lineas_fuga, (
        f"n_lineas_total={informe.n_lineas_total} menor que horizonte+fuga="
        f"{informe.n_lineas_horizonte + informe.n_lineas_fuga}: las franjas deben ser excluyentes"
    )
    assert informe.n_lineas_inliers <= informe.n_lineas_fuga, (
        "no pueden votar más líneas de las que hay"
    )

    # --- score = n_inliers / n_fuga: la fracción publicada en su forma cruda ---
    esperado = informe.n_lineas_inliers / informe.n_lineas_fuga if informe.n_lineas_fuga else 0.0
    assert isclose(s.valor, esperado, abs_tol=TOL), (
        f"score_convergencia={s.valor} no es n_inliers/n_fuga="
        f"{informe.n_lineas_inliers}/{informe.n_lineas_fuga}={esperado}"
    )
    assert s.valor_norm == s.valor, (
        "score_convergencia ya nace normalizada y orientada: valor_norm debe ser igual a valor"
    )
    assert 0.0 <= informe.apertura_haz <= 90.0, f"apertura_haz={informe.apertura_haz} fuera de [0,90]"

    # --- El ángulo respeta la partición, y su adherencia se deriva de la cota ---
    assert abs(a.valor) <= UMBRAL_HORIZONTE_GRADOS, (
        f"angulo_horizonte={a.valor} fuera del cono de {UMBRAL_HORIZONTE_GRADOS}°: "
        "el candidato no debería haber entrado en la franja de horizonte"
    )
    # `valor_norm` existe si y solo si hubo algo que medir. Sin candidatos se emite
    # None y NO un número: cualquier constante ahí (0.0 ó 1.0) sería recomputable a
    # partir del `valor` publicado y daría otro resultado, es decir, el informe se
    # contradiría a sí mismo. Con candidatos, tiene que ser exactamente la fórmula.
    if informe.n_lineas_horizonte == 0:
        assert a.valor_norm is None, (
            f"sin candidatos no hay adherencia que publicar, pero valor_norm={a.valor_norm}"
        )
    else:
        esperado_norm = min(1.0, max(0.0, 1.0 - abs(a.valor) / UMBRAL_HORIZONTE_GRADOS))
        assert a.valor_norm is not None, "con candidatos, angulo_horizonte debe emitir valor_norm"
        assert isclose(a.valor_norm, esperado_norm, abs_tol=TOL), (
            f"angulo_horizonte.valor_norm={a.valor_norm} no es 1 - |valor|/{UMBRAL_HORIZONTE_GRADOS}"
        )
    assert informe.longitud_horizonte >= 0.0
    assert (informe.longitud_horizonte == 0.0) == (informe.n_lineas_horizonte == 0), (
        "longitud_horizonte solo vale 0 cuando no hay ningún candidato"
    )

    # --- LOS DOS GATES SON REPRODUCIBLES desde los campos publicados ---
    # Es la propiedad que sostiene la premisa del sistema: cualquiera puede
    # recomputar por qué el agente afirmó o se calló, sin leer el código.
    gate_h = (
        informe.n_lineas_horizonte > 0
        and informe.longitud_horizonte >= LONGITUD_MIN_HORIZONTE
        and abs(a.valor) <= ANGULO_MAX_HORIZONTE
    )
    assert a.confianza == (1.0 if gate_h else 0.0), (
        f"angulo_horizonte.confianza={a.confianza} no coincide con recomputar sus tres "
        f"condiciones (n={informe.n_lineas_horizonte}, long={informe.longitud_horizonte:.3f}, "
        f"ang={a.valor:+.2f})"
    )
    gate_c = (
        informe.n_lineas_fuga >= MIN_LINEAS_FUGA
        and s.valor > informe.score_convergencia_nulo
        and informe.apertura_haz >= APERTURA_MIN_GRADOS
    )
    assert s.confianza == (1.0 if gate_c else 0.0), (
        f"score_convergencia.confianza={s.confianza} no coincide con recomputar sus tres "
        f"condiciones (n={informe.n_lineas_fuga}, score={s.valor:.3f}, "
        f"nulo={informe.score_convergencia_nulo:.3f}, apertura={informe.apertura_haz:.1f})"
    )

    # --- El nulo publicado es el que corresponde a este n ---
    assert isclose(
        informe.score_convergencia_nulo, score_nulo(tabla_nulo(), informe.n_lineas_fuga), abs_tol=TOL
    ), "score_convergencia_nulo no corresponde a n_lineas_fuga en la tabla"

    # --- El punto de fuga existe si y solo si su gate lo avala ---
    assert (informe.coord_punto_fuga is not None) == (s.confianza >= 1.0), (
        "coord_punto_fuga debe publicarse si y solo si el gate de convergencia está abierto: "
        "una coordenada sin significado invita a citarla como si lo tuviera"
    )

    # --- Contrato uniforme de confianza ---
    for nombre, m in (("angulo_horizonte", a), ("score_convergencia", s)):
        assert m.fuente_confianza, f"{nombre} no explica su confianza en fuente_confianza"

    assert_fichero_valido(informe.verificacion_path, "La imagen de verificación visual")


def casos_analiticos():
    """Ground truth exacto sobre imágenes sintéticas."""
    # Punto de fuga deliberadamente descentrado: (0.5, 0.5) sería un valor "bonito"
    # que podría coincidir por accidente con un fallo (p. ej. devolver el centro).
    pf = (700.0, 250.0)
    pf_norm = (pf[0] / ANCHO, pf[1] / ALTO)

    fallos = []
    generados = []
    n_casos = 0

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def correr(nombre, img):
            ruta = tmp / f"{nombre}.png"
            cv2.imwrite(str(ruta), img)
            informe = lineas_direccion_engine.analizar(str(ruta))
            generados.append(Path(informe.verificacion_path))
            return informe

        # --- Caso 1: punto de fuga conocido ---
        # Cuatro rayos que salen del mismo punto con ángulos repartidos dentro de la
        # banda de fuga. El consenso tiene que recuperar el punto del que salieron.
        img = lienzo()
        for grados in (25.0, 55.0, -25.0, -55.0):
            p1, p2 = rayo_desde(pf, grados, 70, 430)
            trazar(img, p1, p2)
        casos_fuga = correr("analitico_fuga", img)
        n_casos += 1
        try:
            validar_informe(casos_fuga)
            assert casos_fuga.score_convergencia.confianza == 1.0, (
                f"cuatro rayos concurrentes deberían abrir el gate: "
                f"{casos_fuga.score_convergencia.fuente_confianza}"
            )
            x, y = casos_fuga.coord_punto_fuga
            assert isclose(x, pf_norm[0], abs_tol=TOL_PF_NORM) and isclose(
                y, pf_norm[1], abs_tol=TOL_PF_NORM
            ), f"punto de fuga recuperado ({x:.4f}, {y:.4f}), se dibujó en ({pf_norm[0]:.4f}, {pf_norm[1]:.4f})"
            # Los rayos abarcan 110° de abanico, pero la distancia entre
            # ORIENTACIONES está acotada en 90 por ser rectas sin sentido.
            assert casos_fuga.apertura_haz >= APERTURA_MIN_GRADOS, (
                f"apertura={casos_fuga.apertura_haz:.1f}°, se esperaba un abanico amplio"
            )
            assert casos_fuga.angulo_horizonte.confianza == 0.0, (
                "no se dibujó ninguna línea casi horizontal: el gate del horizonte debe cerrarse"
            )
            print(
                f"[OK] analitico/punto_fuga: PF=({x:.4f}, {y:.4f}) esperado "
                f"({pf_norm[0]:.4f}, {pf_norm[1]:.4f})  score={casos_fuga.score_convergencia.valor:.3f} "
                f"apertura={casos_fuga.apertura_haz:.1f}"
            )
        except AssertionError as err:
            fallos.append(("analitico/punto_fuga", str(err)))
            print(f"[FAIL] analitico/punto_fuga: {err}")

        # --- Casos 2-4: el horizonte, con su ángulo y su longitud conocidos ---
        # (nombre, grados, media longitud del trazo, confianza esperada, por qué)
        casos_horizonte = [
            ("nivelado", 5.0, 460, 1.0, "largo y poco inclinado: es un horizonte"),
            ("corto", 0.0, 60, 0.0, "fragmento: no llega a la longitud mínima"),
            ("inclinado", 12.0, 460, 0.0, "largo pero es estructura de la escena"),
        ]
        for nombre, grados, semi, conf_esperada, motivo in casos_horizonte:
            img = lienzo()
            p1, p2 = rayo_desde((ANCHO / 2, ALTO / 2), grados, -semi, semi)
            trazar(img, p1, p2)
            informe = correr(f"analitico_horizonte_{nombre}", img)
            n_casos += 1
            try:
                validar_informe(informe)
                a = informe.angulo_horizonte
                assert isclose(a.valor, grados, abs_tol=TOL_GRADOS), (
                    f"ángulo medido {a.valor:+.2f}°, se dibujó a {grados:+.2f}°"
                )
                assert a.confianza == conf_esperada, (
                    f"confianza={a.confianza}, se esperaba {conf_esperada} ({motivo}). "
                    f"fuente: {a.fuente_confianza}"
                )
                # La longitud publicada debe corresponder al trazo dibujado. Hough
                # puede recortar los extremos, así que se compara con holgura.
                largo_esperado = 2 * semi / DIAGONAL
                assert isclose(informe.longitud_horizonte, largo_esperado, rel_tol=0.15), (
                    f"longitud={informe.longitud_horizonte:.4f} de diagonal, se dibujó "
                    f"{largo_esperado:.4f}"
                )
                print(
                    f"[OK] analitico/horizonte_{nombre}: ang={a.valor:+.2f}° (dibujado "
                    f"{grados:+.1f}) long={informe.longitud_horizonte:.4f} adh={a.valor_norm:.2f} "
                    f"conf={a.confianza:.0f}"
                )
            except AssertionError as err:
                fallos.append((f"analitico/horizonte_{nombre}", str(err)))
                print(f"[FAIL] analitico/horizonte_{nombre}: {err}")

        # --- Caso 5: escena sin ninguna estructura lineal ---
        # Cierra los DOS gates a la vez, y es en sí mismo un hallazgo compositivo.
        informe = correr("analitico_vacio", lienzo())
        n_casos += 1
        try:
            validar_informe(informe)
            assert informe.n_lineas_total == 0, (
                f"un lienzo uniforme no debería dar segmentos, dio {informe.n_lineas_total}"
            )
            assert informe.angulo_horizonte.confianza == 0.0
            assert informe.score_convergencia.confianza == 0.0
            assert informe.coord_punto_fuga is None
            assert informe.angulo_horizonte.valor == 0.0 and informe.longitud_horizonte == 0.0, (
                "sin candidatos, la métrica debe emitirse con valores neutros y no desaparecer"
            )
            print("[OK] analitico/vacio: 0 segmentos, los dos gates cerrados, métricas neutras")
        except AssertionError as err:
            fallos.append(("analitico/vacio", str(err)))
            print(f"[FAIL] analitico/vacio: {err}")

    # Los PNG de verificación de las sintéticas no aportan nada en outputs/, que es
    # material de revisión real.
    for ruta in generados:
        ruta.unlink(missing_ok=True)

    return n_casos, fallos


def integracion_sobre_data():
    """El engine completo sobre el corpus real."""
    imagenes = sorted(DATA_DIR.glob("imagen*.jpg"))
    assert imagenes, f"No se encontraron imágenes de prueba en {DATA_DIR}"

    fallos = []
    abren_horizonte = 0
    abren_convergencia = 0
    # Rango empírico de la longitud, que es lo que hay que volcar al schema (Fase 6)
    # y el insumo para recalibrar el umbral cuando crezca data/.
    longitudes = []

    for image_path in imagenes:
        informe = lineas_direccion_engine.analizar(str(image_path))
        try:
            validar_informe(informe)
            a, s = informe.angulo_horizonte, informe.score_convergencia
            if a.confianza >= 1.0:
                abren_horizonte += 1
                longitudes.append(informe.longitud_horizonte)
            if s.confianza >= 1.0:
                abren_convergencia += 1
            print(
                f"[OK] {image_path.name}: ang={a.valor:+6.2f}° long={informe.longitud_horizonte:.3f} "
                f"conf={a.confianza:.0f} | score={s.valor:.3f} nulo={informe.score_convergencia_nulo:.3f} "
                f"apert={informe.apertura_haz:5.1f} conf={s.confianza:.0f}"
            )
        except AssertionError as err:
            fallos.append((image_path.name, str(err)))
            print(f"[FAIL] {image_path.name}: {err}")

    return len(imagenes), fallos, abren_horizonte, abren_convergencia, longitudes


def main():
    print("--- Casos analíticos (ground truth exacto) ---")
    n_analiticos, fallos_analiticos = casos_analiticos()

    print("\n--- Integración sobre data/ ---")
    n_imagenes, fallos_data, abren_h, abren_c, longitudes = integracion_sobre_data()

    fallos = fallos_analiticos + fallos_data
    print("\n--- Resumen ---")
    print(f"{n_analiticos - len(fallos_analiticos)}/{n_analiticos} casos analíticos OK")
    print(f"{n_imagenes - len(fallos_data)}/{n_imagenes} imágenes de data/ OK")
    print(f"Gate del horizonte ABIERTO en {abren_h}/{n_imagenes}")
    print(f"Gate de convergencia ABIERTO en {abren_c}/{n_imagenes}")
    if longitudes:
        print(
            f"longitud_horizonte de las que abren: min={min(longitudes):.4f} "
            f"max={max(longitudes):.4f} media={sum(longitudes) / len(longitudes):.4f}"
        )
    if fallos:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
