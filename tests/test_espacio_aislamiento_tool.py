"""Prueba manual de EspacioAislamientoTool."""
from pathlib import Path
import sys

import cv2
import numpy as np

from tfg_multiagente_fotografia.tools.espacio_aislamiento_tool import (
    FUENTE_CON_IDENTIDAD,
    VAR_LAPLACIANO_MIN,
    espacio_aislamiento_engine as engine,
)
from tfg_multiagente_fotografia.tools.shared_perception_tool import SharedPerceptionTool

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "data"
DIR_TMP = RAIZ / "outputs" / "_tmp_test_espacio"

TOL = 1e-9        # igualdad de flotantes recomputados
TOL_AREA = 1e-6   # el área es exacta salvo redondeo de conteo entero

fallos = []


def check(condicion, mensaje):
    if not condicion:
        fallos.append(mensaje)
    return condicion


# =============================================================================
# A) Casos analíticos
# =============================================================================

def _lienzo_con_figura(ancho, alto, rect, textura_figura, textura_fondo, semilla=7):
    """Imagen BGR con dos regiones de textura controlada, y su mapa de saliencia.

    `textura_*` es la amplitud del ruido de cada región: 0 = superficie perfectamente lisa.
    Se usa ruido y no un degradado porque la varianza del Laplaciano mide densidad de detalle,
    y el ruido la produce de forma controlable y con la misma media de gris en las dos
    regiones (así la diferencia medida es de DETALLE y no de luminosidad).
    """
    rng = np.random.default_rng(semilla)
    img = np.full((alto, ancho), 128, np.int16)
    x, y, w, h = rect

    if textura_fondo > 0:
        img += rng.integers(-textura_fondo, textura_fondo + 1, (alto, ancho), dtype=np.int16)
    if textura_figura > 0:
        img[y:y + h, x:x + w] = 128 + rng.integers(
            -textura_figura, textura_figura + 1, (h, w), dtype=np.int16
        )
    else:
        img[y:y + h, x:x + w] = 128

    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Mapa de saliencia binario: Otsu lo separa exactamente, así que la máscara resultante es
    # el rectángulo al píxel y el área es ground truth.
    mapa = np.zeros((alto, ancho), np.uint8)
    mapa[y:y + h, x:x + w] = 255
    return bgr, mapa


def _analizar_sintetico(nombre, bgr, mapa, fuente):
    """Escribe los dos ficheros que el engine espera y lo ejecuta por la vía de saliencia."""
    DIR_TMP.mkdir(parents=True, exist_ok=True)
    ruta_img = DIR_TMP / f"{nombre}.png"
    ruta_mapa = DIR_TMP / f"{nombre}_saliency.png"
    cv2.imwrite(str(ruta_img), bgr)
    cv2.imwrite(str(ruta_mapa), mapa)
    # bbox irrelevante en esta vía (sujeto_discreto=False no llama a GrabCut), pero el contrato
    # de entrada lo exige, así que se pasa uno coherente con el rectángulo.
    alto, ancho = bgr.shape[:2]
    bbox = [ancho / 2, alto / 2, ancho / 4, alto / 4]
    return engine.analizar(str(ruta_img), bbox, False, fuente, str(ruta_mapa))


def casos_analiticos():
    print("=== A) Casos analiticos (ground truth exacto) ===")

    # --- A1: área exacta del espacio negativo ---
    # Cuadrado de 300x300 en un lienzo de 900x600: 90.000 de 540.000 px = 1/6 de cobertura,
    # luego espacio negativo = 5/6. El lienzo no llega a 1024, así que no hay reescalado que
    # introduzca redondeo.
    bgr, mapa = _lienzo_con_figura(900, 600, (100, 100, 300, 300), 40, 40)
    inf = _analizar_sintetico("a1_area", bgr, mapa, FUENTE_CON_IDENTIDAD)
    esperado = 1 - (300 * 300) / (900 * 600)
    check(abs(inf.ratio_espacio_negativo.valor - esperado) < TOL_AREA,
          f"A1 espacio negativo: {inf.ratio_espacio_negativo.valor} != {esperado}")
    check(inf.fuente_mascara == "prior_saliencia",
          f"A1 fuente_mascara deberia ser prior_saliencia, es {inf.fuente_mascara}")
    print(f"  A1 area exacta ................ esp_neg={inf.ratio_espacio_negativo.valor:.6f} "
          f"(esperado {esperado:.6f})")

    # --- A2: `valor_norm` del espacio negativo es None SIEMPRE ---
    # No es un olvido: la métrica no nombra ningún patrón al que adherirse, así que emitir un
    # número ahí sería fabricar un juicio de valor.
    check(inf.ratio_espacio_negativo.valor_norm is None,
          "A2 ratio_espacio_negativo.valor_norm deberia ser None siempre")
    print("  A2 valor_norm del espacio ..... None (correcto: no hay patron de adherencia)")

    # --- A3: figura texturada sobre fondo liso -> indice muy positivo ---
    bgr, mapa = _lienzo_con_figura(900, 600, (300, 200, 300, 200), 60, 0)
    inf = _analizar_sintetico("a3_fig_texturada", bgr, mapa, FUENTE_CON_IDENTIDAD)
    check(inf.ratio_nitidez.valor > 0.9,
          f"A3 figura texturada sobre fondo liso deberia dar indice > 0.9, da {inf.ratio_nitidez.valor}")
    check(inf.var_laplaciano_figura > inf.var_laplaciano_fondo,
          "A3 var_figura deberia superar a var_fondo")
    print(f"  A3 figura texturada ........... nitidez={inf.ratio_nitidez.valor:+.4f}")

    # --- A4: el caso simétrico, fondo texturado -> indice muy negativo ---
    # Es el que documenta que el signo negativo es un hecho legitimo y no un error.
    bgr, mapa = _lienzo_con_figura(900, 600, (300, 200, 300, 200), 0, 60)
    inf = _analizar_sintetico("a4_fondo_texturado", bgr, mapa, FUENTE_CON_IDENTIDAD)
    check(inf.ratio_nitidez.valor < -0.9,
          f"A4 fondo texturado deberia dar indice < -0.9, da {inf.ratio_nitidez.valor}")
    print(f"  A4 fondo texturado ............ nitidez={inf.ratio_nitidez.valor:+.4f}")

    # --- A5: misma textura en ambas regiones -> indice ~ 0 ---
    bgr, mapa = _lienzo_con_figura(900, 600, (300, 200, 300, 200), 50, 50)
    inf = _analizar_sintetico("a5_simetrico", bgr, mapa, FUENTE_CON_IDENTIDAD)
    check(abs(inf.ratio_nitidez.valor) < 0.1,
          f"A5 misma textura deberia dar indice ~0, da {inf.ratio_nitidez.valor}")
    # Y su valor_norm debe caer en el 0.5 que la convencion asigna a "ninguna separacion".
    check(abs(inf.ratio_nitidez.valor_norm - 0.5) < 0.05,
          f"A5 valor_norm deberia rondar 0.5, es {inf.ratio_nitidez.valor_norm}")
    print(f"  A5 textura simetrica .......... nitidez={inf.ratio_nitidez.valor:+.4f} "
          f"norm={inf.ratio_nitidez.valor_norm:.4f}")

    # --- A6: lienzo uniforme -> no hay medicion -> valor_norm None ---
    # Es la leccion del `valor_norm` del Agente 2: el valor neutro de algo no medido no puede
    # ser un numero que su propia formula contradiga. Aqui el 0.0 del indice es una convencion
    # (0/0), no una medida, asi que la adherencia no se publica.
    bgr, mapa = _lienzo_con_figura(900, 600, (300, 200, 300, 200), 0, 0)
    inf = _analizar_sintetico("a6_uniforme", bgr, mapa, FUENTE_CON_IDENTIDAD)
    check(inf.var_laplaciano_figura == 0.0 and inf.var_laplaciano_fondo == 0.0,
          "A6 un lienzo uniforme deberia dar las dos varianzas a 0")
    check(inf.ratio_nitidez.valor_norm is None,
          "A6 sin medicion, ratio_nitidez.valor_norm deberia ser None")
    check(inf.ratio_nitidez.confianza == 0.0,
          "A6 sin textura, el gate de nitidez tiene que cerrarse")
    print("  A6 lienzo uniforme ............ var=0/0, valor_norm=None, gate cerrado")

    # --- A7: el gate de identidad, los dos lados ---
    # Misma imagen, misma mascara, mismas cifras: lo unico que cambia es de donde salio el
    # bbox. Es la comprobacion de que el gate mira la PROCEDENCIA y no la medida.
    bgr, mapa = _lienzo_con_figura(900, 600, (300, 200, 300, 200), 60, 10)
    con_yolo = _analizar_sintetico("a7_yolo", bgr, mapa, FUENTE_CON_IDENTIDAD)
    sin_yolo = _analizar_sintetico("a7_saliencia", bgr, mapa, "saliencia")
    check(con_yolo.ratio_espacio_negativo.confianza == 1.0,
          "A7 con fuente=yolo el gate de espacio deberia abrir")
    check(sin_yolo.ratio_espacio_negativo.confianza == 0.0,
          "A7 con fuente=saliencia el gate de espacio deberia cerrar")
    check(abs(con_yolo.ratio_nitidez.valor - sin_yolo.ratio_nitidez.valor) < TOL,
          "A7 el VALOR medido no debe depender del gate, solo la confianza")
    check(sin_yolo.ratio_espacio_negativo.fuente_confianza is not None,
          "A7 un gate cerrado tiene que explicar por que en fuente_confianza")
    print("  A7 gate de identidad .......... yolo=1.0 / saliencia=0.0, mismo valor medido")

    # --- A8: los dos gates son INDEPENDIENTES ---
    # Textura suficiente pero sin identidad: nitidez cierra por la condicion heredada, no por
    # la suya. Y al reves no puede pasar (nitidez hereda la de espacio), que es justamente la
    # asimetria 1 condicion / 2 condiciones del contrato.
    check(sin_yolo.ratio_nitidez.confianza == 0.0,
          "A8 sin identidad, la nitidez tambien tiene que cerrar (hereda la condicion)")
    check(con_yolo.ratio_nitidez.confianza == 1.0,
          "A8 con identidad y textura de sobra, la nitidez deberia abrir")
    print("  A8 independencia de gates ..... nitidez hereda la condicion de identidad")

    for p in DIR_TMP.glob("*"):
        p.unlink()
    DIR_TMP.rmdir()


# =============================================================================
# B) Integración sobre data/
# =============================================================================

def integracion():
    print("\n=== B) Integracion percepcion compartida -> espacio y aislamiento ===")
    rutas = sorted(
        (p for p in DIR_DATOS.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: (len(p.stem), p.stem),
    )
    tool = SharedPerceptionTool()
    abren_esp = abren_nit = 0

    for ruta in rutas:
        p = tool._run(str(ruta))
        args = (str(ruta), p["bbox"], p["sujeto_discreto"], p["fuente"],
                p["mapa_saliencia_path"])
        inf = engine.analizar(*args)
        en, ni = inf.ratio_espacio_negativo, inf.ratio_nitidez
        n = ruta.stem

        # --- Dominios declarados por el schema ---
        check(0.0 <= en.valor <= 1.0, f"{n}: ratio_espacio_negativo fuera de [0,1]: {en.valor}")
        check(-1.0 <= ni.valor <= 1.0, f"{n}: ratio_nitidez fuera de [-1,1]: {ni.valor}")
        check(en.valor_norm is None, f"{n}: espacio negativo no debe publicar valor_norm")

        # --- El indice se recomputa desde sus dos operandos ---
        # Es lo que sostiene que el crítico pueda citar el índice Y las varianzas y que ambas
        # cosas describan la misma medición.
        total = inf.var_laplaciano_figura + inf.var_laplaciano_fondo
        if total > 0:
            esperado = (inf.var_laplaciano_figura - inf.var_laplaciano_fondo) / total
            check(abs(ni.valor - esperado) < TOL,
                  f"{n}: ratio_nitidez {ni.valor} no cuadra con sus varianzas ({esperado})")

        # --- valor_norm recomputado desde el valor ---
        if ni.valor_norm is not None:
            check(abs(ni.valor_norm - (ni.valor + 1) / 2) < TOL,
                  f"{n}: valor_norm {ni.valor_norm} no cuadra con valor {ni.valor}")
        else:
            check(total == 0.0,
                  f"{n}: valor_norm None solo se admite con varianza total nula")

        # --- LAS DOS CONFIANZAS, RECOMPUTADAS DESDE LOS CAMPOS PUBLICADOS ---
        hay_identidad = p["fuente"] == FUENTE_CON_IDENTIDAD
        hay_textura = max(inf.var_laplaciano_figura, inf.var_laplaciano_fondo) >= VAR_LAPLACIANO_MIN
        check(en.confianza == (1.0 if hay_identidad else 0.0),
              f"{n}: confianza de espacio {en.confianza} no cuadra con fuente={p['fuente']}")
        check(ni.confianza == (1.0 if (hay_identidad and hay_textura) else 0.0),
              f"{n}: confianza de nitidez {ni.confianza} no cuadra con sus dos condiciones")

        # --- Un gate cerrado SIEMPRE explica por que; uno abierto no inventa motivo ---
        check((en.fuente_confianza is None) == (en.confianza == 1.0),
              f"{n}: fuente_confianza de espacio incoherente con su confianza")
        check((ni.fuente_confianza is None) == (ni.confianza == 1.0),
              f"{n}: fuente_confianza de nitidez incoherente con su confianza")

        # --- La VIA depende de sujeto_discreto, NO del gate (son ortogonales) ---
        esperada = "grabcut" if p["sujeto_discreto"] else "prior_saliencia"
        check(inf.fuente_mascara == esperada,
              f"{n}: via {inf.fuente_mascara} no cuadra con sujeto_discreto={p['sujeto_discreto']}")

        # --- Dimensiones de la imagen ORIGINAL, no las del espacio de trabajo ---
        alto, ancho = cv2.imread(str(ruta)).shape[:2]
        check(inf.dimensiones_imagen == (ancho, alto),
              f"{n}: dimensiones_imagen {inf.dimensiones_imagen} != ({ancho}, {alto})")
        check(Path(inf.verificacion_path).exists(),
              f"{n}: no se genero el PNG de verificacion")

        # --- DETERMINISMO: dos analisis completos, model_dump identico ---
        check(inf.model_dump() == engine.analizar(*args).model_dump(),
              f"{n}: el analisis NO es determinista (grabCut sin sembrar el RNG?)")

        abren_esp += en.confianza == 1.0
        abren_nit += ni.confianza == 1.0
        print(f"  {n:<12} {p['fuente']:<17} esp_neg={en.valor:.3f}({en.confianza:.0f}) "
              f"nitidez={ni.valor:+.3f}({ni.confianza:.0f})")

    print(f"\n  gates abiertos: espacio {abren_esp}/{len(rutas)}  "
          f"nitidez {abren_nit}/{len(rutas)}")
    return len(rutas)


def main():
    casos_analiticos()
    n = integracion()

    print("\n" + "=" * 60)
    if fallos:
        print(f"FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"  - {f}")
        return 1
    print(f"TODO OK: 8/8 casos analiticos + {n}/{n} imagenes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
