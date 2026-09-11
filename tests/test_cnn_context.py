"""Prueba manual de carga de imágenes del clasificador de contexto."""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

from tfg_multiagente_fotografia.tools import cnn_context
from tfg_multiagente_fotografia.tools.cnn_context import (
    LIMITE_MEGAPIXELES,
    MEGAPIXELES_DECODIFICADO_REDUCIDO,
    _abrir_para_clasificar,
)

# Limite de Pillow, en pixeles. Avisa por encima del valor y LANZA por encima del doble.
LIMITE_PIL_ERROR = 2 * 89478485

# 14000x14000 = 196 MP: por encima del limite de error de PIL (179 MP) y por debajo del
# nuestro (400 MP), que es justo la banda donde antes se caia y ahora tiene que funcionar.
LADO_ENORME = 14000


def _fabricar_enorme(destino: Path) -> float:
    """Un JPEG de 196 MP. En escala de grises y en degradado a proposito.

    Gris para que la matriz ocupe 196 MB en vez de 588 MB, y degradado para que comprima
    a unos pocos MB: lo que se esta probando es la CABECERA (PIL decide con las
    dimensiones declaradas, antes de decodificar), no el contenido.
    """
    fila = np.linspace(0, 255, LADO_ENORME, dtype=np.uint8)
    lienzo = np.broadcast_to(fila, (LADO_ENORME, LADO_ENORME))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DecompressionBombWarning al construir
        previo = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            Image.fromarray(lienzo, mode="L").save(destino, quality=70)
        finally:
            Image.MAX_IMAGE_PIXELS = previo
    return LADO_ENORME * LADO_ENORME / 1e6


# =============================================================================
# A) No regresion sobre el corpus
# =============================================================================

def test_corpus_sin_cambios():
    """Las 37 imagenes de data/ deben dar pixeles IDENTICOS a `Image.open()` pelado.

    Es identidad por construccion —por debajo de MEGAPIXELES_DECODIFICADO_REDUCIDO el
    codigo hace literalmente lo de antes— pero se comprueba igual, porque es la asercion
    que impide que un retoque futuro del umbral meta `draft()` en el camino de las
    imagenes con las que se valido el sistema.
    """
    fallos = []
    imagenes = sorted(p for p in DATA_DIR.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not imagenes:
        print("[SKIP] no hay imagenes en data/")
        return fallos

    reducidas = 0
    for ruta in imagenes:
        con_politica = _abrir_para_clasificar(ruta)
        directa = Image.open(ruta).convert("RGB")
        try:
            if con_politica.size != directa.size:
                fallos.append((ruta.name, f"tamano {con_politica.size} vs {directa.size}"))
                continue
            if not np.array_equal(np.asarray(con_politica), np.asarray(directa)):
                fallos.append((ruta.name, "los pixeles difieren del open() directo"))
                continue
            mp = directa.size[0] * directa.size[1] / 1e6
            if mp > MEGAPIXELES_DECODIFICADO_REDUCIDO:
                reducidas += 1
        finally:
            con_politica.close()
            directa.close()

    if fallos:
        for nombre, motivo in fallos:
            print(f"[FAIL] {nombre}: {motivo}")
    else:
        print(f"[OK] corpus sin cambios: {len(imagenes)}/{len(imagenes)} identicas al open() directo")
        print(f"     ninguna supera los {MEGAPIXELES_DECODIFICADO_REDUCIDO:.0f} MP del corte "
              f"(en camino reducido: {reducidas})")
    return fallos


# =============================================================================
# B) La imagen enorme de verdad
# =============================================================================

def test_imagen_enorme(destino: Path):
    """196 MP: PIL sola tiene que reventar, y el clasificador tiene que poder con ella."""
    fallos = []
    mp = _fabricar_enorme(destino)

    # 1) Sin esto el test no probaria nada: hay que ver el fallo original.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Image.open(destino).convert("RGB")
        fallos.append(("reproduccion", "Image.open() NO lanzo: el limite de PIL ha cambiado "
                                       "y este test ya no prueba el fallo que dice probar"))
        print("[FAIL] reproduccion: PIL no lanzo con 196 MP")
    except Image.DecompressionBombError:
        print(f"[OK] reproducido el fallo original: PIL lanza DecompressionBombError con "
              f"{mp:.0f} MP (su limite son {LIMITE_PIL_ERROR/1e6:.0f} MP)")

    # 2) Y ahora el arreglo.
    try:
        img = _abrir_para_clasificar(destino)
        ancho, alto = img.size
        img.close()
        if ancho >= LADO_ENORME:
            fallos.append(("decodificado reducido",
                           f"draft() no redujo nada: sigue en {ancho}x{alto}"))
            print(f"[FAIL] decodificado reducido: {ancho}x{alto}")
        else:
            print(f"[OK] decodificado reducido: {LADO_ENORME}x{LADO_ENORME} -> {ancho}x{alto} "
                  f"({mp:.0f} MP -> {ancho*alto/1e6:.1f} MP), muy por encima de los 224 px del modelo")
    except Exception as err:
        fallos.append(("imagen enorme", f"{type(err).__name__}: {err}"))
        print(f"[FAIL] imagen enorme: {type(err).__name__}: {err}")
    return fallos


# =============================================================================
# C) Rechazo por encima del limite del sistema
# =============================================================================

def test_rechazo_por_limite(destino: Path):
    """Por encima de LIMITE_MEGAPIXELES se rechaza, con un mensaje que se pueda leer.

    Se baja el limite en vez de fabricar un fichero de 400 MP: la rama de codigo es la
    misma y fabricarlo costaria medio giga de RAM para no probar nada nuevo.
    """
    fallos = []
    original = cnn_context.LIMITE_MEGAPIXELES
    cnn_context.LIMITE_MEGAPIXELES = 1.0
    try:
        _abrir_para_clasificar(destino)
        fallos.append(("rechazo", "no lanzo con el limite bajado a 1 MP"))
        print("[FAIL] rechazo: no lanzo")
    except ValueError as err:
        texto = str(err)
        # El mensaje lo va a leer un usuario en la interfaz, no un desarrollador en un log.
        faltan = [t for t in (destino.name, "MP", "límite") if t not in texto]
        if faltan:
            fallos.append(("mensaje", f"no menciona {faltan}: {texto}"))
            print(f"[FAIL] mensaje incompleto: {texto}")
        else:
            print(f"[OK] rechazo con mensaje legible: \"{texto}\"")
    except Exception as err:
        fallos.append(("rechazo", f"lanzo {type(err).__name__} en vez de ValueError"))
        print(f"[FAIL] rechazo: {type(err).__name__}")
    finally:
        cnn_context.LIMITE_MEGAPIXELES = original
    return fallos


# =============================================================================
# D) La guarda global de PIL no se queda desactivada
# =============================================================================

def test_guarda_restaurada(destino: Path):
    """`MAX_IMAGE_PIXELS` es estado GLOBAL: una fuga desprotege a todo el proceso.

    Se comprueba en los tres caminos, y el que de verdad importa es el del RECHAZO: es
    el unico que sale por una excepcion, o sea el unico donde una restauracion mal
    puesta (fuera de un `finally`) pasaria desapercibida.
    """
    fallos = []
    esperado = Image.MAX_IMAGE_PIXELS

    _abrir_para_clasificar(sorted(DATA_DIR.glob("*.jpg"))[0]).close()
    if Image.MAX_IMAGE_PIXELS != esperado:
        fallos.append(("tras camino normal", f"{Image.MAX_IMAGE_PIXELS} != {esperado}"))

    _abrir_para_clasificar(destino).close()
    if Image.MAX_IMAGE_PIXELS != esperado:
        fallos.append(("tras camino reducido", f"{Image.MAX_IMAGE_PIXELS} != {esperado}"))

    original = cnn_context.LIMITE_MEGAPIXELES
    cnn_context.LIMITE_MEGAPIXELES = 1.0
    try:
        _abrir_para_clasificar(destino)
    except ValueError:
        pass
    finally:
        cnn_context.LIMITE_MEGAPIXELES = original
    if Image.MAX_IMAGE_PIXELS != esperado:
        fallos.append(("tras rechazo", f"{Image.MAX_IMAGE_PIXELS} != {esperado}"))

    if fallos:
        for donde, motivo in fallos:
            print(f"[FAIL] guarda {donde}: {motivo}")
    else:
        print(f"[OK] guarda de PIL restaurada en los tres caminos ({esperado})")
    return fallos


# =============================================================================

def main() -> int:
    print("=" * 78)
    print("CARGA DE IMAGEN DEL CLASIFICADOR — politica de resolucion extrema")
    print(f"  corte de decodificado reducido: {MEGAPIXELES_DECODIFICADO_REDUCIDO:.0f} MP")
    print(f"  limite del sistema            : {LIMITE_MEGAPIXELES:.0f} MP")
    print("=" * 78)

    fallos = []
    print("\n--- A) No regresion sobre data/ ---")
    fallos += test_corpus_sin_cambios()

    with tempfile.TemporaryDirectory() as tmp:
        enorme = Path(tmp) / "enorme.jpg"
        print(f"\n--- B) Imagen de {LADO_ENORME}x{LADO_ENORME} (fabricandola, tarda unos segundos) ---")
        fallos += test_imagen_enorme(enorme)
        print("\n--- C) Rechazo por encima del limite ---")
        fallos += test_rechazo_por_limite(enorme)
        print("\n--- D) Guarda global restaurada ---")
        fallos += test_guarda_restaurada(enorme)

    print("\n--- Resumen ---")
    if fallos:
        for donde, motivo in fallos:
            print(f"  {donde}: {motivo}")
        print(f"{len(fallos)} FALLO(S)")
        return 1
    print("Todo OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
