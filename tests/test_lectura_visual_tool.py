"""Prueba manual de LecturaVisualTool."""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# La clave de la API vive en .env y este script se ejecuta suelto, fuera de `crewai run`,
# así que hay que cargarla a mano antes de importar nada que construya un cliente.
load_dotenv()

from tfg_multiagente_fotografia.tools.lectura_visual_tool import (  # noqa: E402
    LADO_MAYOR_ENVIO,
    MAX_OBSERVACIONES,
    VERSION_PROMPT,
    LecturaVisualEngine,
    LecturaVisualTool,
    lectura_visual_engine,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# =============================================================================
# A) Casos analíticos — parseo de la respuesta (sin API)
# =============================================================================

def test_parseo():
    """El parseo es tolerante al formato pero estricto con el contenido.

    La tolerancia tiene un motivo económico: la llamada ya está pagada cuando llega la
    respuesta, así que rescatar un JSON envuelto en vallas de código vale más que tirarla
    por un detalle de formato. La estrictitud tiene otro: una lista vacía o un texto que no
    es JSON significan que no hay lectura visual, y eso debe fallar de forma ruidosa en vez
    de devolver silenciosamente cero observaciones —el crítico creería que miró la foto y
    que no había nada que decir—.
    """
    extraer = LecturaVisualEngine._extraer_observaciones
    fallos = []

    casos = [
        (
            "JSON limpio",
            '{"observaciones": ["Una persona camina.", "Hay un perro."]}',
            ["Una persona camina.", "Hay un perro."],
        ),
        (
            "JSON envuelto en valla de codigo",
            '```json\n{"observaciones": ["Un arbol solitario."]}\n```',
            ["Un arbol solitario."],
        ),
        (
            "lista suelta sin envolver en objeto",
            '["Dos cebras.", "Un muro."]',
            ["Dos cebras.", "Un muro."],
        ),
        (
            "entradas vacias o en blanco descartadas",
            '{"observaciones": ["Valida.", "", "   "]}',
            ["Valida."],
        ),
        (
            f"se recorta a MAX_OBSERVACIONES={MAX_OBSERVACIONES}",
            json.dumps({"observaciones": [f"Obs {i}." for i in range(MAX_OBSERVACIONES + 4)]}),
            [f"Obs {i}." for i in range(MAX_OBSERVACIONES)],
        ),
    ]

    for nombre, entrada, esperado in casos:
        try:
            obtenido = extraer(entrada)
            assert obtenido == esperado, f"obtenido={obtenido}, esperado={esperado}"
            print(f"[OK] {nombre}")
        except AssertionError as err:
            fallos.append((nombre, str(err)))
            print(f"[FAIL] {nombre}: {err}")

    # Los dos casos que DEBEN fallar.
    for nombre, entrada in [
        ("texto que no es JSON lanza", "Pues verás, en la foto hay un árbol."),
        ("lista de observaciones vacia lanza", '{"observaciones": []}'),
    ]:
        try:
            extraer(entrada)
            fallos.append((nombre, "no lanzó ValueError"))
            print(f"[FAIL] {nombre}: no lanzó ValueError")
        except ValueError:
            print(f"[OK] {nombre}")

    return fallos


# =============================================================================
# A) Casos analíticos — preparación de la imagen (sin API)
# =============================================================================

def test_preparacion_imagen(imagenes):
    """La reducción no es una optimización: sin ella la petición no cabe.

    `data/` tiene fotografías de ~200 MP, y `inline_data` de Gemini no admite peticiones de
    ese tamaño. Se comprueba además que lo que sale es un JPEG de verdad (cabecera FFD8) y
    no un buffer cualquiera, y que las imágenes pequeñas NO se amplían —ampliar no añade
    información y encarecería la llamada sin ninguna contrapartida—.
    """
    import cv2

    fallos = []
    for ruta in imagenes:
        try:
            datos, (ancho, alto) = LecturaVisualEngine._preparar_imagen(ruta)

            assert datos[:2] == b"\xff\xd8", "los bytes enviados no son un JPEG"
            assert max(ancho, alto) <= LADO_MAYOR_ENVIO, (
                f"lado mayor {max(ancho, alto)} > {LADO_MAYOR_ENVIO}"
            )

            original = cv2.imread(str(ruta))
            alto_orig, ancho_orig = original.shape[:2]
            if max(alto_orig, ancho_orig) <= LADO_MAYOR_ENVIO:
                assert (ancho, alto) == (ancho_orig, alto_orig), "una imagen pequeña se amplió"
            else:
                # La relación de aspecto se conserva: si se deformara, las observaciones
                # sobre la disposición de los elementos describirían otra escena.
                aspecto_orig = ancho_orig / alto_orig
                assert abs(ancho / alto - aspecto_orig) < 0.01, (
                    f"la relación de aspecto cambió: {ancho/alto:.4f} vs {aspecto_orig:.4f}"
                )

            kb = len(datos) / 1024
            print(f"[OK] {ruta.name:<16} {ancho_orig}x{alto_orig} -> {ancho}x{alto}  ({kb:.0f} KB)")
        except AssertionError as err:
            fallos.append((ruta.name, str(err)))
            print(f"[FAIL] {ruta.name}: {err}")

    # Una ruta inexistente falla de forma explícita en vez de mandar bytes vacíos.
    try:
        LecturaVisualEngine._preparar_imagen(DATA_DIR / "no_existe_esta_imagen.jpg")
        fallos.append(("imagen inexistente", "no lanzó ValueError"))
        print("[FAIL] imagen inexistente: no lanzó ValueError")
    except ValueError:
        print("[OK] una imagen inexistente lanza ValueError")

    return fallos


# =============================================================================
# B) Corpus — contrato de la salida y caché
# =============================================================================

def test_lecturas(imagenes):
    """Contrato de la salida y reutilización de la lectura persistida.

    La caché no es solo un ahorro: es lo que hace AUDITABLE la única parte no recomputable
    del sistema. Una crítica escrita sobre una lectura concreta se puede volver a revisar
    contra esa misma lectura, en vez de contra una respuesta nueva del modelo.
    """
    fallos = []
    for ruta in imagenes:
        try:
            lectura = lectura_visual_engine.leer(ruta)
            obs = lectura["observaciones"]

            assert isinstance(obs, list) and obs, "no devolvió observaciones"
            assert len(obs) <= MAX_OBSERVACIONES, f"{len(obs)} observaciones > {MAX_OBSERVACIONES}"
            assert all(isinstance(o, str) and o.strip() for o in obs), (
                "hay observaciones que no son texto no vacío"
            )
            assert lectura["version_prompt"] == VERSION_PROMPT, (
                f"version_prompt={lectura['version_prompt']}, módulo={VERSION_PROMPT}"
            )

            persistida = Path(lectura["lectura_path"])
            assert persistida.exists(), f"no se persistió la lectura en {persistida}"

            # Segunda llamada: debe venir de disco y ser IDÉNTICA. Es la comprobación que
            # sostiene la auditabilidad.
            otra = lectura_visual_engine.leer(ruta)
            assert otra["desde_cache"] is True, "la segunda llamada no usó la caché"
            assert otra["observaciones"] == obs, "la caché devolvió otras observaciones"

            origen = "cache" if lectura["desde_cache"] else "API"
            print(f"[OK] {ruta.name:<16} {len(obs)} observaciones ({origen})")
            print(f"       > {obs[0][:88]}")
        except (AssertionError, ValueError) as err:
            fallos.append((ruta.name, str(err)))
            print(f"[FAIL] {ruta.name}: {err}")

    return fallos


def test_tool():
    """La tool devuelve un dict, no el modelo: el LLM tiene que ver JSON limpio.

    Mismo reparto que en los cuatro especialistas —los tests atacan al engine y la tool solo
    serializa—, de modo que el contrato con CrewAI se comprueba una vez y aparte.
    """
    fallos = []
    imagen = next(iter(sorted(DATA_DIR.glob("*.jpg"))), None)
    if imagen is None:
        return [("tool", "no hay imágenes en data/")]

    salida = LecturaVisualTool()._run(ruta_imagen=str(imagen))
    try:
        assert isinstance(salida, dict), f"la tool devolvió {type(salida).__name__}, no dict"
        assert "observaciones" in salida, "falta la clave 'observaciones'"
        json.dumps(salida, ensure_ascii=False)  # tiene que ser serializable tal cual
        print(f"[OK] la tool devuelve un dict serializable con {len(salida)} claves")
    except (AssertionError, TypeError) as err:
        fallos.append(("tool", str(err)))
        print(f"[FAIL] tool: {err}")
    return fallos


# =============================================================================

def seleccionar_imagenes():
    """Imágenes a probar: las de la línea de comandos, o las que ya tienen lectura.

    Por defecto se limita a las que ya están en caché para que reejecutar el test sea
    gratis. Si no hay ninguna, arranca con una sola imagen para no disparar 33 llamadas de
    golpe la primera vez.
    """
    if len(sys.argv) > 1:
        return [DATA_DIR / nombre for nombre in sys.argv[1:]]

    cacheadas = []
    for lectura in sorted((PROJECT_ROOT / "outputs" / "lectura_visual").glob("*_lectura.json")):
        stem = lectura.stem.replace("_lectura", "")
        candidata = next(iter(DATA_DIR.glob(f"{stem}.*")), None)
        if candidata is not None:
            cacheadas.append(candidata)

    if cacheadas:
        return cacheadas
    return sorted(DATA_DIR.glob("*.jpg"))[:1]


def main():
    imagenes = seleccionar_imagenes()
    faltan = [r for r in imagenes if not r.exists()]
    if faltan:
        raise SystemExit(f"No existen: {[str(r) for r in faltan]}")

    print("=" * 78)
    print("A) CASOS ANALITICOS — parseo de la respuesta (sin API)")
    print("=" * 78)
    fallos = test_parseo()

    print("\n" + "=" * 78)
    print("A) CASOS ANALITICOS — preparacion de la imagen (sin API)")
    print("=" * 78)
    fallos += test_preparacion_imagen(imagenes)

    print("\n" + "=" * 78)
    print(f"B) CORPUS — contrato y cache ({len(imagenes)} imagenes)")
    print("=" * 78)
    fallos += test_lecturas(imagenes)

    print("\n" + "=" * 78)
    print("B) CONTRATO CON CREWAI")
    print("=" * 78)
    fallos += test_tool()

    print("\n--- Resumen ---")
    if fallos:
        print(f"{len(fallos)} FALLOS:")
        for nombre, err in fallos:
            print(f"  - {nombre}: {err}")
        raise SystemExit(1)
    print(f"Todo OK sobre {len(imagenes)} imagenes.")


if __name__ == "__main__":
    main()
