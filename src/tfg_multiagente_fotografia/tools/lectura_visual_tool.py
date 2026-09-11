"""Lectura visual independiente del agente crítico, persistida para su auditoría."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Type

import cv2
from crewai.tools import BaseTool
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from tfg_multiagente_fotografia.tools.capa_verificacion import OUTPUTS_PATH

SUBCARPETA = "lectura_visual"

MODELO = "gemini-2.5-flash"

# Temperatura baja para acotar la variabilidad entre ejecuciones. No la elimina —ver el
# bloque DETERMINISMO del docstring—, así que no se le pide a este número más de lo que da.
TEMPERATURA = 0.2

# --- Envío de la imagen ---
LADO_MAYOR_ENVIO = 1024
CALIDAD_JPEG = 90

MAX_OBSERVACIONES = 6

VERSION_PROMPT = 1

PROMPT = """Vas a ver UNA fotografía y solo eso. Describe su CONTENIDO.

Devuelve entre 3 y 6 observaciones. Cada una, una sola frase en español.

DE QUÉ TIENES QUE HABLAR (lo que solo se sabe mirando):
- Qué se representa: el asunto concreto, el lugar, la situación, la acción.
- El sujeto principal: hacia dónde mira o se orienta, si se dirige hacia dentro o hacia
  fuera del encuadre, qué está haciendo.
- Qué otros elementos compiten con él por la atención, o lo acompañan.
- Qué contienen realmente las zonas que parecen vacías: una pared lisa, un cielo con nubes
  y una multitud desenfocada son tres cosas distintas.
- La relación entre los elementos: quién mira a quién, qué tapa a qué, qué va delante y qué
  detrás.
- Cualquier rasgo llamativo del contenido que un análisis puramente geométrico no recogería.

DE QUÉ NO PUEDES HABLAR EN ABSOLUTO. Otras herramientas ya lo miden con precisión y tu
opinión sobre ello no aporta nada:
- Encuadre y posición: regla de los tercios, centrado, equilibrio, en qué parte del marco
  cae el sujeto, si está desplazado.
- Líneas: si el horizonte está nivelado o torcido, perspectiva, puntos de fuga.
- Espacio y nitidez: cuánto espacio ocupa el sujeto, espacio negativo, si está más o menos
  definido que el fondo, enfoque, desenfoque.
- Luz y color: exposición, si la imagen es clara u oscura, contraste, esquema de color,
  saturación, qué colores dominan.

REGLAS:
- Describe lo que hay, no si está bien o mal. Prohibido valorar: nada de "logrado",
  "acertado", "buena", "mejorable", "interesante".
- No des consejos ni digas cómo se podría mejorar la fotografía.
- Si no estás seguro de algo, no lo digas. Tres observaciones seguras valen más que seis
  con inventos.
- Nada de preámbulos ni de frases de cierre.

FORMATO: devuelve únicamente un objeto JSON con esta forma exacta, sin ningún texto
alrededor y sin vallas de código:
{"observaciones": ["...", "..."]}"""


class LecturaVisualEngine:
    """Manda la imagen a Gemini y devuelve las observaciones ya persistidas en disco."""

    def __init__(self, modelo: str = MODELO):
        self.modelo = modelo
        # El cliente se construye en la PRIMERA llamada y no al importar
        self._cliente: genai.Client | None = None

    # -- infraestructura -------------------------------------------------------------

    def _obtener_cliente(self) -> genai.Client:
        if self._cliente is None:
            # El SDK reconoce ambos nombres. Admitirlos mantiene compatibilidad con
            # instalaciones que usan la convención de Google o la propia del proyecto.
            clave = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not clave:
                raise ValueError(
                    "No hay clave de API para Gemini: la lectura visual no puede "
                    "consultarlo. Define GEMINI_API_KEY (o GOOGLE_API_KEY) en .env."
                )
            self._cliente = genai.Client(api_key=clave)
        return self._cliente

    @staticmethod
    def _ruta_lectura(ruta_imagen: str | Path) -> Path:
        carpeta = OUTPUTS_PATH / SUBCARPETA
        carpeta.mkdir(parents=True, exist_ok=True)
        return carpeta / f"{Path(ruta_imagen).stem}_lectura.json"

    @staticmethod
    def _preparar_imagen(ruta_imagen: str | Path) -> tuple[bytes, tuple[int, int]]:
        """Lee, reduce y recodifica a JPEG. Devuelve los bytes y las dimensiones enviadas."""
        img = cv2.imread(str(ruta_imagen))
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {ruta_imagen}")

        alto, ancho = img.shape[:2]
        escala = LADO_MAYOR_ENVIO / max(alto, ancho)
        if escala < 1.0:
            nuevo = (int(round(ancho * escala)), int(round(alto * escala)))
            # INTER_AREA promedia al reducir: no fabrica detalle que no estuviera.
            img = cv2.resize(img, nuevo, interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPEG])
        if not ok:
            raise ValueError(f"No se pudo codificar la imagen a JPEG: {ruta_imagen}")

        alto_env, ancho_env = img.shape[:2]
        return buffer.tobytes(), (ancho_env, alto_env)

    # -- parseo ----------------------------------------------------------------------

    @staticmethod
    def _extraer_observaciones(texto: str) -> list[str]:
        """Convierte la respuesta del modelo en la lista de observaciones.

        Se pide JSON explícitamente y con `response_mime_type`, pero el parseo es tolerante
        a que venga envuelto en vallas de código: preferimos recuperar una lectura válida a
        tirar una llamada ya pagada por un detalle de formato.
        """
        limpio = texto.strip()
        if limpio.startswith("```"):
            limpio = limpio.split("```")[1] if "```" in limpio[3:] else limpio[3:]
            if limpio.lstrip().lower().startswith("json"):
                limpio = limpio.lstrip()[4:]
            limpio = limpio.strip()

        try:
            datos = json.loads(limpio)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"La lectura visual no devolvió JSON válido. Respuesta: {texto[:300]}"
            ) from exc

        observaciones = datos.get("observaciones") if isinstance(datos, dict) else datos
        if not isinstance(observaciones, list):
            raise ValueError(
                f"La lectura visual no devolvió una lista de observaciones: {texto[:300]}"
            )

        limpias = [str(o).strip() for o in observaciones if str(o).strip()]
        if not limpias:
            raise ValueError("La lectura visual devolvió una lista de observaciones vacía.")
        # Se recorta en vez de fallar: una observación de más no invalida las anteriores.
        return limpias[:MAX_OBSERVACIONES]

    # -- API pública -----------------------------------------------------------------

    def leer(self, ruta_imagen: str | Path, forzar: bool = False) -> dict:
        """Devuelve las observaciones visuales, reutilizando la lectura de disco si la hay.

        `forzar=True` ignora la caché y vuelve a consultar. Es el mando para regenerar una
        lectura concreta sin tener que borrar el fichero a mano.
        """
        destino = self._ruta_lectura(ruta_imagen)

        if destino.exists() and not forzar:
            guardada = json.loads(destino.read_text(encoding="utf-8"))
            # Si el prompt ha cambiado desde que se escribió, la lectura guardada responde
            # a otras instrucciones: se regenera en vez de servirla en silencio.
            if guardada.get("version_prompt") == VERSION_PROMPT:
                guardada["desde_cache"] = True
                return guardada

        datos_imagen, dimensiones = self._preparar_imagen(ruta_imagen)

        respuesta = self._obtener_cliente().models.generate_content(
            model=self.modelo,
            contents=[
                types.Part.from_bytes(data=datos_imagen, mime_type="image/jpeg"),
                types.Part.from_text(text=PROMPT),
            ],
            config=types.GenerateContentConfig(
                temperature=TEMPERATURA,
                response_mime_type="application/json",
            ),
        )

        lectura = {
            "ruta_imagen": str(ruta_imagen),
            "observaciones": self._extraer_observaciones(respuesta.text or ""),
            "modelo": self.modelo,
            "version_prompt": VERSION_PROMPT,
            "dimensiones_enviadas": list(dimensiones),
            "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "lectura_path": str(destino),
        }
        destino.write_text(
            json.dumps(lectura, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        lectura["desde_cache"] = False
        return lectura


lectura_visual_engine = LecturaVisualEngine()


class LecturaVisualInput(BaseModel):
    """Esquema de entrada: solo la ruta de la imagen."""

    ruta_imagen: str = Field(
        ..., description="Ruta local de la fotografía a observar (.jpg, .jpeg, .png)."
    )


class LecturaVisualTool(BaseTool):
    name: str = "Lectura visual"
    description: str = (
        "Mira la fotografía y devuelve entre 3 y 6 observaciones sobre su CONTENIDO: de "
        "qué es la escena, hacia dónde se orienta el sujeto, qué elementos compiten con "
        "él y qué hay realmente en las zonas aparentemente vacías. "
        "IMPORTANTE: estas observaciones NO son métricas y no se pueden citar como tales. "
        "Son la única parte del análisis sin respaldo verificable, así que se recogen "
        "aparte y marcadas como observación visual, nunca mezcladas con una afirmación "
        "métrica ni usadas para contradecir, corregir o matizar ninguna medición de los "
        "especialistas. "
        "Tampoco hablan de encuadre, líneas, espacio, nitidez, luz ni color: esas cuatro "
        "dimensiones ya están medidas y esta herramienta tiene prohibido opinar sobre "
        "ellas. Sirve para lo que ninguna métrica alcanza."
    )
    args_schema: Type[BaseModel] = LecturaVisualInput

    def _run(self, ruta_imagen):
        return lectura_visual_engine.leer(ruta_imagen)
