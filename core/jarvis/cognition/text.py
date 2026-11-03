"""Normalizacion del texto y extraccion de entidades para el clasificador.

Whisper `base` escribe las ordenes de muchas formas: "Sube el volumen al 70%",
"sube el volumen a setenta por ciento", "Súbele al volumen". Antes de comparar
contra el catalogo todo se lleva a una forma canonica: minusculas, sin acentos,
sin puntuacion y con los numeros escritos en cifras.

Todo aqui es puro y sincrono: corre en microsegundos dentro del presupuesto de
10 ms del clasificador.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "duration_seconds",
    "extract_number",
    "normalize",
    "words_to_numbers",
]

_UNITS: dict[str, int] = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiun": 21, "veintiuno": 21, "veintidos": 22,
    "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}  # fmt: skip
_TENS: dict[str, int] = {
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90,
}  # fmt: skip

_PUNCTUATION = re.compile(r"[^\w\s%:.]")
_LOOSE_DOT = re.compile(r"(?<!\w)\.|\.(?!\w)")
"""Un punto solo sobrevive entre letras: "youtube.com" si, "hola." no."""
_SPACES = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+")
_WORD_NUMBER = re.compile(
    r"\b(?:(?P<tens>" + "|".join(_TENS) + r")(?:\s+y\s+(?P<unit>" + "|".join(_UNITS) + r"))?"
    r"|(?P<single>" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r"))\b"
)
_HUNDREDS: dict[str, int] = {
    "ciento": 100, "cien": 100, "doscientos": 200, "doscientas": 200,
    "trescientos": 300, "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400,
    "quinientos": 500, "quinientas": 500, "seiscientos": 600, "seiscientas": 600,
    "setecientos": 700, "setecientas": 700, "ochocientos": 800, "ochocientas": 800,
    "novecientos": 900, "novecientas": 900,
}  # fmt: skip
_HUNDRED = re.compile(
    r"\b(?P<hundred>" + "|".join(sorted(_HUNDREDS, key=len, reverse=True)) + r")\b"
    r"(?:\s+(?P<rest>\d{1,2})\b)?"
)
_PERCENT_WORDS = re.compile(r"\s*por\s*ciento\b")
_DURATION = re.compile(
    r"(?P<n>\d+)\s*(?P<unit>segundos?|seg|minutos?|min|horas?|h)\b"
    r"|(?P<half>media\s+hora)"
    r"|(?P<quarter>(?:un\s+)?cuarto\s+de\s+hora)"
)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    # La enie no es un acento: "ano" y "año" son palabras distintas.
    kept = (c for c in decomposed if unicodedata.category(c) != "Mn" or c == "̃")
    return unicodedata.normalize("NFC", "".join(kept))


def words_to_numbers(text: str) -> str:
    """Convierte numeros escritos en palabras a cifras: "setenta y cinco" -> "75".

    Cubre de cero a novecientos noventa y nueve, que es lo que una orden hablada usa
    (volumen, brillo, minutos). "un" y "una" solo se convierten delante de una
    unidad de tiempo, para no romper "abre una pestana".
    """

    def replace(match: re.Match[str]) -> str:
        if match["tens"]:
            return str(_TENS[match["tens"]] + (_UNITS[match["unit"]] if match["unit"] else 0))
        word = match["single"]
        if word in {"un", "una", "uno"}:
            tail = text[match.end() : match.end() + 12]
            if not re.match(r"\s+(?:segundo|minuto|hora)", tail):
                return word
        return str(_UNITS[word])

    # Primero decenas y unidades; despues "ciento 20" se junta en 120.
    small = _WORD_NUMBER.sub(replace, text)
    return _HUNDRED.sub(lambda m: str(_HUNDREDS[m["hundred"]] + int(m["rest"] or 0)), small)


def normalize(text: str) -> str:
    """Forma canonica para comparar: minusculas, sin acentos ni puntuacion.

    >>> normalize("¿Súbele al volumen a setenta por ciento!")
    'subele al volumen a 70%'
    """
    lowered = _strip_accents(text.lower())
    lowered = _LOOSE_DOT.sub(" ", _PUNCTUATION.sub(" ", lowered))
    lowered = _SPACES.sub(" ", lowered).strip()
    # "por ciento" va antes que los numeros: si no, "ciento" se vuelve 100.
    lowered = _PERCENT_WORDS.sub("%", lowered)
    return words_to_numbers(lowered)


def extract_number(text: str) -> int | None:
    """El primer numero entero del texto ya normalizado, o None."""
    match = _NUMBER.search(text)
    return int(match.group()) if match else None


def duration_seconds(text: str) -> int | None:
    """Duracion total en segundos de expresiones como "5 minutos y 30 segundos".

    Devuelve None si no hay ninguna duracion.
    """
    total = 0
    found = False
    for match in _DURATION.finditer(text):
        found = True
        if match["half"]:
            total += 1800
        elif match["quarter"]:
            total += 900
        else:
            unit = match["unit"]
            factor = 3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1
            total += int(match["n"]) * factor
    return total if found else None
