"""Limpieza del texto transcrito antes de que llegue al clasificador.

El audio que se transcribe incluye el preroll (los 500 ms previos al disparo),
asi que Whisper casi siempre oye la palabra clave: "Jarvis, sube el volumen".
La `Utterance` final lleva solo la orden; la palabra clave ya cumplio su papel.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize", "strip_wake_word"]

#: Saludos que suelen ir delante de la palabra clave.
_GREETINGS = frozenset({"hey", "oye", "oiga", "ey", "hola", "ok", "okay", "eh"})

_WORD = re.compile(r"[\w']+", re.UNICODE)


def normalize(text: str) -> str:
    """Minusculas y sin acentos, para comparar palabras."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _distance(a: str, b: str) -> int:
    """Distancia de edicion de Levenshtein entre dos palabras cortas."""
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)),
            )
        previous = current
    return previous[-1]


def _sounds_like(word: str, wake_word: str) -> bool:
    """Si una palabra transcrita es la palabra clave mal escrita.

    Whisper escribe "Jarvis" de varias formas segun el acento: "Yarvis",
    "Charvis", "Jarvi". Una distancia de edicion corta las cubre sin comerse
    palabras de verdad; con palabras clave cortas no se tolera ningun error.
    """
    if word == wake_word:
        return True
    # La cola de la palabra clave: el preroll a veces arranca a media palabra
    # y Whisper transcribe solo "vis" o "arvis".
    if len(word) >= 3 and wake_word.endswith(word):
        return True
    tolerance = 2 if len(wake_word) >= 5 else 0
    return abs(len(word) - len(wake_word)) <= tolerance and _distance(word, wake_word) <= tolerance


def strip_wake_word(text: str, wake_word: str) -> str:
    """Quita la palabra clave (y un saludo delante) del principio del texto.

    Solo actua al principio: "dile a Jarvis que..." no se toca. Si la frase era
    solo la palabra clave, devuelve cadena vacia, que el pipeline trata como
    "dijo Jarvis y se quedo callado".
    """
    target = normalize(wake_word.strip())
    if not target:
        return text.strip()

    position = 0
    words = list(_WORD.finditer(text))
    index = 0
    if index < len(words) and normalize(words[index].group()) in _GREETINGS:
        index += 1
    if index < len(words) and _sounds_like(normalize(words[index].group()), target):
        position = words[index].end()
    else:
        return text.strip()

    # Se come la puntuacion que separa la palabra clave de la orden, pero no la
    # de apertura: el "¿" de "Jarvis, ¿que hora es?" es parte de la pregunta.
    remainder = text[position:]
    return remainder.lstrip(" \t\n,.;:!?-").strip()
