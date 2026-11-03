"""Utilidades de texto y tiempo que comparten las capas de memoria."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime

from jarvis.cognition.text import normalize

__all__ = [
    "clip",
    "fact_key",
    "fold",
    "from_epoch",
    "fts_query",
    "keywords",
    "raw_capture",
    "sentence",
    "to_epoch",
    "to_second_person",
]

_STOPWORDS = frozenset(
    " ".join(
        (
            "a al algo algun alguna alguno ante como con contra cual cuando de del desde donde",
            "durante e el ella ellas ellos en entre era eres es esa ese eso esta este esto estos",
            "fue ha han hay la las le les lo los mas me mi mis muy nada ni no nos o os para pero",
            "por que quien se sea ser si sin sobre son su sus te ti tu tus un una uno unos y ya yo",
            "sabes sabe recuerdas recuerda dime dije conte olvida jarvis acuerdas",
        )
    ).split()
)
_WORD = re.compile(r"\w+")


def fold(text: str) -> str:
    """Minusculas y sin acentos, caracter por caracter: conserva el largo.

    Sirve para buscar con regex sobre una version simple del texto y recortar
    el ORIGINAL en las mismas posiciones, con sus mayusculas y tildes.
    """
    return "".join(unicodedata.normalize("NFD", ch.lower())[0] for ch in text)


def raw_capture(text: str, pattern: re.Pattern[str], group: str) -> str | None:
    """El grupo `group` del patron, recortado del texto original.

    El patron se escribe para el texto plegado (`fold`): minusculas sin tildes.
    """
    match = pattern.search(fold(text))
    if match is None or match.group(group) is None:
        return None
    value = text[match.start(group) : match.end(group)]
    return value.strip(" \t\n.,;:!?¿¡\"'«»") or None


def keywords(text: str) -> list[str]:
    """Las palabras con contenido de un texto, normalizadas y sin repetir."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(normalize(text)):
        if len(word) > 2 and word not in _STOPWORDS:
            seen.setdefault(word, None)
    return list(seen)


def fts_query(text: str) -> str | None:
    """Una consulta FTS5 segura: palabras clave entre comillas unidas con OR.

    Nunca pasa el texto tal cual: FTS5 tiene su propia sintaxis (AND, NEAR,
    comillas, asteriscos) y una orden cualquiera podria romper la consulta.
    """
    words = keywords(text)
    if not words:
        return None
    return " OR ".join(f'"{word}"' for word in words)


_FIRST_TO_SECOND = (
    (r"\bme llamo\b", "te llamas"),
    (r"\bmi\b", "tu"),
    (r"\bmis\b", "tus"),
    (r"\bme\b", "te"),
    (r"\byo\b", "tú"),
    (r"\bconmigo\b", "contigo"),
    (r"\bsoy\b", "eres"),
    (r"\bestoy\b", "estás"),
    (r"\btengo\b", "tienes"),
    (r"\bvivo\b", "vives"),
    (r"\btrabajo\b", "trabajas"),
    (r"\bprefiero\b", "prefieres"),
    (r"\bodio\b", "odias"),
    (r"\bquiero\b", "quieres"),
    (r"\bnecesito\b", "necesitas"),
    (r"\buso\b", "usas"),
)


def to_second_person(text: str) -> str:
    """ "mi hermana se llama Ana" -> "tu hermana se llama Ana".

    DECISION: los hechos se guardan en segunda persona, como Jarvis le habla al
    usuario. Guardados tal cual ("mi hermana..."), el modelo los lee como si
    fueran de Jarvis; y "Anotado: tu hermana se llama Ana" suena natural.
    """
    result = text
    for pattern, replacement in _FIRST_TO_SECOND:

        def swap(match: re.Match[str], new: str = replacement) -> str:
            word = match.group()
            return new.capitalize() if word[:1].isupper() else new

        result = re.sub(pattern, swap, result, flags=re.IGNORECASE)
    # Infinitivo con pronombre: "para concentrarme" -> "para concentrarte".
    return re.sub(r"\b(\w+[aei]r)me\b", r"\1te", result)


def sentence(text: str) -> str:
    """Primera letra mayuscula y punto final."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?…" else text + "."


_COPULAS = re.compile(
    r"\b(?:se llama|se llaman|es|son|era|vive en|viven en|trabaja en|trabajan en|"
    r"esta en|estan en|cumple|cumplen|prefiere|prefieres|le gusta|te gusta|"
    r"te llamas|eres|estas|vives en|trabajas en|tienes|usa|usas)\b"
)


def fact_key(content: str) -> str:
    """El "sujeto + verbo" de un hecho, para detectar que cambio.

    "Tu hermana se llama Ana" y "Tu hermana se llama Lucia" comparten la clave
    "tu hermana se llama": el segundo reemplaza al primero. Sin verbo
    reconocible no hay clave y el hecho nunca reemplaza a otro.
    """
    normalized = normalize(content)
    match = _COPULAS.search(normalized)
    if match is None or match.start() == 0:
        return ""
    return normalized[: match.end()].strip()


def clip(text: str, limit: int) -> str:
    """Recorta con puntos suspensivos, sin partir la ultima palabra."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:") + "…"


def to_epoch(moment: datetime) -> float:
    """Segundos desde 1970 (UTC). Ingenuo se interpreta como hora local."""
    return moment.timestamp()


def from_epoch(value: float) -> datetime:
    """De segundos a fecha con zona (UTC)."""
    return datetime.fromtimestamp(value, tz=UTC)
