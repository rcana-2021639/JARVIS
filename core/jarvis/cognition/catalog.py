"""El catalogo de intenciones: que entiende Jarvis y a que nivel lo manda.

Cada intencion declara tres cosas, todas sobre texto ya normalizado (ver
`text.normalize`: sin acentos, minusculas, numeros en cifras):

    patterns   regex de forma fija. Si uno encaja, la intencion es segura y
               sus grupos con nombre se vuelven entidades.
    phrases    formas canonicas para la coincidencia difusa con rapidfuzz.
               Rescatan los errores de Whisper: "sierra todas las ventanas".
    examples   frases naturales para el clasificador de embeddings. Cubren lo
               que ningun patron preve: "me podrias poner algo de musica".

El ORDEN del catalogo es la prioridad de los patrones: la primera intencion
cuyo patron encaje gana. Por eso las especificas van antes que las generales
("abre el archivo X" antes que "abre X").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jarvis.cognition.text import duration_seconds, extract_number
from jarvis.contracts.enums import CascadeLevel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic import JsonValue

__all__ = [
    "Entities",
    "IntentSpec",
    "build_catalog",
    "clean_target",
    "entity_app",
    "entity_expression",
    "entity_query",
    "entity_timer",
    "entity_value",
    "spec",
]

Entities = dict[str, "JsonValue"]
Extractor = "Callable[[str, re.Match[str] | None], Entities]"


@dataclass(frozen=True, slots=True)
class IntentSpec:
    """La definicion de una intencion del catalogo."""

    name: str
    level: CascadeLevel
    description: str
    patterns: tuple[re.Pattern[str], ...] = ()
    phrases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    extract: Callable[[str, re.Match[str] | None], Entities] | None = field(default=None)

    @property
    def domain(self) -> str:
        """El espacio de nombres: `media` para `media.volume_up`."""
        return self.name.split(".", 1)[0]

    @property
    def training_texts(self) -> tuple[str, ...]:
        """Lo que ve el clasificador de embeddings: frases fijas y ejemplos."""
        return self.phrases + self.examples

    def entities(self, text: str, match: re.Match[str] | None) -> Entities:
        """Entidades extraidas del texto normalizado."""
        return self.extract(text, match) if self.extract is not None else {}


def spec(
    name: str,
    level: CascadeLevel,
    description: str,
    *,
    patterns: Sequence[str] = (),
    phrases: Sequence[str] = (),
    examples: Sequence[str] = (),
    extract: Callable[[str, re.Match[str] | None], Entities] | None = None,
) -> IntentSpec:
    """Atajo para declarar intenciones legibles en los modulos de datos."""
    return IntentSpec(
        name=name,
        level=level,
        description=description,
        patterns=tuple(re.compile(p) for p in patterns),
        phrases=tuple(phrases),
        examples=tuple(examples),
        extract=extract,
    )


# ---------------------------------------------------------------------------
# Extractores de entidades compartidos
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"^(?:el|la|los|las|un|una|mi|mis|al|a)\s+")
_TRAILING = re.compile(r"\s+(?:por favor|porfa|ahora|ya|ahorita|rapido)$")


def clean_target(raw: str) -> str:
    """Quita articulos y muletillas de un objetivo: "el spotify por favor" -> "spotify"."""
    value = raw.strip()
    previous = None
    while previous != value:
        previous = value
        value = _TRAILING.sub("", _ARTICLES.sub("", value)).strip()
    return value


def entity_value(text: str, match: re.Match[str] | None) -> Entities:
    """El numero de la orden, acotado a 0-100: "al 70%" -> {"value": 70}."""
    number = extract_number(text)
    return {} if number is None else {"value": max(0, min(100, number))}


def entity_app(text: str, match: re.Match[str] | None) -> Entities:
    """La aplicacion del grupo `app` del patron."""
    if match is None or not match.groupdict().get("app"):
        return {}
    target = clean_target(match["app"])
    return {"app": target} if target else {}


def entity_query(text: str, match: re.Match[str] | None) -> Entities:
    """Lo que hay que buscar o reproducir, del grupo `query`."""
    if match is None or not match.groupdict().get("query"):
        return {}
    target = clean_target(match["query"])
    return {"query": target} if target else {}


_NUM = r"\d+(?:[.,]\d+)?"
_PERCENT_OF = re.compile(rf"({_NUM})\s*%\s*de\s*({_NUM})")
_OPERATORS = (
    (re.compile(r"\bdividido (?:entre|por)\b"), "/"),
    (re.compile(r"\belevado a\b"), "**"),
    (re.compile(r"\b(?:por|x|multiplicado por)\b"), "*"),
    (re.compile(r"\b(?:entre|sobre)\b"), "/"),
    (re.compile(r"\bmas\b"), "+"),
    (re.compile(r"\bmenos\b"), "-"),
)


def entity_expression(text: str, match: re.Match[str] | None) -> Entities:
    """La cuenta en notacion aritmetica: "el 15% de 230" -> "(15/100)*230"."""
    if match is None or not match.groupdict().get("expr"):
        return {}
    expression = _PERCENT_OF.sub(r"(\1/100)*\2", match["expr"])
    for pattern, symbol in _OPERATORS:
        expression = pattern.sub(f" {symbol} ", expression)
    return {"expression": " ".join(expression.replace(",", ".").split())}


_TIMER_LABEL = re.compile(r"\b(?:para|que)\s+(?P<label>(?!\d).+)$")


def entity_timer(text: str, match: re.Match[str] | None) -> Entities:
    """Duracion en segundos y, si la hay, la etiqueta: "... para sacar el pan"."""
    entities: Entities = {}
    seconds = duration_seconds(text)
    if seconds is not None:
        entities["seconds"] = seconds
    label = _TIMER_LABEL.search(text)
    if label:
        entities["label"] = clean_target(label["label"])
    return entities


def build_catalog() -> tuple[IntentSpec, ...]:
    """El catalogo completo, en orden de prioridad de patrones.

    Las intenciones abiertas con patrones muy especificos (archivo, navegador,
    nube explicita, codigo) van delante de las fijas generales como `apps.open`.
    Las de memoria van antes que las fijas: "olvida lo de X" no es "olvidalo".
    """
    from jarvis.cognition.intents_fast import FAST_INTENTS
    from jarvis.cognition.intents_memory import MEMORY_INTENTS
    from jarvis.cognition.intents_open import OPEN_INTENTS, PRIORITY_OPEN_INTENTS

    catalog = (*PRIORITY_OPEN_INTENTS, *MEMORY_INTENTS, *FAST_INTENTS, *OPEN_INTENTS)
    names = [item.name for item in catalog]
    duplicated = {name for name in names if names.count(name) > 1}
    if duplicated:
        msg = f"Intenciones duplicadas en el catalogo: {sorted(duplicated)}"
        raise ValueError(msg)
    return catalog
