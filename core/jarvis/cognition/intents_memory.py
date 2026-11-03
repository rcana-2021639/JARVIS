"""Las intenciones de la memoria (Fase 3) y de las confirmaciones habladas.

DECISION: "recuerda que" y "olvida lo de" no llevan ejemplos para MiniLM:
solo se activan por patron. Medido con el modelo real, "¿como se llama mi
hermana?" se parecia lo bastante a "acuerdate de que mi hermana se llama Ana"
para guardarse como orden de recordar. Una pregunta nunca debe escribir ni
borrar memoria por parecido.

Todas son de nivel 1: guardar un hecho, olvidar, decir lo que se sabe de algo
o lo que paso ayer no necesitan un modelo. Las atiende el sistema de memoria,
que registra sus manejadores al conectarse; sin memoria, el nivel 1 no las
conoce y el enrutador las manda al nivel 2, que dice que no puede.

"Recordar algo que te conte" (`memory.recall`) sigue siendo abierta: la
contesta un modelo con los recuerdos recuperados en su contexto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.cognition.catalog import IntentSpec, clean_target, spec
from jarvis.contracts.enums import CascadeLevel

if TYPE_CHECKING:
    import re

    from jarvis.cognition.catalog import Entities

__all__ = ["MEMORY_INTENTS"]

L1 = CascadeLevel.FAST_PATH


def entity_fact(text: str, match: re.Match[str] | None) -> Entities:
    """Lo que hay que recordar, del grupo `fact` (normalizado; el manejador usa el original)."""
    if match is None or not match.groupdict().get("fact"):
        return {}
    return {"fact": match["fact"].strip()}


def entity_topic(text: str, match: re.Match[str] | None) -> Entities:
    """El tema de "olvida lo de X" o "que sabes de X"."""
    if match is None or not match.groupdict().get("topic"):
        return {}
    topic = clean_target(match["topic"])
    return {"topic": topic} if topic else {}


def entity_when(text: str, match: re.Match[str] | None) -> Entities:
    """El periodo de "que hicimos ayer": ayer, hoy, el martes, esta semana..."""
    if match is None:
        return {}
    when = (match.groupdict().get("when") or "").strip()
    return {"when": when or "hoy"}


MEMORY_INTENTS: tuple[IntentSpec, ...] = (
    spec(
        "memory.remember", L1, "Guardar un dato que el usuario cuenta.",
        patterns=[
            r"^(?:recuerda|anota|apunta|no olvides|acuerdate de|memoriza) que (?P<fact>.+)$",
            r"^(?:guarda|toma nota|anota|apunta)(?: de)?(?: esto| eso)?(?: que)? (?P<fact>.+)$",
        ],
        extract=entity_fact,
    ),
    spec(
        "memory.forget", L1, "Borrar algo de la memoria. Pide confirmacion.",
        patterns=[
            r"^olvida(?:te)? (?:todo )?(?:lo de|lo que sabes de|lo que te dije de|lo que te dije sobre|lo relacionado con|que) (?P<topic>.+)$",
            r"^borra (?:de tu memoria|de la memoria|tus recuerdos de)(?: lo de)? (?P<topic>.+)$",
        ],
        extract=entity_topic,
    ),
    spec(
        "memory.about", L1, "Lo que Jarvis sabe de algo o alguien.",
        patterns=[
            r"^que (?:sabes|recuerdas|tienes guardado|tienes anotado) (?:de|del|sobre|acerca de|de la|de los) (?P<topic>.+)$",
        ],
        examples=["que sabes de maria", "que tienes sobre el proyecto aurora"],
        extract=entity_topic,
    ),
    spec(
        "memory.timeline", L1, "Lo que paso en un periodo.",
        patterns=[
            r"^que (?:hicimos|hablamos|te pedi|me dijiste|hiciste|paso|estuvimos haciendo)\b(?P<when>.*)$",
        ],
        examples=["de que hablamos ayer", "que cosas te pedi esta semana"],
        extract=entity_when,
    ),
    spec(
        "memory.learned", L1, "Lo que aprendio la ultima consolidacion.",
        patterns=[r"^que (?:aprendiste|has aprendido)\b"],
        examples=["que aprendiste anoche"],
    ),
    spec(
        "jarvis.confirm", L1, "Si a la pregunta de confirmacion.",
        patterns=[r"^(?:si|confirmo|confirmado|hazlo|adelante|de acuerdo|dale|procede)(?: (?:hazlo|adelante|confirmo|por favor|claro|si))*$"],
        phrases=["si", "confirmo", "hazlo"],
    ),
    spec(
        "jarvis.deny", L1, "No a la pregunta de confirmacion.",
        patterns=[r"^(?:no|mejor no|no lo hagas|negativo|no gracias)$"],
        phrases=["no", "mejor no"],
    ),
)  # fmt: skip
