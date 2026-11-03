"""El control por voz de la memoria: los manejadores del nivel 1.

    "Jarvis, recuerda que [X]"    guarda el hecho con importancia alta
    "Jarvis, olvida lo de [X]"    pregunta y, con un "si", lo borra de todo
    "Jarvis, que sabes de [X]"    grafo + hechos
    "Jarvis, que hicimos ayer"    linea de tiempo episodica
    "Jarvis, que aprendiste"      el reporte de la ultima consolidacion

Todos son deterministas: ninguno llama a un modelo.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from jarvis.cognition.fastpath import FastOutcome
from jarvis.contracts.enums import Emotion
from jarvis.memory.extraction import remembered_fact
from jarvis.memory.periods import period_bounds
from jarvis.memory.text import clip, keywords, raw_capture, sentence, to_second_person

if TYPE_CHECKING:
    from jarvis.cognition.catalog import Entities
    from jarvis.cognition.confirm import PendingConfirmations
    from jarvis.cognition.fastpath import FastContext, FastPath
    from jarvis.memory.service import MemorySystem

__all__ = ["MemoryVoice", "lower_first"]

_FORGET = re.compile(
    r"(?:olvida(?:te)?|borra)\b(?: de tu memoria| de la memoria| tus recuerdos de)?"
    r"(?: todo)?(?: lo de| lo que sabes de| lo que te dije de| lo que te dije sobre"
    r"| lo relacionado con| que)? (?P<topic>.+)$"
)
_ABOUT = re.compile(
    r"\bque (?:sabes|recuerdas|tienes guardado|tienes anotado) "
    r"(?:de|del|sobre|acerca de) (?P<topic>.+)$"
)
_ARTICLE = re.compile(r"^(?:el|la|los|las|mi|mis)\s+", re.IGNORECASE)
_SPOKEN_ITEMS = 3
_TIMELINE_TOPICS = 3


def lower_first(text: str) -> str:
    """ "Tu hermana..." -> "tu hermana...", sin tocar nombres propios."""
    first = text.split(" ", 1)[0]
    if first in {"Tu", "Tus", "Te", "No", "Trabajas", "Vives", "Odias", "Prefieres", "Eres"}:
        return text[:1].lower() + text[1:]
    return text


def _topic(ctx: FastContext, entities: Entities, pattern: re.Pattern[str]) -> str | None:
    raw = raw_capture(ctx.text, pattern, "topic") if ctx.text else None
    value = raw or entities.get("topic")
    return _ARTICLE.sub("", str(value)).strip() or None if value else None


class MemoryVoice:
    """Los manejadores, ligados al sistema de memoria."""

    def __init__(self, memory: MemorySystem) -> None:
        """Args: memory, las cinco capas."""
        self._memory = memory

    def register(self, fast: FastPath) -> None:
        """Da de alta las intenciones de memoria en el nivel 1."""
        fast.register("memory.remember", self.remember)
        fast.register("memory.forget", self.forget)
        fast.register("memory.about", self.about)
        fast.register("memory.timeline", self.timeline)
        fast.register("memory.learned", self.learned)

    async def remember(self, entities: Entities, ctx: FastContext) -> FastOutcome:
        """ "Recuerda que mi hermana se llama Ana" -> "Anotado: tu hermana se llama Ana"."""
        fact = remembered_fact(ctx.text) if ctx.text else None
        if fact is None and entities.get("fact"):
            fact = sentence(to_second_person(str(entities["fact"])))
        if not fact:
            return FastOutcome("¿Qué quieres que recuerde?", Emotion.INTERROGATIVA)
        written = await self._memory.save_fact(fact, source="usuario", text=ctx.text or fact)
        if written.outcome == "duplicado":
            return FastOutcome("Eso ya lo tenía anotado.", Emotion.NEUTRAL)
        if written.outcome == "actualizado":
            return FastOutcome(f"Actualizado: {lower_first(written.fact.content)}")
        return FastOutcome(f"Anotado: {lower_first(written.fact.content)}")

    async def forget(self, entities: Entities, ctx: FastContext) -> FastOutcome:
        """Encuentra lo que hay sobre el tema y pide confirmacion antes de borrar."""
        topic = _topic(ctx, entities, _FORGET)
        if not topic:
            return FastOutcome("¿Qué quieres que olvide?", Emotion.INTERROGATIVA)
        if ctx.confirmations is None:
            return FastOutcome("No puedo pedir confirmación ahora; no borro nada.", Emotion.ERROR)
        found = await stage_forget(self._memory, topic, ctx.confirmations)
        if found == 0:
            return FastOutcome(f"No tengo nada guardado sobre {topic}.", Emotion.NEUTRAL)
        what = "un recuerdo" if found == 1 else f"{found} recuerdos"
        return FastOutcome(
            f"Tengo {what} sobre {topic}. ¿Lo olvido? Di: Jarvis, sí.", Emotion.INTERROGATIVA
        )

    async def about(self, entities: Entities, ctx: FastContext) -> FastOutcome:
        """Lo que se sabe de algo o alguien: relaciones del grafo y hechos."""
        topic = _topic(ctx, entities, _ABOUT)
        if not topic:
            return FastOutcome("¿De qué o de quién?", Emotion.INTERROGATIVA)
        items = await self._memory.about(topic)
        if not items:
            return FastOutcome(f"No sé nada de {topic} todavía.", Emotion.NEUTRAL)
        spoken = [lower_first(item.rstrip(".")) for item in items[:_SPOKEN_ITEMS]]
        listed = spoken[0] if len(spoken) == 1 else "; ".join(spoken[:-1]) + "; y " + spoken[-1]
        text = f"De {topic} sé esto: {listed}."
        if len(items) > _SPOKEN_ITEMS:
            text += f" Y {len(items) - _SPOKEN_ITEMS} cosas más."
        return FastOutcome(text, Emotion.NEUTRAL)

    async def timeline(self, entities: Entities, ctx: FastContext) -> FastOutcome:
        """Lo que paso en un periodo, a partir de los episodios."""
        start, end, label = period_bounds(str(entities.get("when") or ""), ctx.now)
        episodes = await asyncio.to_thread(
            self._memory.episodic.between, start, end, kinds=("turno", "resumen")
        )
        asked = [
            e.user_text for e in episodes
            if e.user_text and e.intent not in {"memory.timeline", "memory.forget"}
        ]  # fmt: skip
        if not asked:
            return FastOutcome(f"{label} no hablamos de nada.", Emotion.NEUTRAL)
        distinct = list(dict.fromkeys(clip(text.rstrip("?.!"), 50) for text in asked))
        quoted = [f"«{text}»" for text in distinct[:_TIMELINE_TOPICS]]
        if len(asked) == 1:
            return FastOutcome(f"{label} solo me pediste una cosa: {quoted[0]}.", Emotion.NEUTRAL)
        listed = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " y " + quoted[-1]
        return FastOutcome(
            f"{label} me pediste {len(asked)} cosas. Entre ellas: {listed}.", Emotion.NEUTRAL
        )

    async def learned(self, entities: Entities, ctx: FastContext) -> FastOutcome:
        """El reporte de la ultima consolidacion."""
        report = await asyncio.to_thread(self._memory.scheduler.last_report)
        if report is None:
            return FastOutcome("Todavía no he consolidado la memoria ninguna noche.")
        return FastOutcome(report.spoken(), Emotion.NEUTRAL)


async def stage_forget(
    memory: MemorySystem, topic: str, confirmations: PendingConfirmations
) -> int:
    """Busca todo lo que hay sobre un tema y deja el borrado esperando un "si".

    Abarca hechos (por significado y por palabra), episodios que lo mencionan,
    la entidad del grafo con ese nombre y los turnos en RAM. Devuelve cuantos
    recuerdos se borrarian; cero no deja nada pendiente.
    """
    found = await memory.find_forgettable(topic)
    if found.total == 0:
        return 0

    async def run() -> str:
        removed = await memory.forget(found, keywords(topic))
        # DECISION: la confirmacion no repite el tema. Lo que Jarvis dice queda
        # en la episodica, y repetirlo ahi haria que el tema se siguiera
        # encontrando justo despues de olvidarlo.
        what = "un recuerdo" if removed == 1 else f"{removed} recuerdos"
        return f"Olvidado. Borré {what}."

    confirmations.ask(f"olvidar {found.total} recuerdos", run)
    return found.total
