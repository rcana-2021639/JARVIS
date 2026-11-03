"""El contexto que ve el modelo, con presupuesto estricto de tokens.

Prioridad, de lo que nunca se poda a lo primero que se poda:

    1. instruccion del sistema      nunca
    2. estado actual                hora, modo; viaja pegado a la orden
    3. memoria recuperada           la recuperacion hibrida de la Fase 3
    4. ultimos turnos               los mas recientes se conservan
    5. resumenes de lo viejo        una linea por turno ya expulsado

Cuando el total pasa del presupuesto se poda desde abajo: resumenes mas
viejos, luego turnos mas viejos, luego memoria, luego estado. La orden del
usuario y el sistema siempre entran; si aun asi no caben, el enrutador ve el
total y escala a un modelo con mas ventana.

DECISION: el estado va al final, junto a la orden, y no en el sistema. Asi el
prefijo (sistema + historial) es identico entre peticiones y Ollama reutiliza
su cache: en CPU, re-procesar el prefijo cuesta cientos de milisegundos.

DECISION: los tokens se estiman (caracteres / 3.5) en vez de contarse con el
tokenizador de cada modelo. Contar exacto costaria una llamada por peticion y
el presupuesto es un limite de seguridad, no una factura.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from jarvis.cognition.llm import ChatMessage
from jarvis.cognition.prompts import system_prompt
from jarvis.cognition.spoken import local_now, spoken_date

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jarvis.contracts.enums import CascadeLevel

__all__ = [
    "BuiltContext",
    "ContextBuilder",
    "ConversationHistory",
    "MemoryAccess",
    "PromptAssembler",
    "Recollection",
    "Turn",
    "estimate_tokens",
]

_CHARS_PER_TOKEN = 3.5
_MESSAGE_OVERHEAD = 4
"""Tokens que cada mensaje gasta en marcas de rol en las plantillas de chat."""
_SUMMARY_CHARS = 90
_MAX_SUMMARIES = 30


def estimate_tokens(text: str) -> int:
    """Estimacion barata de tokens para espanol."""
    return math.ceil(len(text) / _CHARS_PER_TOKEN) if text else 0


@dataclass(frozen=True, slots=True)
class Turn:
    """Un intercambio completo."""

    user: str
    assistant: str
    level: CascadeLevel
    at: datetime


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class ConversationHistory:
    """Historial en RAM: ultimos turnos completos y resumenes de los viejos.

    DECISION: los resumenes no usan un modelo. Cada turno expulsado se reduce a
    una linea recortada. Resumir con un LLM costaria segundos en esta CPU; la
    consolidacion de verdad es trabajo de la Fase 3.
    """

    def __init__(self, max_turns: int) -> None:
        """Args: max_turns, cuantos turnos completos se conservan."""
        self._turns: deque[Turn] = deque()
        self._summaries: deque[str] = deque(maxlen=_MAX_SUMMARIES)
        self._max_turns = max_turns

    def add(self, turn: Turn) -> None:
        """Agrega un turno. Al pasarse del tope, la mitad mas vieja pasa a resumen.

        DECISION: se expulsa por bloques y no de a uno. Expulsar un turno por
        orden cambia el principio del contexto en cada peticion y Ollama tiene
        que reprocesarlo entero: medido, 11-25 s en esta CPU. Por bloques, el
        prefijo solo cambia cada `max_turns / 2` turnos.
        """
        self._turns.append(turn)
        if len(self._turns) <= self._max_turns:
            return
        keep = max(1, self._max_turns // 2)
        while len(self._turns) > keep:
            old = self._turns.popleft()
            self._summaries.append(
                f"Usuario: {_clip(old.user, _SUMMARY_CHARS)} / "
                f"Jarvis: {_clip(old.assistant, _SUMMARY_CHARS)}"
            )

    def resize(self, max_turns: int) -> None:
        """Ajusta el tope en caliente."""
        self._max_turns = max_turns

    @property
    def turns(self) -> tuple[Turn, ...]:
        """Turnos completos, del mas viejo al mas nuevo."""
        return tuple(self._turns)

    @property
    def summaries(self) -> tuple[str, ...]:
        """Resumenes, del mas viejo al mas nuevo."""
        return tuple(self._summaries)

    def clear(self) -> None:
        """Vacia todo."""
        self._turns.clear()
        self._summaries.clear()


@dataclass(slots=True)
class BuiltContext:
    """El contexto listo para el modelo."""

    messages: list[ChatMessage]
    tokens: int
    dropped: list[str] = field(default_factory=list)
    """Que se podo, para el registro."""
    over_budget: bool = False
    """Si ni podando todo cupo: el enrutador lo usa para escalar."""


class ContextBuilder:
    """Arma la lista de mensajes respetando la prioridad y el presupuesto."""

    def build(
        self,
        *,
        system: str,
        user_text: str,
        budget_tokens: int,
        history: ConversationHistory,
        state: str = "",
        memory: Sequence[str] = (),
    ) -> BuiltContext:
        """Construye el contexto. El sistema y la orden entran siempre."""
        summaries = list(history.summaries)
        turns = list(history.turns)
        memory_items = list(memory)
        state_text = state
        dropped: list[str] = []

        def total() -> int:
            return sum(
                estimate_tokens(m.content) + _MESSAGE_OVERHEAD
                for m in self._assemble(
                    system, user_text, state_text, memory_items, turns, summaries
                )
            )

        while total() > budget_tokens:
            if summaries:
                summaries.pop(0)
                dropped.append("resumen")
            elif turns:
                turns.pop(0)
                dropped.append("turno")
            elif memory_items:
                memory_items.pop()
                dropped.append("memoria")
            elif state_text:
                state_text = ""
                dropped.append("estado")
            else:
                break

        messages = self._assemble(system, user_text, state_text, memory_items, turns, summaries)
        tokens = total()
        return BuiltContext(messages, tokens, dropped, over_budget=tokens > budget_tokens)

    @staticmethod
    def _assemble(
        system: str,
        user_text: str,
        state: str,
        memory: list[str],
        turns: list[Turn],
        summaries: list[str],
    ) -> list[ChatMessage]:
        system_text = system
        if summaries:
            system_text += "\n\nConversacion anterior, resumida:\n" + "\n".join(summaries)
        messages = [ChatMessage("system", system_text)]
        for turn in turns:
            messages.append(ChatMessage("user", turn.user))
            messages.append(ChatMessage("assistant", turn.assistant))
        preface: list[str] = []
        if memory:
            preface.append(
                "Lo que recuerdas y viene al caso:\n" + "\n".join(f"- {m}" for m in memory)
            )
        if state:
            preface.append(f"(Contexto, no lo menciones salvo que venga al caso: {state}.)")
        content = "\n".join([*preface, user_text]) if preface else user_text
        messages.append(ChatMessage("user", content))
        return messages


@dataclass(frozen=True, slots=True)
class Recollection:
    """Lo que la memoria aporta a una orden, ya cortado a cada presupuesto."""

    local: tuple[str, ...] = ()
    """Para el nivel 2: pocos tokens, que en CPU cada uno cuesta."""
    cloud: tuple[str, ...] = ()
    """Para el nivel 3: el presupuesto completo de recuperacion."""


class MemoryAccess(Protocol):
    """Lo que la cascada necesita de la memoria (Fase 3). La implementa `jarvis.memory`."""

    async def recollect(self, text: str) -> Recollection:
        """Recupera lo relevante para una orden. Nunca lanza: sin memoria, vacio."""
        ...


class PromptAssembler:
    """Arma el contexto de cada nivel: sistema, historial, memoria y estado."""

    def __init__(
        self,
        history: ConversationHistory,
        user_name: Callable[[], str],
        clock: Callable[[], datetime] | None,
    ) -> None:
        """Args: history, los turnos; user_name; clock, la hora (simulable)."""
        self.history = history
        self._user_name = user_name
        self._clock = clock
        self._builder = ContextBuilder()

    def build(
        self, text: str, *, structured: bool, budget: int, memory: Sequence[str] = ()
    ) -> BuiltContext:
        """El contexto de una orden para el nivel 2 (`structured`) o el 3."""
        now = self._clock() if self._clock else local_now()
        # DECISION: el nivel 2 no recibe estado. Medido: qwen3:4b lo repetia en
        # voz alta ("Modo HYBRID activado", "Hoy es lunes... 20:26"). Fecha y
        # hora las contesta el nivel 1; la nube si las recibe.
        state = "" if structured else f"{spoken_date(now)[:-1]}, {now:%H:%M}"
        return self._builder.build(
            system=system_prompt(self._user_name(), structured=structured),
            user_text=text,
            budget_tokens=budget,
            history=self.history,
            state=state,
            memory=memory,
        )
