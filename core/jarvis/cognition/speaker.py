"""El canal hablado de una respuesta en curso.

Recibe texto conforme el modelo lo genera, lo corta en frases y las publica
como `cognition.response.chunk` para que la voz las diga sin esperar al
final. Tambien lleva la cuenta de lo dicho: cuando salio la primera frase, si
se quedo texto sin decir (va al chat) y los avisos que se intercalaron.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from jarvis.cognition.streaming import SentenceChunker
from jarvis.contracts.cascade import ResponseChunk
from jarvis.contracts.enums import CascadeLevel, Emotion

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["ResponseStream"]

_OVERFLOW_NOTICE = "El resto está en el chat."


class ResponseStream:
    """Publica frases hablables y recuerda que se dijo."""

    def __init__(
        self,
        publish: Callable[[ResponseChunk], Awaitable[None]],
        correlation_id: str,
        *,
        max_sentences: int,
        started: float,
    ) -> None:
        """Args: publish, manda un trozo al bus; started, el reloj de la interaccion."""
        self._publish = publish
        self._cid = correlation_id
        self._max_sentences = max_sentences
        self._started = started
        self._chunker = SentenceChunker(max_sentences)
        self._index = 0
        self.level = CascadeLevel.LOCAL_LLM
        self.emotion = Emotion.NEUTRAL
        self.spoken: list[str] = []
        self.notices: list[str] = []
        """Avisos dichos antes o en lugar de la respuesta."""
        self.trailing: list[str] = []
        """Avisos que van despues de la respuesta (lo que quedo para el chat)."""
        self._answer: list[str] = []
        self.first_chunk_ms: float | None = None

    @property
    def any_spoken(self) -> bool:
        """Si ya salio algo por el canal hablado."""
        return bool(self.spoken)

    @property
    def answer_text(self) -> str:
        """Solo las frases de la respuesta que ya se dijeron, sin los avisos."""
        return " ".join(self._answer)

    @property
    def spoken_text(self) -> str:
        """Todo lo dicho, en orden."""
        return " ".join(self.spoken)

    def restart(self, level: CascadeLevel) -> None:
        """Empieza una respuesta nueva (tras una escalada) sin olvidar los avisos."""
        self.level = level
        self._chunker = SentenceChunker(self._max_sentences)

    async def feed(self, delta: str) -> None:
        """Texto nuevo del modelo."""
        for sentence in self._chunker.feed(delta):
            self._answer.append(sentence)
            await self._send(sentence)

    async def notice(
        self, text: str, emotion: Emotion = Emotion.NEUTRAL, *, after: bool = False
    ) -> None:
        """Un aviso hablado que no cuenta contra el tope de frases.

        Con `after`, en el texto final va despues de la respuesta (lo que paso
        a media frase); si no, antes (lo que se decidio antes de contestar).
        """
        (self.trailing if after else self.notices).append(text)
        await self._send(text, emotion)

    async def finish(self) -> None:
        """Fin de la generacion: lo pendiente, y el aviso si algo quedo sin decir."""
        for sentence in self._chunker.flush():
            self._answer.append(sentence)
            await self._send(sentence)
        if self._chunker.overflowed:
            self.trailing.append(_OVERFLOW_NOTICE)
            await self._send(_OVERFLOW_NOTICE)

    async def salvage(self) -> None:
        """Tras un corte: dice la ultima frase pendiente si quedo completa."""
        for sentence in self._chunker.salvage():
            self._answer.append(sentence)
            await self._send(sentence)

    async def _send(self, text: str, emotion: Emotion | None = None) -> None:
        if self.first_chunk_ms is None:
            self.first_chunk_ms = (time.perf_counter() - self._started) * 1000.0
        self.spoken.append(text)
        await self._publish(
            ResponseChunk(
                text=text,
                index=self._index,
                emotion=emotion or self.emotion,
                level=self.level,
                correlation_id=self._cid,
            )
        )
        self._index += 1
