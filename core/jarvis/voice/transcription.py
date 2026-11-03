"""La transcripcion de una escucha: parciales en vivo y la final.

Whisper es el paso mas caro del pipeline (~0.9 s en esta CPU) y corre en un
hilo. Un solo candado garantiza que nunca haya dos transcripciones a la vez:
una parcial nunca le quita CPU a la final.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from jarvis.contracts.base import CorrelationId, utc_now
from jarvis.contracts.enums import EventType, Language
from jarvis.contracts.voice import Utterance
from jarvis.logging import get_logger, traced, user_text
from jarvis.voice.text import strip_wake_word

if TYPE_CHECKING:
    from datetime import datetime

    from jarvis.config import VoiceConfig
    from jarvis.voice.audio import AudioArray
    from jarvis.voice.engines import Transcription
    from jarvis.voice.runtime import PipelineRuntime
    from jarvis.voice.vad import EndpointDetector

__all__ = ["Listening", "Transcriber"]

_log = get_logger("jarvis.voice.pipeline")


@dataclass(slots=True, eq=False)
class Listening:
    """Una escucha en curso, de la palabra clave a la transcripcion final.

    `eq=False` a proposito: cada escucha es unica por identidad, y asi puede
    vivir en un conjunto mientras su transcripcion termina.
    """

    correlation_id: CorrelationId
    endpoint: EndpointDetector
    started_audio_ms: float
    started_at: datetime
    chunks: list[AudioArray] = field(default_factory=list)
    finalizing: bool = False
    language: Language | None = None
    last_partial_ms: float = 0.0

    def is_closing(self) -> bool:
        """Si la escucha ya se cerro.

        Es un metodo y no una lectura directa del campo a proposito: mypy
        estrecha `finalizing` a False tras comprobarlo y no ve que otra tarea
        puede cambiarlo durante un await.
        """
        return self.finalizing


class Transcriber:
    """Transcribe escuchas y publica el resultado al bus."""

    def __init__(self, runtime: PipelineRuntime) -> None:
        """Args: runtime, el estado compartido del pipeline."""
        self._rt = runtime
        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        """Si hay una transcripcion corriendo ahora mismo."""
        return self._lock.locked()

    @traced("voice.stt.partial")
    async def partial(self, listening: Listening, cfg: VoiceConfig) -> None:
        """Transcripcion en vivo. Si la final ya empezo, se descarta."""
        rt = self._rt
        async with self._lock:
            if listening.is_closing():
                return
            audio = np.concatenate(listening.chunks)
            result = await asyncio.to_thread(
                rt.engines.stt.transcribe, audio, language=listening.language
            )
        if listening.is_closing():
            return
        listening.language = result.language
        text = strip_wake_word(result.text, cfg.wake_word)
        await rt.emit(
            EventType.VOICE_STT_PARTIAL,
            Utterance(
                text=text,
                confidence=min(max(result.confidence, 0.0), 1.0),
                language=result.language,
                duration_ms=audio.shape[0] / rt.sample_rate * 1000.0,
                started_at=listening.started_at,
                ended_at=utc_now(),
                is_final=False,
                correlation_id=listening.correlation_id,
            ),
            "voice.stt",
            listening.correlation_id,
        )

    async def final(self, listening: Listening) -> str:
        """La transcripcion que alimenta al clasificador.

        Devuelve el motivo con que cerrar la escucha: `transcrito` o `error_stt`.
        """
        try:
            result = await self._run_final(listening)
        except Exception as error:
            _log.exception("voice.stt_failed")
            await self._rt.report_error("voice.stt", error, listening.correlation_id)
            return "error_stt"
        cfg = self._rt.config()
        await self.publish(
            listening,
            text=strip_wake_word(result.text, cfg.wake_word),
            confidence=result.confidence,
            language=result.language,
        )
        return "transcrito"

    @traced("voice.wake.stt")
    async def spot(self, audio: AudioArray, language: Language | None) -> Transcription:
        """Transcribe una frase para buscar la palabra clave al principio."""
        async with self._lock:
            return await asyncio.to_thread(
                self._rt.engines.stt.transcribe, audio, language=language
            )

    @traced("voice.stt.final")
    async def _run_final(self, listening: Listening) -> Transcription:
        audio = np.concatenate(listening.chunks)
        async with self._lock:
            return await asyncio.to_thread(
                self._rt.engines.stt.transcribe, audio, language=listening.language
            )

    async def publish(
        self,
        listening: Listening,
        *,
        text: str,
        confidence: float,
        language: Language | None,
    ) -> None:
        """Publica `voice.stt.final`. Texto vacio es valido: "Jarvis" y silencio."""
        samples = sum(chunk.shape[0] for chunk in listening.chunks)
        utterance = Utterance(
            text=text,
            confidence=min(max(confidence, 0.0), 1.0),
            language=language or Language.ES,
            duration_ms=samples / self._rt.sample_rate * 1000.0,
            started_at=listening.started_at,
            ended_at=max(utc_now(), listening.started_at),
            is_final=True,
            correlation_id=listening.correlation_id,
        )
        await self._rt.emit(
            EventType.VOICE_STT_FINAL, utterance, "voice.stt", listening.correlation_id
        )
        _log.info("voice.final", text=user_text(text), confidence=round(utterance.confidence, 3))
