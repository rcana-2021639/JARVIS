"""El habla del pipeline: sintesis, amplitud para el HUD y barge-in.

Dice en orden lo que llega por el bus (`cognition.response` entera o
`cognition.response.chunk` frase a frase) o lo que se pida con `speak`. Nunca
habla encima del usuario: espera a que termine de hablar.

DECISION: el barge-in se dispara con la palabra clave, no con cualquier voz.
Sin cancelacion de eco, el VAD oye a Jarvis hablando por los altavoces y lo
tomaria por el usuario: se interrumpiria a si mismo. La palabra clave es
sagrada tambien aqui -- "Jarvis, para" funciona; toser no.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jarvis.contracts.cascade import ResponseChunk
from jarvis.contracts.cognition import Response
from jarvis.contracts.enums import CoreState, Emotion, EventType
from jarvis.contracts.voice import SpeechOutput, TtsAmplitude
from jarvis.logging import get_logger
from jarvis.voice.audio import frame_level

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jarvis.bus.base import Subscription
    from jarvis.contracts.base import CorrelationId
    from jarvis.contracts.events import Event
    from jarvis.voice.audio import AudioArray
    from jarvis.voice.runtime import PipelineRuntime

__all__ = ["VoiceSpeaker"]

_log = get_logger("jarvis.voice.pipeline")

#: Ganancia que lleva la energia RMS del habla a un rango util para el HUD.
#: La voz de Piper ronda 0.1-0.25 de RMS; sin ganancia la esfera apenas latiria.
_AMPLITUDE_GAIN = 3.0

#: Interacciones viejas cuyo habla pendiente se descarta. Acotado: solo importan
#: las ultimas, y un conjunto sin tope creceria mientras Jarvis viva.
_SILENCED_MEMORY = 64

#: Trozos de sintesis que pueden esperar a ser reproducidos. Piper produce una
#: frase por trozo, asi que esto es "hasta cuatro frases por delante".
_SYNTH_QUEUE = 4


@dataclass(slots=True)
class _Speaking:
    """Una respuesta hablada en curso."""

    text: str
    emotion: Emotion
    correlation_id: CorrelationId | None
    sample_rate: int
    played_samples: int = 0
    interrupted: bool = False
    stop: threading.Event = field(default_factory=threading.Event)

    @property
    def played_ms(self) -> float:
        return self.played_samples / self.sample_rate * 1000.0


class VoiceSpeaker:
    """Convierte texto en voz, de a una frase por vez y en orden."""

    def __init__(self, runtime: PipelineRuntime) -> None:
        """Args: runtime, el estado compartido del pipeline."""
        self._rt = runtime
        self._speak_lock = asyncio.Lock()
        self._speak_waiting = 0
        """Frases esperando turno: si hay, no se vuelve a DORMIDO entre una y otra."""
        self._speaking: _Speaking | None = None
        self._subscriptions: list[Subscription] = []
        self._spoken_cids: set[CorrelationId] = set()
        """Interacciones que ya mandaron algo a decir desde la ultima palabra clave."""
        self._silenced: deque[CorrelationId] = deque(maxlen=_SILENCED_MEMORY)
        """Interacciones superadas: lo que aun tengan por decir se descarta."""
        runtime.busy_checks.append(self.busy)

    @property
    def is_speaking(self) -> bool:
        """Si esta sonando algo ahora mismo."""
        return self._speaking is not None

    def busy(self) -> bool:
        """Si hay algo sonando o esperando turno."""
        return self._speaking is not None or self._speak_waiting > 0

    def subscribe(self) -> None:
        """Empieza a decir lo que la cascada cognitiva publique."""
        bus = self._rt.bus
        self._subscriptions = [
            bus.subscribe(
                EventType.COGNITION_RESPONSE, self._on_response, name="voice.speak_responses"
            ),
            bus.subscribe(
                EventType.COGNITION_RESPONSE_CHUNK, self._on_chunk, name="voice.speak_chunks"
            ),
        ]

    async def unsubscribe(self) -> None:
        """Deja de escuchar al bus."""
        for subscription in self._subscriptions:
            await subscription.unsubscribe()
        self._subscriptions = []

    def supersede(self) -> None:
        """Una orden nueva deja atras a las anteriores.

        Lo que aun tuvieran pendiente de decir (frases en cola, respuestas
        tardias) ya no se dice.
        """
        self._silenced.extend(self._spoken_cids)
        self._spoken_cids.clear()

    async def abort(self) -> None:
        """Al detener el pipeline: corta lo que este sonando."""
        if self._speaking is not None:
            self._speaking.interrupted = True
            self._speaking.stop.set()
            await self._rt.engines.sink.abort()

    async def speak(
        self,
        text: str,
        *,
        emotion: Emotion = Emotion.NEUTRAL,
        correlation_id: CorrelationId | None = None,
    ) -> SpeechOutput:
        """Dice algo en voz alta. Devuelve cuanto sono y si lo interrumpieron.

        Si el usuario esta hablando, espera a que termine: Jarvis no habla
        encima de nadie. Varias llamadas seguidas se dicen en orden.
        """
        rt = self._rt
        self._speak_waiting += 1
        rt.idle.clear()
        async with self._speak_lock:
            self._speak_waiting -= 1
            await rt.not_listening.wait()
            if correlation_id is not None and correlation_id in self._silenced:
                rt.refresh_idle()
                return SpeechOutput(
                    text=text, emotion=emotion, interrupted=True, correlation_id=correlation_id
                )
            requested = time.perf_counter()
            tts = rt.engines.tts
            session = _Speaking(
                text=text,
                emotion=emotion,
                correlation_id=correlation_id,
                sample_rate=tts.sample_rate,
            )
            self._speaking = session
            rt.idle.clear()
            await rt.set_state(CoreState.HABLANDO, "respuesta", correlation_id)
            await rt.emit(
                EventType.VOICE_TTS_STARTED,
                SpeechOutput(text=text, emotion=emotion, correlation_id=correlation_id),
                "voice.tts",
                correlation_id,
            )

            first_audio_ms: float | None = None
            block_samples = max(1, tts.sample_rate // rt.config().tts_amplitude_hz)
            sink = rt.engines.sink
            try:
                async for piece in self._synthesize(session):
                    for start in range(0, piece.shape[0], block_samples):
                        if session.interrupted:
                            break
                        block = piece[start : start + block_samples]
                        await sink.write(block, tts.sample_rate)
                        if first_audio_ms is None:
                            first_audio_ms = (time.perf_counter() - requested) * 1000.0
                        session.played_samples += block.shape[0]
                        rms, _ = frame_level(block)
                        await rt.emit(
                            EventType.VOICE_TTS_AMPLITUDE,
                            TtsAmplitude(
                                amplitude=min(1.0, rms * _AMPLITUDE_GAIN),
                                correlation_id=correlation_id,
                            ),
                            "voice.tts",
                            correlation_id,
                        )
                    if session.interrupted:
                        break
                if not session.interrupted:
                    await sink.drain()
            except Exception as error:
                _log.exception("voice.tts_failed")
                await rt.report_error("voice.tts", error, correlation_id)
            finally:
                session.stop.set()
                if self._speaking is session:
                    self._speaking = None

            output = SpeechOutput(
                text=text,
                emotion=emotion,
                duration_ms=session.played_ms,
                first_audio_ms=first_audio_ms,
                interrupted=session.interrupted,
                correlation_id=correlation_id,
            )
            await rt.emit(EventType.VOICE_TTS_FINISHED, output, "voice.tts", correlation_id)
            if rt.state is CoreState.HABLANDO and self._speak_waiting == 0:
                await rt.set_state(CoreState.DORMIDO, "fin_respuesta", correlation_id)
            rt.refresh_idle()
            return output

    async def interrupt(self) -> None:
        """Barge-in: calla al instante."""
        session = self._speaking
        if session is None:
            return
        session.interrupted = True
        session.stop.set()
        await self._rt.engines.sink.abort()
        await self._rt.emit(
            EventType.VOICE_BARGE_IN,
            SpeechOutput(
                text=session.text,
                emotion=session.emotion,
                duration_ms=session.played_ms,
                interrupted=True,
                correlation_id=session.correlation_id,
            ),
            "voice.tts",
            session.correlation_id,
        )
        _log.info("voice.barge_in", played_ms=round(session.played_ms))

    # ------------------------------------------------------------------

    async def _synthesize(self, session: _Speaking) -> AsyncIterator[AudioArray]:
        """Corre Piper en un hilo y entrega sus trozos en cuanto existen.

        Piper sintetiza frase por frase; asi la primera suena mientras la segunda
        todavia se esta generando, que es lo que mantiene bajo el tiempo hasta el
        primer audio en respuestas largas.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[AudioArray | BaseException | None] = asyncio.Queue(_SYNTH_QUEUE)
        tts = self._rt.engines.tts

        def produce() -> None:
            try:
                for piece in tts.synthesize(session.text):
                    if session.stop.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(piece), loop).result()
            except BaseException as error:
                asyncio.run_coroutine_threadsafe(queue.put(error), loop).result()
            finally:
                if not loop.is_closed():
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        producer = loop.run_in_executor(None, produce)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            session.stop.set()
            # Vacia la cola para que el productor no quede bloqueado en un put.
            while not producer.done():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                await asyncio.sleep(0.005)

    async def _on_response(self, event: Event) -> None:
        """Dice en voz alta lo que la cascada cognitiva conteste."""
        response = event.payload_as(Response)
        if response.streamed:
            # Ya se dijo frase a frase por cognition.response.chunk.
            return
        if response.should_speak and response.text:
            self._spoken_cids.add(response.correlation_id)
            self._rt.spawn(
                self.speak(
                    response.text,
                    emotion=response.emotion,
                    correlation_id=response.correlation_id,
                )
            )

    async def _on_chunk(self, event: Event) -> None:
        """Dice una frase de una respuesta que todavia se esta generando.

        `speak` las encola en orden por su candado: la segunda frase suena
        cuando termina la primera, mientras el modelo sigue con la tercera.
        """
        chunk = event.payload_as(ResponseChunk)
        if chunk.correlation_id in self._silenced:
            return
        self._spoken_cids.add(chunk.correlation_id)
        self._rt.spawn(
            self.speak(chunk.text, emotion=chunk.emotion, correlation_id=chunk.correlation_id)
        )
