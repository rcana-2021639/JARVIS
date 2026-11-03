"""El pipeline de voz: de la palabra clave a la transcripcion, y de vuelta al habla.

Ciclo de una interaccion:

    DORMIDO      el microfono llena el buffer circular; solo corre el detector
       |         de palabra clave. Nada sale de aqui.
       | wake
    DESPIERTO    nace el id de correlacion; se rescatan los 500 ms previos
       |
    ESCUCHANDO   el VAD decide cuando empieza y cuando termina la frase; se
       |         publican nivel a ~30 Hz y, si estan activas, parciales
       | silencio / tope / nadie hablo
       v
    (transcripcion final -> voice.stt.final) -> DORMIDO

    speak() / cognition.response
       v
    HABLANDO     Piper sintetiza frase a frase; se publica la amplitud a ~60 Hz
       |         sincronizada con lo que suena. La palabra clave lo interrumpe
       |         (barge-in) y se pasa directo a escuchar.
       v
    DORMIDO

DECISION: el pipeline no pasa a PENSANDO tras la transcripcion final. Ese estado
es de la cascada cognitiva (Fase 2), que es quien sabe cuando empieza y cuando
termina de pensar. El pipeline publica `voice.stt.final` y vuelve a dormir; si
nadie contesta, no se queda colgado en un estado que no le pertenece.

DECISION: el barge-in se dispara con la palabra clave, no con cualquier voz.
Sin cancelacion de eco, el VAD oye a Jarvis hablando por los altavoces y lo
tomaria por el usuario: se interrumpiria a si mismo. La palabra clave es
sagrada tambien aqui -- "Jarvis, para" funciona; toser no.

DECISION: el tiempo del pipeline se mide en audio procesado, no en reloj. Los
plazos (enfriamiento, nadie hablo, tope de escucha) cuentan milisegundos de
audio. Con el microfono es lo mismo que el reloj, y con una fuente sintetica
las pruebas son deterministas sin esperar de verdad.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis.config import VoiceConfig
from jarvis.contracts.base import BaseContract, CorrelationId, new_correlation_id, utc_now
from jarvis.contracts.cascade import ResponseChunk
from jarvis.contracts.cognition import Response
from jarvis.contracts.enums import CoreState, Emotion, EventType, Language, SpeechEndReason
from jarvis.contracts.events import Event
from jarvis.contracts.system import StateChange
from jarvis.contracts.voice import (
    AudioLevel,
    SpeechActivity,
    SpeechOutput,
    TtsAmplitude,
    Utterance,
    WakeDetection,
)
from jarvis.logging import clear_correlation_id, get_logger, set_correlation_id, traced
from jarvis.voice.audio import RingBuffer, frame_level
from jarvis.voice.text import strip_wake_word
from jarvis.voice.vad import EndpointDetector, EndpointEvent

if TYPE_CHECKING:
    import contextvars
    from collections.abc import AsyncIterator, Callable, Coroutine

    from jarvis.bus.base import EventBus, Subscription
    from jarvis.voice.audio import AudioArray
    from jarvis.voice.engines import Transcription, VoiceEngines

__all__ = ["VoicePipeline"]

_log = get_logger(__name__)

#: Ganancia que lleva la energia RMS del habla a un rango util para el HUD.
#: La voz de Piper ronda 0.1-0.25 de RMS; sin ganancia la esfera apenas latiria.
_AMPLITUDE_GAIN = 3.0

#: Interacciones viejas cuyo habla pendiente se descarta. Acotado: solo importan
#: las ultimas, y un conjunto sin tope creceria mientras Jarvis viva.
_SILENCED_MEMORY = 64

#: Trozos de sintesis que pueden esperar a ser reproducidos. Piper produce una
#: frase por trozo, asi que esto es "hasta cuatro frases por delante".
_SYNTH_QUEUE = 4


@dataclass(slots=True, eq=False)
class _Listening:
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


class VoicePipeline:
    """Orquesta los seis motores de voz y publica todo lo que pasa al bus."""

    def __init__(
        self,
        bus: EventBus,
        engines: VoiceEngines,
        config: VoiceConfig | Callable[[], VoiceConfig],
        *,
        speak_responses: bool = True,
    ) -> None:
        """Prepara el pipeline. No abre nada hasta `start`.

        Args:
            bus: donde se publica todo.
            engines: los motores ya construidos (`voice.factory.build_engines`).
            config: la seccion de voz, o una funcion que la devuelva. Con una
                funcion -- tipicamente `lambda: manager.current.voice` -- los
                umbrales se recargan en caliente: cada escucha nueva lee el valor
                vigente. Los modelos, en cambio, se cargan una vez.
            speak_responses: si dice en voz alta las `cognition.response` que
                lleguen por el bus.
        """
        self._bus = bus
        self._engines = engines
        self._config: Callable[[], VoiceConfig] = (
            (lambda: config) if isinstance(config, VoiceConfig) else config
        )
        self._speak_responses = speak_responses

        cfg = self._config()
        self._sample_rate = cfg.sample_rate
        self._frame_ms = cfg.frame_ms
        self._ring = RingBuffer(cfg.ring_buffer_seconds, cfg.sample_rate)

        self._state = CoreState.DORMIDO
        self._audio_ms = 0.0
        self._last_wake_ms = float("-inf")
        self._listening: _Listening | None = None
        """La escucha activa: la que recibe los bloques del microfono."""
        self._pending: set[_Listening] = set()
        """Escuchas ya cerradas cuya transcripcion final sigue en curso."""
        self._speaking: _Speaking | None = None
        self._correlation_token: contextvars.Token[CorrelationId | None] | None = None

        self._stt_lock = asyncio.Lock()
        self._speak_lock = asyncio.Lock()
        self._speak_waiting = 0
        """Frases esperando turno: si hay, no se vuelve a DORMIDO entre una y otra."""
        self._not_listening = asyncio.Event()
        self._not_listening.set()
        self._idle = asyncio.Event()
        self._idle.set()

        self._run_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[object]] = set()
        self._subscriptions: list[Subscription] = []
        self._spoken_cids: set[CorrelationId] = set()
        """Interacciones que ya mandaron algo a decir desde la ultima palabra clave."""
        self._silenced: deque[CorrelationId] = deque(maxlen=_SILENCED_MEMORY)
        """Interacciones superadas: lo que aun tengan por decir se descarta."""
        self._source_finished = asyncio.Event()
        self._stopping = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    @property
    def state(self) -> CoreState:
        """Estado actual del pipeline."""
        return self._state

    @property
    def source_finished(self) -> asyncio.Event:
        """Se activa cuando la fuente de audio se agota (fuentes finitas, pruebas)."""
        return self._source_finished

    async def start(self) -> None:
        """Empieza a consumir el microfono. Idempotente."""
        if self._run_task is not None:
            return
        if self._speak_responses:
            self._subscriptions = [
                self._bus.subscribe(
                    EventType.COGNITION_RESPONSE,
                    self._on_response,
                    name="voice.speak_responses",
                ),
                self._bus.subscribe(
                    EventType.COGNITION_RESPONSE_CHUNK,
                    self._on_chunk,
                    name="voice.speak_chunks",
                ),
            ]
        self._run_task = asyncio.create_task(self._run(), name="voice.pipeline")
        _log.info("voice.started", sample_rate=self._sample_rate, frame_ms=self._frame_ms)

    async def stop(self) -> None:
        """Detiene todo: escucha, habla y dispositivos."""
        self._stopping = True
        for subscription in self._subscriptions:
            await subscription.unsubscribe()
        self._subscriptions = []
        await self._engines.source.close()
        if self._run_task is not None:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None
        # El token pertenecia al contexto del bucle de captura, que acaba de
        # morir con el. Liberarlo desde aqui lanzaria ValueError.
        self._correlation_token = None
        if self._speaking is not None:
            self._speaking.interrupted = True
            self._speaking.stop.set()
            await self._engines.sink.abort()
        listening = self._listening
        if listening is not None:
            await self._end_listening(listening, SpeechEndReason.CANCELADA)
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._engines.sink.close()
        _log.info("voice.stopped")

    async def wait_idle(self, timeout: float | None = None) -> bool:  # noqa: ASYNC109
        """Espera a que no haya escucha, transcripcion ni habla en curso.

        Como `EventBus.wait_idle`, vencer el plazo es una respuesta y no un error.
        """
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
        except TimeoutError:
            return False
        return True

    # ------------------------------------------------------------------
    # Captura
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Bucle principal: un bloque de audio a la vez."""
        try:
            async for frame in self._engines.source.frames():
                await self._on_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log.exception("voice.capture_failed")
            await self._report_error("voice.capture", error)
        finally:
            self._source_finished.set()
        # La fuente se agoto (o el microfono murio) a media escucha. Sin esto la
        # escucha quedaria abierta para siempre: no llegan mas bloques que
        # cumplan ningun plazo. El fin del audio cuenta como silencio.
        listening = self._listening
        if listening is not None and not self._stopping:
            reason = (
                SpeechEndReason.SILENCIO
                if listening.endpoint.started
                else SpeechEndReason.SIN_HABLA
            )
            await self._end_listening(listening, reason)

    async def _on_frame(self, frame: AudioArray) -> None:
        """Procesa un bloque segun el estado."""
        self._audio_ms += self._frame_ms
        self._ring.append(frame)
        cfg = self._config()

        listening = self._listening
        if listening is not None:
            await self._listen_frame(listening, frame, cfg)
            return

        # Sin escucha activa se vigila la palabra clave: en reposo, mientras
        # habla (barge-in) y tambien mientras se transcribe la frase anterior,
        # para que un segundo "Jarvis" inmediato no se pierda.
        if self._state is CoreState.HABLANDO and not cfg.barge_in_enabled:
            return

        score = await self._score_wake(frame)
        if cfg.publish_idle_level and self._state is CoreState.DORMIDO:
            rms, peak = frame_level(frame)
            await self._emit(EventType.VOICE_LEVEL, AudioLevel(rms=rms, peak=peak), "voice.level")

        cooled = self._audio_ms - self._last_wake_ms >= cfg.wake_cooldown_s * 1000.0
        if score >= cfg.wake_threshold and cooled:
            await self._on_wake(score, cfg)

    @traced("voice.wake")
    async def _score_wake(self, frame: AudioArray) -> float:
        return await asyncio.to_thread(self._engines.wake.score, frame)

    @traced("voice.vad")
    async def _speech_probability(self, frame: AudioArray) -> float:
        return await asyncio.to_thread(self._engines.vad.probability, frame)

    # ------------------------------------------------------------------
    # Palabra clave
    # ------------------------------------------------------------------

    async def _on_wake(self, score: float, cfg: VoiceConfig) -> None:
        """La palabra clave se disparo: empieza una interaccion."""
        barge_in = self._speaking is not None
        if barge_in:
            await self._interrupt_speech()
        # Una orden nueva deja atras a las anteriores: lo que aun tuvieran
        # pendiente de decir (frases en cola, respuestas tardias) ya no se dice.
        self._silenced.extend(self._spoken_cids)
        self._spoken_cids.clear()

        correlation_id = new_correlation_id()
        if self._correlation_token is not None:
            clear_correlation_id(self._correlation_token)
        self._correlation_token = set_correlation_id(correlation_id)

        self._last_wake_ms = self._audio_ms
        self._engines.wake.reset()
        self._engines.vad.reset()
        preroll = self._ring.last(cfg.wake_preroll_ms)
        preroll_ms = preroll.shape[0] / self._sample_rate * 1000.0

        self._idle.clear()
        self._not_listening.clear()
        await self._set_state(CoreState.DESPIERTO, "barge_in" if barge_in else "wake")
        await self._emit(
            EventType.VOICE_WAKE_DETECTED,
            WakeDetection(
                keyword=self._engines.wake.keyword,
                score=min(max(score, 0.0), 1.0),
                preroll_ms=round(preroll_ms),
                barge_in=barge_in,
                correlation_id=correlation_id,
            ),
            "voice.wake",
            correlation_id,
        )
        _log.info("voice.wake", score=round(score, 3), barge_in=barge_in)

        self._listening = _Listening(
            correlation_id=correlation_id,
            endpoint=EndpointDetector(
                threshold=cfg.vad_threshold,
                frame_ms=float(cfg.frame_ms),
                min_speech_ms=float(cfg.vad_min_speech_ms),
                silence_ms=float(cfg.vad_silence_ms),
            ),
            started_audio_ms=self._audio_ms,
            started_at=utc_now() - timedelta(milliseconds=preroll_ms),
            chunks=[preroll],
            language=None if cfg.stt_language == "auto" else Language(cfg.stt_language),
        )
        await self._set_state(CoreState.ESCUCHANDO, "wake")

    # ------------------------------------------------------------------
    # Escucha
    # ------------------------------------------------------------------

    async def _listen_frame(
        self, listening: _Listening, frame: AudioArray, cfg: VoiceConfig
    ) -> None:
        """Un bloque durante la escucha: VAD, nivel, plazos y parciales."""
        listening.chunks.append(frame)
        probability = await self._speech_probability(frame)
        rms, peak = frame_level(frame)
        cid = listening.correlation_id
        await self._emit(
            EventType.VOICE_LEVEL,
            AudioLevel(
                rms=rms,
                peak=peak,
                speech_probability=min(max(probability, 0.0), 1.0),
                correlation_id=cid,
            ),
            "voice.level",
            cid,
        )

        event = listening.endpoint.update(probability)
        elapsed_ms = self._audio_ms - listening.started_audio_ms

        if event is EndpointEvent.STARTED:
            await self._emit(
                EventType.VOICE_SPEECH_STARTED,
                SpeechActivity(duration_ms=0.0, correlation_id=cid),
                "voice.vad",
                cid,
            )
        elif event is EndpointEvent.ENDED:
            await self._end_listening(listening, SpeechEndReason.SILENCIO)
        elif not listening.endpoint.started and elapsed_ms >= cfg.speech_start_timeout_s * 1000.0:
            await self._end_listening(listening, SpeechEndReason.SIN_HABLA)
        elif elapsed_ms >= cfg.max_listen_s * 1000.0:
            await self._end_listening(listening, SpeechEndReason.TOPE)
        elif (
            cfg.stt_partials
            and listening.endpoint.started
            and elapsed_ms - listening.last_partial_ms >= cfg.stt_partial_interval_ms
            and not self._stt_lock.locked()
        ):
            listening.last_partial_ms = elapsed_ms
            self._spawn(self._transcribe_partial(listening, cfg))

    async def _end_listening(self, listening: _Listening, reason: SpeechEndReason) -> None:
        """Cierra la escucha y lanza la transcripcion final.

        La escucha se suelta de inmediato: el microfono vuelve a vigilar la
        palabra clave mientras la transcripcion corre aparte. El estado sigue en
        ESCUCHANDO hasta que la final se publica, para que el HUD no muestre a
        Jarvis dormido con una orden a medio entender.
        """
        listening.finalizing = True
        if self._listening is listening:
            self._listening = None
        self._pending.add(listening)
        cid = listening.correlation_id
        try:
            await self._emit(
                EventType.VOICE_SPEECH_ENDED,
                SpeechActivity(
                    duration_ms=listening.endpoint.speech_ms,
                    reason=reason,
                    correlation_id=cid,
                ),
                "voice.vad",
                cid,
            )
            _log.info(
                "voice.speech_ended",
                reason=reason.value,
                speech_ms=listening.endpoint.speech_ms,
            )
            if reason is SpeechEndReason.CANCELADA:
                await self._close_listening(listening, "cancelada")
            elif reason is SpeechEndReason.SIN_HABLA:
                await self._publish_final(listening, text="", confidence=0.0, language=None)
                await self._close_listening(listening, "sin_habla")
            else:
                # La tarea final hereda una copia del contexto, con el id de
                # correlacion de esta interaccion.
                self._spawn(self._transcribe_final(listening))
        finally:
            # Lo que el microfono haga a partir de ahora ya no pertenece a esta
            # orden. Se libera aqui, en el contexto del bucle de captura, que es
            # el unico donde el token es valido.
            if self._correlation_token is not None:
                clear_correlation_id(self._correlation_token)
                self._correlation_token = None

    async def _close_listening(self, listening: _Listening, reason: str) -> None:
        """Vuelve a dormir tras una escucha, salvo que ya haya empezado otra."""
        self._pending.discard(listening)
        if self._listening is None:
            self._not_listening.set()
            if self._state is CoreState.ESCUCHANDO:
                await self._set_state(CoreState.DORMIDO, reason, listening.correlation_id)
        self._refresh_idle()

    # ------------------------------------------------------------------
    # Transcripcion
    # ------------------------------------------------------------------

    @traced("voice.stt.partial")
    async def _transcribe_partial(self, listening: _Listening, cfg: VoiceConfig) -> None:
        """Transcripcion en vivo. Si la final ya empezo, se descarta."""
        async with self._stt_lock:
            if listening.is_closing():
                return
            audio = np.concatenate(listening.chunks)
            result = await asyncio.to_thread(
                self._engines.stt.transcribe, audio, language=listening.language
            )
        if listening.is_closing():
            return
        listening.language = result.language
        text = strip_wake_word(result.text, cfg.wake_word)
        await self._emit(
            EventType.VOICE_STT_PARTIAL,
            Utterance(
                text=text,
                confidence=min(max(result.confidence, 0.0), 1.0),
                language=result.language,
                duration_ms=audio.shape[0] / self._sample_rate * 1000.0,
                started_at=listening.started_at,
                ended_at=utc_now(),
                is_final=False,
                correlation_id=listening.correlation_id,
            ),
            "voice.stt",
            listening.correlation_id,
        )

    async def _transcribe_final(self, listening: _Listening) -> None:
        """La transcripcion que alimenta al clasificador."""
        try:
            result = await self._run_final_stt(listening)
        except Exception as error:
            _log.exception("voice.stt_failed")
            await self._report_error("voice.stt", error, listening.correlation_id)
            await self._close_listening(listening, "error_stt")
            return
        cfg = self._config()
        await self._publish_final(
            listening,
            text=strip_wake_word(result.text, cfg.wake_word),
            confidence=result.confidence,
            language=result.language,
        )
        await self._close_listening(listening, "transcrito")

    @traced("voice.stt.final")
    async def _run_final_stt(self, listening: _Listening) -> Transcription:
        audio = np.concatenate(listening.chunks)
        async with self._stt_lock:
            return await asyncio.to_thread(
                self._engines.stt.transcribe, audio, language=listening.language
            )

    async def _publish_final(
        self,
        listening: _Listening,
        *,
        text: str,
        confidence: float,
        language: Language | None,
    ) -> None:
        samples = sum(chunk.shape[0] for chunk in listening.chunks)
        utterance = Utterance(
            text=text,
            confidence=min(max(confidence, 0.0), 1.0),
            language=language or Language.ES,
            duration_ms=samples / self._sample_rate * 1000.0,
            started_at=listening.started_at,
            ended_at=max(utc_now(), listening.started_at),
            is_final=True,
            correlation_id=listening.correlation_id,
        )
        await self._emit(
            EventType.VOICE_STT_FINAL, utterance, "voice.stt", listening.correlation_id
        )
        _log.info("voice.final", text=text, confidence=round(utterance.confidence, 3))

    # ------------------------------------------------------------------
    # Habla
    # ------------------------------------------------------------------

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
        self._speak_waiting += 1
        self._idle.clear()
        async with self._speak_lock:
            self._speak_waiting -= 1
            await self._not_listening.wait()
            if correlation_id is not None and correlation_id in self._silenced:
                self._refresh_idle()
                return SpeechOutput(
                    text=text, emotion=emotion, interrupted=True, correlation_id=correlation_id
                )
            requested = time.perf_counter()
            tts = self._engines.tts
            session = _Speaking(
                text=text,
                emotion=emotion,
                correlation_id=correlation_id,
                sample_rate=tts.sample_rate,
            )
            self._speaking = session
            self._idle.clear()
            await self._set_state(CoreState.HABLANDO, "respuesta", correlation_id)
            await self._emit(
                EventType.VOICE_TTS_STARTED,
                SpeechOutput(text=text, emotion=emotion, correlation_id=correlation_id),
                "voice.tts",
                correlation_id,
            )

            first_audio_ms: float | None = None
            cfg = self._config()
            block_samples = max(1, tts.sample_rate // cfg.tts_amplitude_hz)
            sink = self._engines.sink
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
                        await self._emit(
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
                await self._report_error("voice.tts", error, correlation_id)
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
            await self._emit(EventType.VOICE_TTS_FINISHED, output, "voice.tts", correlation_id)
            if self._state is CoreState.HABLANDO and self._speak_waiting == 0:
                await self._set_state(CoreState.DORMIDO, "fin_respuesta", correlation_id)
            self._refresh_idle()
            return output

    async def _synthesize(self, session: _Speaking) -> AsyncIterator[AudioArray]:
        """Corre Piper en un hilo y entrega sus trozos en cuanto existen.

        Piper sintetiza frase por frase; asi la primera suena mientras la segunda
        todavia se esta generando, que es lo que mantiene bajo el tiempo hasta el
        primer audio en respuestas largas.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[AudioArray | BaseException | None] = asyncio.Queue(_SYNTH_QUEUE)

        def produce() -> None:
            try:
                for piece in self._engines.tts.synthesize(session.text):
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

    async def _interrupt_speech(self) -> None:
        """Barge-in: calla al instante."""
        session = self._speaking
        if session is None:
            return
        session.interrupted = True
        session.stop.set()
        await self._engines.sink.abort()
        await self._emit(
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

    async def _on_response(self, event: Event) -> None:
        """Dice en voz alta lo que la cascada cognitiva conteste."""
        response = event.payload_as(Response)
        if response.streamed:
            # Ya se dijo frase a frase por cognition.response.chunk.
            return
        if response.should_speak and response.text:
            self._spoken_cids.add(response.correlation_id)
            self._spawn(
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
        self._spawn(
            self.speak(chunk.text, emotion=chunk.emotion, correlation_id=chunk.correlation_id)
        )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _spawn(self, coroutine: Coroutine[Any, Any, object]) -> asyncio.Task[object]:
        """Lanza una tarea y la retiene hasta que termine.

        asyncio solo guarda referencias debiles a las tareas: sin este conjunto,
        una transcripcion en curso podria desaparecer a medias.
        """
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _refresh_idle(self) -> None:
        if (
            self._listening is None
            and self._speaking is None
            and not self._pending
            and self._speak_waiting == 0
        ):
            self._idle.set()

    async def _set_state(
        self,
        state: CoreState,
        reason: str,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        previous = self._state
        if previous is state:
            return
        self._state = state
        cid = correlation_id
        if cid is None and self._listening is not None:
            cid = self._listening.correlation_id
        await self._emit(
            EventType.CORE_STATE_CHANGED,
            StateChange(
                previous=previous,
                current=state,
                reason=reason,
                source="voice.pipeline",
                correlation_id=cid,
            ),
            "voice.pipeline",
            cid,
        )

    async def _emit(
        self,
        event_type: EventType,
        payload: BaseContract,
        source: str,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        await self._bus.publish(
            Event.of(event_type, payload, source=source, correlation_id=correlation_id)
        )

    async def _report_error(
        self,
        module: str,
        error: BaseException,
        correlation_id: CorrelationId | None = None,
    ) -> None:
        """Nada falla en silencio: todo fallo llega al bus como `core.error`."""
        await self._bus.publish(
            Event(
                type=EventType.CORE_ERROR,
                payload={"module": module, "error": type(error).__name__, "detail": str(error)},
                source=module,
                correlation_id=correlation_id,
            )
        )
