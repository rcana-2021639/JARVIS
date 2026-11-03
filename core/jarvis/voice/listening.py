"""La escucha del pipeline: del microfono a la transcripcion final.

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

DECISION: el tiempo se mide en audio procesado, no en reloj. Los plazos
(enfriamiento, nadie hablo, tope de escucha) cuentan milisegundos de audio.
Con el microfono es lo mismo que el reloj, y con una fuente sintetica las
pruebas son deterministas sin esperar de verdad.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

from jarvis.contracts.base import CorrelationId, new_correlation_id, utc_now
from jarvis.contracts.enums import CoreState, EventType, Language, SpeechEndReason
from jarvis.contracts.voice import AudioLevel, SpeechActivity, WakeDetection
from jarvis.logging import (
    clear_correlation_id,
    get_logger,
    set_correlation_id,
    traced,
    user_text,
)
from jarvis.voice.audio import RingBuffer, frame_level
from jarvis.voice.spotting import PhraseSegmenter
from jarvis.voice.text import split_wake_word
from jarvis.voice.transcription import Listening, Transcriber
from jarvis.voice.vad import EndpointDetector, EndpointEvent

if TYPE_CHECKING:
    import contextvars

    from jarvis.config import VoiceConfig
    from jarvis.voice.audio import AudioArray
    from jarvis.voice.runtime import PipelineRuntime
    from jarvis.voice.speaking import VoiceSpeaker

__all__ = ["VoiceListener"]

_log = get_logger("jarvis.voice.pipeline")


class VoiceListener:
    """Consume el microfono, vigila la palabra clave y transcribe."""

    def __init__(self, runtime: PipelineRuntime, speaker: VoiceSpeaker) -> None:
        """Args: runtime, el estado compartido; speaker, para el barge-in."""
        self._rt = runtime
        self._speaker = speaker
        cfg = runtime.config()
        self._ring = RingBuffer(cfg.ring_buffer_seconds, cfg.sample_rate)
        self._audio_ms = 0.0
        self._last_wake_ms = float("-inf")
        self._listening: Listening | None = None
        """La escucha activa: la que recibe los bloques del microfono."""
        self._pending: set[Listening] = set()
        """Escuchas ya cerradas cuya transcripcion final sigue en curso."""
        self._correlation_token: contextvars.Token[CorrelationId | None] | None = None
        self._stt = Transcriber(runtime)
        self._segmenter = PhraseSegmenter(cfg)
        self.source_finished = asyncio.Event()
        """Se activa cuando la fuente de audio se agota (fuentes finitas, pruebas)."""
        runtime.busy_checks.append(self.busy)
        runtime.active_correlation_id = self.active_correlation_id

    def busy(self) -> bool:
        """Si hay una escucha abierta o una transcripcion en curso."""
        return self._listening is not None or bool(self._pending)

    def active_correlation_id(self) -> CorrelationId | None:
        """La interaccion que se esta escuchando, si hay una."""
        return self._listening.correlation_id if self._listening is not None else None

    async def run(self) -> None:
        """Bucle principal: un bloque de audio a la vez."""
        rt = self._rt
        try:
            async for frame in rt.engines.source.frames():
                await self._on_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _log.exception("voice.capture_failed")
            await rt.report_error("voice.capture", error)
        finally:
            self.source_finished.set()
        # La fuente se agoto (o el microfono murio) a media escucha. Sin esto la
        # escucha quedaria abierta para siempre: no llegan mas bloques que
        # cumplan ningun plazo. El fin del audio cuenta como silencio.
        listening = self._listening
        if listening is not None and not rt.stopping:
            reason = (
                SpeechEndReason.SILENCIO
                if listening.endpoint.started
                else SpeechEndReason.SIN_HABLA
            )
            await self._end_listening(listening, reason)

    async def cancel(self) -> None:
        """Al detener el pipeline: cierra la escucha abierta sin transcribirla."""
        # El token pertenecia al contexto del bucle de captura, que acaba de
        # morir con el. Liberarlo desde aqui lanzaria ValueError.
        self._correlation_token = None
        listening = self._listening
        if listening is not None:
            await self._end_listening(listening, SpeechEndReason.CANCELADA)

    # ------------------------------------------------------------------
    # Captura y palabra clave
    # ------------------------------------------------------------------

    async def _on_frame(self, frame: AudioArray) -> None:
        """Procesa un bloque segun el estado."""
        rt = self._rt
        self._audio_ms += rt.frame_ms
        self._ring.append(frame)
        cfg = rt.config()

        listening = self._listening
        if listening is not None:
            await self._listen_frame(listening, frame, cfg)
            return

        # Sin escucha activa se vigila la palabra clave: en reposo, mientras
        # habla (barge-in) y tambien mientras se transcribe la frase anterior,
        # para que un segundo "Jarvis" inmediato no se pierda.
        if rt.state is CoreState.HABLANDO and not cfg.barge_in_enabled:
            return
        if cfg.wake_engine == "whisper":
            await self._spot_frame(frame, cfg)
            return

        score = await self._score_wake(frame)
        if cfg.publish_idle_level and rt.state is CoreState.DORMIDO:
            rms, peak = frame_level(frame)
            await rt.emit(EventType.VOICE_LEVEL, AudioLevel(rms=rms, peak=peak), "voice.level")

        cooled = self._audio_ms - self._last_wake_ms >= cfg.wake_cooldown_s * 1000.0
        if score >= cfg.wake_threshold and cooled:
            await self._on_wake(score, cfg)

    @traced("voice.wake")
    async def _score_wake(self, frame: AudioArray) -> float:
        return await asyncio.to_thread(self._rt.engines.wake.score, frame)

    @traced("voice.vad")
    async def _speech_probability(self, frame: AudioArray) -> float:
        return await asyncio.to_thread(self._rt.engines.vad.probability, frame)

    async def _spot_frame(self, frame: AudioArray, cfg: VoiceConfig) -> None:
        """Modo whisper: corta frases con el VAD y manda cada una a revisar."""
        probability = await self._speech_probability(frame)
        phrase = self._segmenter.feed(frame, probability, self._ring)
        if phrase is not None:
            # En otra tarea: el microfono no se detiene mientras Whisper piensa.
            self._rt.spawn(self._check_phrase(phrase, cfg))

    async def _check_phrase(self, phrase: AudioArray, cfg: VoiceConfig) -> None:
        """Transcribe una frase; si empieza con la palabra clave, la atiende."""
        language = None if cfg.stt_language == "auto" else Language(cfg.stt_language)
        try:
            result = await self._stt.spot(phrase, language)
        except Exception as error:
            _log.exception("voice.spot_failed")
            await self._rt.report_error("voice.wake", error)
            return
        command = split_wake_word(result.text, cfg.wake_word)
        if command is None:
            # Sin palabra clave no ocurre nada: ni evento, ni texto en el registro.
            _log.debug("voice.phrase_ignored", text=user_text(result.text))
            return
        if self._listening is not None:
            return
        listening = await self._on_wake(1.0, cfg, bind_context=False)
        if not command:
            return  # solo "Jarvis": la escucha queda abierta esperando la orden
        # "Jarvis, que hora es" de corrido: esta misma transcripcion es la final.
        listening.chunks = [phrase]
        listening.language = result.language
        listening.finalizing = True
        self._listening = None
        self._pending.add(listening)
        await self._stt.publish(
            listening, text=command, confidence=result.confidence, language=result.language
        )
        await self._close_listening(listening, "transcrito")

    async def _on_wake(
        self, score: float, cfg: VoiceConfig, *, bind_context: bool = True
    ) -> Listening:
        """La palabra clave se disparo: empieza una interaccion.

        `bind_context=False` cuando lo llama una tarea aparte (modo whisper): el
        token del id de correlacion solo vale en el contexto del bucle de captura.
        """
        rt = self._rt
        barge_in = self._speaker.is_speaking
        if barge_in:
            await self._speaker.interrupt()
        self._speaker.supersede()

        correlation_id = new_correlation_id()
        if bind_context:
            if self._correlation_token is not None:
                clear_correlation_id(self._correlation_token)
            self._correlation_token = set_correlation_id(correlation_id)

        self._last_wake_ms = self._audio_ms
        rt.engines.wake.reset()
        rt.engines.vad.reset()
        self._segmenter.reset(cfg)
        preroll = self._ring.last(cfg.wake_preroll_ms)
        preroll_ms = preroll.shape[0] / rt.sample_rate * 1000.0

        rt.idle.clear()
        rt.not_listening.clear()
        await rt.set_state(CoreState.DESPIERTO, "barge_in" if barge_in else "wake")
        await rt.emit(
            EventType.VOICE_WAKE_DETECTED,
            WakeDetection(
                keyword=cfg.wake_word if cfg.wake_engine == "whisper" else rt.engines.wake.keyword,
                score=min(max(score, 0.0), 1.0),
                preroll_ms=round(preroll_ms),
                barge_in=barge_in,
                correlation_id=correlation_id,
            ),
            "voice.wake",
            correlation_id,
        )
        _log.info("voice.wake", score=round(score, 3), barge_in=barge_in)

        listening = Listening(
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
        self._listening = listening
        await rt.set_state(CoreState.ESCUCHANDO, "wake")
        return listening

    # ------------------------------------------------------------------
    # Escucha
    # ------------------------------------------------------------------

    async def _listen_frame(
        self, listening: Listening, frame: AudioArray, cfg: VoiceConfig
    ) -> None:
        """Un bloque durante la escucha: VAD, nivel, plazos y parciales."""
        rt = self._rt
        listening.chunks.append(frame)
        probability = await self._speech_probability(frame)
        rms, peak = frame_level(frame)
        cid = listening.correlation_id
        await rt.emit(
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
            await rt.emit(
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
            and not self._stt.busy
        ):
            listening.last_partial_ms = elapsed_ms
            rt.spawn(self._stt.partial(listening, cfg))

    async def _end_listening(self, listening: Listening, reason: SpeechEndReason) -> None:
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
            await self._rt.emit(
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
                await self._stt.publish(listening, text="", confidence=0.0, language=None)
                await self._close_listening(listening, "sin_habla")
            else:
                # La tarea final hereda una copia del contexto, con el id de
                # correlacion de esta interaccion.
                self._rt.spawn(self._transcribe_final(listening))
        finally:
            # Lo que el microfono haga a partir de ahora ya no pertenece a esta
            # orden. Se libera aqui, en el contexto del bucle de captura, que es
            # el unico donde el token es valido.
            if self._correlation_token is not None:
                clear_correlation_id(self._correlation_token)
                self._correlation_token = None

    async def _close_listening(self, listening: Listening, reason: str) -> None:
        """Vuelve a dormir tras una escucha, salvo que ya haya empezado otra."""
        rt = self._rt
        self._pending.discard(listening)
        if self._listening is None:
            rt.not_listening.set()
            if rt.state is CoreState.ESCUCHANDO:
                await rt.set_state(CoreState.DORMIDO, reason, listening.correlation_id)
        rt.refresh_idle()

    async def _transcribe_final(self, listening: Listening) -> None:
        reason = await self._stt.final(listening)
        await self._close_listening(listening, reason)
