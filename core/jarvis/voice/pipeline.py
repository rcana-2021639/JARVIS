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

    speak() / cognition.response / cognition.response.chunk
       v
    HABLANDO     Piper sintetiza frase a frase; se publica la amplitud a ~60 Hz
       |         sincronizada con lo que suena. La palabra clave lo interrumpe
       |         (barge-in) y se pasa directo a escuchar.
       v
    DORMIDO

Las piezas:

    runtime.py    estado compartido, cuando se esta ocioso, publicacion al bus
    listening.py  captura, palabra clave, VAD, plazos y transcripcion
    speaking.py   sintesis, amplitud, barge-in y lo que llega por el bus
    pipeline.py   esta fachada: las une y expone la API publica

DECISION: el pipeline no pasa a PENSANDO tras la transcripcion final. Ese estado
es de la cascada cognitiva (Fase 2), que es quien sabe cuando empieza y cuando
termina de pensar. El pipeline publica `voice.stt.final` y vuelve a dormir; si
nadie contesta, no se queda colgado en un estado que no le pertenece.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from jarvis.contracts.enums import Emotion
from jarvis.logging import get_logger
from jarvis.voice.listening import VoiceListener
from jarvis.voice.runtime import PipelineRuntime
from jarvis.voice.speaking import VoiceSpeaker

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.bus.base import EventBus
    from jarvis.config import VoiceConfig
    from jarvis.contracts.base import CorrelationId
    from jarvis.contracts.enums import CoreState
    from jarvis.contracts.voice import SpeechOutput
    from jarvis.voice.engines import VoiceEngines

__all__ = ["VoicePipeline"]

_log = get_logger(__name__)


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
            speak_responses: si dice en voz alta las `cognition.response` y los
                `cognition.response.chunk` que lleguen por el bus.
        """
        self._rt = PipelineRuntime(bus, engines, config)
        self._speaker = VoiceSpeaker(self._rt)
        self._listener = VoiceListener(self._rt, self._speaker)
        self._speak_responses = speak_responses
        self._run_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> CoreState:
        """Estado actual del pipeline."""
        return self._rt.state

    @property
    def source_finished(self) -> asyncio.Event:
        """Se activa cuando la fuente de audio se agota (fuentes finitas, pruebas)."""
        return self._listener.source_finished

    async def start(self) -> None:
        """Empieza a consumir el microfono. Idempotente."""
        if self._run_task is not None:
            return
        if self._speak_responses:
            self._speaker.subscribe()
        self._run_task = asyncio.create_task(self._listener.run(), name="voice.pipeline")
        _log.info("voice.started", sample_rate=self._rt.sample_rate, frame_ms=self._rt.frame_ms)

    async def stop(self) -> None:
        """Detiene todo: escucha, habla y dispositivos."""
        rt = self._rt
        rt.stopping = True
        await self._speaker.unsubscribe()
        await rt.engines.source.close()
        if self._run_task is not None:
            self._run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None
        await self._speaker.abort()
        await self._listener.cancel()
        for task in list(rt.tasks):
            task.cancel()
        await asyncio.gather(*rt.tasks, return_exceptions=True)
        await rt.engines.sink.close()
        _log.info("voice.stopped")

    async def wait_idle(self, timeout: float | None = None) -> bool:  # noqa: ASYNC109
        """Espera a que no haya escucha, transcripcion ni habla en curso.

        Como `EventBus.wait_idle`, vencer el plazo es una respuesta y no un error.
        """
        try:
            await asyncio.wait_for(self._rt.idle.wait(), timeout)
        except TimeoutError:
            return False
        return True

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
        return await self._speaker.speak(text, emotion=emotion, correlation_id=correlation_id)
