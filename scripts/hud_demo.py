"""Alimenta el HUD con el nucleo real: estados y la voz de Jarvis por Piper.

Levanta el bus, el WebSocket del HUD y el pipeline de voz (solo para hablar) y
recorre en bucle los ocho estados. En HABLANDO, Piper dice una frase de verdad
por los altavoces y la esfera se deforma con las 32 bandas de esa voz. Sirve
para ver el HUD conectado al nucleo sin microfono ni modelos de lenguaje.

Uso:

    python scripts/hud_demo.py                 # hasta Ctrl+C
    python scripts/hud_demo.py --seconds 300   # el banco de 5 minutos

Y en otra terminal, dentro de hud/:  npm run electron
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.bus import InMemoryEventBus
from jarvis.config_watcher import ConfigManager
from jarvis.contracts import CoreState, Event, EventType, StateChange
from jarvis.hudlink import start_hud
from jarvis.logging import bind_bus, configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jarvis.contracts import Language
    from jarvis.voice.engines import Transcription

_PHRASES = (
    "Sistemas en línea. Todo en orden.",
    "Tu reunión de presupuesto es mañana a las diez.",
    "Encontré tres archivos del proyecto Aurora. Los dejo en el chat.",
    "Son las cinco y cuarto de la tarde.",
)
_SCRIPT: tuple[tuple[CoreState, float], ...] = (
    (CoreState.DORMIDO, 4.0),
    (CoreState.DESPIERTO, 0.9),
    (CoreState.ESCUCHANDO, 2.5),
    (CoreState.PENSANDO, 2.5),
    (CoreState.HABLANDO, 0.0),  # dura lo que dure la frase
    (CoreState.EJECUTANDO, 2.5),
    (CoreState.DORMIDO, 2.0),
    (CoreState.ERROR, 1.5),
    (CoreState.CONFIRMANDO, 3.0),
)


class _Silence:
    """Fuente muda: el pipeline solo se usa para hablar."""

    def __init__(self, sample_rate: int, frame_ms: int) -> None:
        self.sample_rate = sample_rate
        self._frame = np.zeros(sample_rate * frame_ms // 1000, dtype=np.float32)
        self._seconds = frame_ms / 1000
        self._closed = asyncio.Event()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while not self._closed.is_set():
            yield self._frame
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._closed.wait(), self._seconds)

    async def close(self) -> None:
        self._closed.set()


class _NoSTT:
    def transcribe(self, audio: np.ndarray, *, language: Language | None) -> Transcription:
        raise RuntimeError("hud_demo no transcribe")


async def run(seconds: float | None) -> int:
    from jarvis.voice import VoiceEngines, VoicePipeline
    from jarvis.voice.adapters.sounddevice_io import SpeakerSink
    from jarvis.voice.factory import NeverWake, build_tts
    from jarvis.voice.vad import EnergyVAD

    configure_logging(level="WARNING", development=True, log_file=None)
    bus = InMemoryEventBus()
    manager = ConfigManager(bus=bus)
    async with bus:
        bind_bus(bus)
        await manager.start()
        hud = await start_hud(bus, manager.current.ui)
        if hud is None:
            print("No se pudo abrir el puerto del HUD (ver el registro).")
            return 1
        voice = manager.current.voice
        engines = VoiceEngines(
            source=_Silence(voice.sample_rate, voice.frame_ms),
            wake=NeverWake(),
            vad=EnergyVAD(),
            stt=_NoSTT(),
            tts=build_tts(voice),
            sink=SpeakerSink(device=voice.output_device),
        )
        pipeline = VoicePipeline(bus, engines, lambda: manager.current.voice)
        await pipeline.start()
        print(f"HUD en ws://127.0.0.1:{hud.port}. Abre hud/ con: npm run electron")
        previous = CoreState.DORMIDO
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds if seconds else None
        phrase = 0
        try:
            while deadline is None or loop.time() < deadline:
                for state, duration in _SCRIPT:
                    if state is CoreState.HABLANDO:
                        # El pipeline publica HABLANDO y las amplitudes con bandas.
                        await pipeline.speak(_PHRASES[phrase % len(_PHRASES)])
                        await pipeline.wait_idle(30)
                        phrase += 1
                        previous = CoreState.DORMIDO
                        continue
                    change = StateChange(
                        previous=previous, current=state, reason="demo", source="hud_demo"
                    )
                    await bus.publish(
                        Event.of(EventType.CORE_STATE_CHANGED, change, source="hud_demo")
                    )
                    previous = state
                    await asyncio.sleep(duration)
        finally:
            await pipeline.stop()
            await hud.stop()
            await manager.stop()
            bind_bus(None)
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seconds", type=float, help="Cuanto tiempo correr. Sin esto, hasta Ctrl+C."
    )
    try:
        return asyncio.run(run(parser.parse_args().seconds))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
