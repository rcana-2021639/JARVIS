"""Habla con Jarvis por el microfono. La demostracion de la Fase 1.

Escucha la palabra clave, transcribe lo que digas y te lo repite en voz alta.
Es un eco: entender y actuar es trabajo de la Fase 2. Lo que demuestra es el
pipeline completo con hardware real -- microfono, palabra clave, VAD, Whisper,
Piper, altavoces y barge-in -- y lo que publica al bus.

Uso:

    python scripts/voice_live.py                   # di "Hey Jarvis" y una orden
    python scripts/voice_live.py --list-devices    # que microfonos ve PortAudio
    python scripts/voice_live.py --seconds 20      # se cierra solo

Con `--verbose` imprime tambien el nivel del microfono y la amplitud del TTS.
Ctrl+C para salir.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.bus import InMemoryEventBus, OverflowPolicy
from jarvis.config_watcher import ConfigManager
from jarvis.contracts import (
    CascadeLevel,
    Event,
    EventType,
    Response,
    SpeechActivity,
    StateChange,
    Utterance,
    WakeDetection,
)
from jarvis.logging import bind_bus, configure_logging
from jarvis.voice import VoicePipeline
from jarvis.voice.factory import build_engines
from jarvis.voice.models import ModelMissingError


def _print_event(event: Event) -> None:
    kind = event.type
    if kind is EventType.CORE_STATE_CHANGED:
        change = event.payload_as(StateChange)
        print(f"  [{change.current.value:<10}] <- {change.reason}")
    elif kind is EventType.VOICE_WAKE_DETECTED:
        wake = event.payload_as(WakeDetection)
        extra = "  (interrumpiendo)" if wake.barge_in else ""
        print(f"  * palabra clave  {wake.score:.2f}{extra}")
    elif kind is EventType.VOICE_SPEECH_ENDED:
        reason = event.payload_as(SpeechActivity).reason
        print(f"  . fin de frase   {reason.value if reason else ''}")
    elif kind is EventType.VOICE_STT_PARTIAL:
        print(f"  ~ {event.payload_as(Utterance).text}")
    elif kind is EventType.VOICE_STT_FINAL:
        utterance = event.payload_as(Utterance)
        shown = utterance.text or "(nada)"
        print(f"  > {shown}   [{utterance.confidence:.2f}]")
    elif kind is EventType.CORE_ERROR:
        print(f"  ! error en {event.payload.get('module')}: {event.payload.get('detail')}")


async def run(args: argparse.Namespace) -> int:
    bus = InMemoryEventBus()
    manager = ConfigManager(bus=bus)
    config = manager.current
    configure_logging(level="WARNING" if not args.verbose else "INFO", development=True)

    try:
        engines = build_engines(config.voice)
    except ModelMissingError as error:
        print(error)
        return 2
    except RuntimeError as error:  # DeviceNotFoundError y fallos de PortAudio
        print(f"No se pudo abrir el audio: {error}")
        return 3

    mic = getattr(engines.source, "device_name", "?")
    speaker = getattr(engines.sink, "device_name", "?")
    print(f"Microfono: {mic}")
    print(f"Salida:    {speaker}")
    print(f"Palabra clave: '{config.voice.wake_model}' (umbral {config.voice.wake_threshold})")
    print('Di "Hey Jarvis" y una orden. Ctrl+C para salir.\n')

    async def show(event: Event) -> None:
        _print_event(event)

    async def echo(event: Event) -> None:
        """El sustituto de la Fase 2: repetir lo que se oyo."""
        utterance = event.payload_as(Utterance)
        text = f"Escuche: {utterance.text}" if utterance.text else "Te escucho."
        response = Response(
            text=text,
            handled_by_level=CascadeLevel.FAST_PATH,
            correlation_id=utterance.correlation_id,
        )
        await bus.publish(Event.of(EventType.COGNITION_RESPONSE, response, source="voice_live"))

    async with bus:
        bind_bus(bus)
        await manager.start()
        bus.subscribe("*", show, name="console", max_queue=4096)
        bus.subscribe(EventType.VOICE_STT_FINAL, echo, name="echo")
        if args.verbose:

            async def meter(event: Event) -> None:
                value = event.payload.get("rms", event.payload.get("amplitude", 0.0))
                bar = "#" * int(float(value) * 120)  # type: ignore[arg-type]
                print(f"    {event.type.value:<20} {bar}")

            bus.subscribe(
                "voice.*.amplitude",
                meter,
                name="meter_tts",
                max_queue=8,
                overflow=OverflowPolicy.DROP_OLDEST,
            )
            bus.subscribe(
                EventType.VOICE_LEVEL,
                meter,
                name="meter_mic",
                max_queue=8,
                overflow=OverflowPolicy.DROP_OLDEST,
            )

        pipeline = VoicePipeline(bus, engines, lambda: manager.current.voice)
        await pipeline.start()
        await pipeline.speak("Sistemas de voz en linea.")
        try:
            if args.seconds:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(pipeline.source_finished.wait(), args.seconds)
            else:
                await pipeline.source_finished.wait()
        finally:
            await pipeline.stop()
            await manager.stop()
            bind_bus(None)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list-devices", action="store_true", help="Listar dispositivos y salir.")
    parser.add_argument("--seconds", type=float, help="Cerrarse solo pasado este tiempo.")
    parser.add_argument("--verbose", action="store_true", help="Mostrar niveles y registro.")
    args = parser.parse_args()

    if args.list_devices:
        from jarvis.voice.adapters.sounddevice_io import describe_devices, resolve_device

        print(describe_devices())
        for kind in ("input", "output"):
            try:
                print(f"\nSe usaria para {kind}: #{resolve_device(None, kind=kind)}")
            except RuntimeError as error:
                print(f"\n{error}")
        return 0

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nHasta luego.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
