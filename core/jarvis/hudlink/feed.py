"""Del bus al HUD: la vista compacta de lo que pasa, lista para dibujar.

El HUD no necesita cada evento: necesita el estado del nucleo, el audio (nivel
y 32 bandas, entrando o saliendo) y la carga cognitiva. Este modulo traduce los
eventos del bus a esos cuatro mensajes JSON (ver `hud/src/services/protocol.ts`).

    core.state.changed    -> {"type": "state", ...}
    voice.level           -> {"type": "audio", "source": "mic", ...}
    voice.tts.amplitude   -> {"type": "audio", "source": "tts", ...}
    cognition.routed      -> {"type": "cognition", "load": ...}
    cognition.completed   -> {"type": "cognition", "load": ~0}

DECISION: el audio se suscribe con cola corta y DROP_OLDEST. Si el HUD o el
bus se atrasan, se pierde un cuadro de amplitud y nadie lo nota; frenar al
pipeline de voz por un dibujo seria inaceptable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.bus.base import OverflowPolicy
from jarvis.contracts.cascade import RouteDecision
from jarvis.contracts.enums import CascadeLevel, CoreState, EventType
from jarvis.contracts.system import StateChange
from jarvis.contracts.voice import AudioLevel, TtsAmplitude
from jarvis.voice.spectrum import BANDS

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.bus.base import EventBus, Subscription
    from jarvis.contracts.events import Event

__all__ = ["PROTOCOL_VERSION", "HudFeed"]

PROTOCOL_VERSION = 1
_MIC_GAIN = 4.0
"""El RMS de una voz al microfono ronda 0.05-0.2: se escala para que llene 0..1."""
_AUDIO_QUEUE = 8
_LOAD_BY_LEVEL = {
    CascadeLevel.FAST_PATH: 0.2,
    CascadeLevel.LOCAL_LLM: 0.7,
    CascadeLevel.CLOUD_LLM: 1.0,
}
_THINKING_LOAD = 0.5
_IDLE_LOAD = 0.05

Message = dict[str, Any]


class HudFeed:
    """Traduce eventos del bus a mensajes del HUD y se los pasa a `send`."""

    def __init__(self, bus: EventBus, send: Callable[[Message], None]) -> None:
        """Args: bus, de donde se lee; send, quien reparte a los HUD conectados."""
        self._bus = bus
        self._send = send
        self.state = CoreState.DORMIDO
        self._subscriptions: list[Subscription] = []

    def hello(self) -> Message:
        """El saludo para un HUD que se acaba de conectar: version y estado actual."""
        return {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "bands": BANDS,
            "state": self.state.value,
        }

    def start(self) -> None:
        """Se suscribe al bus."""
        if self._subscriptions:
            return
        sub = self._bus.subscribe
        self._subscriptions = [
            sub(EventType.CORE_STATE_CHANGED, self._on_state, name="hud.state"),
            sub(EventType.COGNITION_ROUTED, self._on_routed, name="hud.routed"),
            sub(EventType.COGNITION_COMPLETED, self._on_completed, name="hud.completed"),
            sub(
                EventType.VOICE_LEVEL, self._on_level, name="hud.mic",
                max_queue=_AUDIO_QUEUE, overflow=OverflowPolicy.DROP_OLDEST,
            ),
            sub(
                EventType.VOICE_TTS_AMPLITUDE, self._on_tts, name="hud.tts",
                max_queue=_AUDIO_QUEUE, overflow=OverflowPolicy.DROP_OLDEST,
            ),
        ]  # fmt: skip

    async def stop(self) -> None:
        """Se da de baja."""
        for subscription in self._subscriptions:
            await subscription.unsubscribe()
        self._subscriptions.clear()

    # ------------------------------------------------------------------

    async def _on_state(self, event: Event) -> None:
        change = event.payload_as(StateChange)
        self.state = change.current
        self._send(
            {
                "type": "state",
                "state": change.current.value,
                "previous": change.previous.value,
                "reason": change.reason,
            }
        )
        if change.current is CoreState.PENSANDO:
            self._send({"type": "cognition", "load": _THINKING_LOAD, "level": 0})

    async def _on_routed(self, event: Event) -> None:
        decision = event.payload_as(RouteDecision)
        load = _LOAD_BY_LEVEL.get(decision.level, _THINKING_LOAD)
        self._send({"type": "cognition", "load": load, "level": int(decision.level)})

    async def _on_completed(self, event: Event) -> None:
        self._send({"type": "cognition", "load": _IDLE_LOAD, "level": 0})

    async def _on_level(self, event: Event) -> None:
        level = event.payload_as(AudioLevel)
        self._send(
            {
                "type": "audio",
                "source": "mic",
                "level": round(min(1.0, level.rms * _MIC_GAIN), 3),
                "bands": level.bands,
            }
        )

    async def _on_tts(self, event: Event) -> None:
        amplitude = event.payload_as(TtsAmplitude)
        self._send(
            {
                "type": "audio",
                "source": "tts",
                "level": round(amplitude.amplitude, 3),
                "bands": amplitude.bands,
            }
        )
