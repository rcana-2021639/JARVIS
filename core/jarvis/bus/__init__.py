"""Bus de eventos: como se hablan entre si los modulos del nucleo.

Nadie llama directamente a nadie. El pipeline de voz publica que detecto la
palabra clave; el orquestador, el HUD y la bitacora se enteran porque estan
suscritos. Agregar un consumidor nuevo no obliga a tocar al productor.

Uso tipico:

    bus = InMemoryEventBus()
    async with bus:
        bus.subscribe("voice.*", on_voice_event, name="hud")
        await bus.publish(Event.of(EventType.VOICE_STT_FINAL, utterance, source="stt"))
        await bus.wait_idle(timeout=1.0)

Importa siempre `EventBus` para tipar, nunca `InMemoryEventBus`: es lo que
permitira cambiar la implementacion sin tocar a un solo consumidor.
"""

from jarvis.bus.base import (
    DEFAULT_RETRY_POLICY,
    EventBus,
    EventHandler,
    OverflowPolicy,
    RetryPolicy,
    Subscription,
)
from jarvis.bus.memory_bus import InMemoryEventBus
from jarvis.bus.metrics import BusMetrics, TypeMetrics

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "BusMetrics",
    "EventBus",
    "EventHandler",
    "InMemoryEventBus",
    "OverflowPolicy",
    "RetryPolicy",
    "Subscription",
    "TypeMetrics",
]
