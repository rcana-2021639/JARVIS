"""La ruta rapida: el nivel 1 de la cascada. Determinista, sin modelo.

Cada intencion de forma fija mapea a un manejador que recibe las entidades y
devuelve que decir. Hay dos clases de manejador:

    nativos       los resuelve Jarvis solo: hora, fecha, temporizadores,
                  modo de privacidad, gasto, repetir, callarse, cortesia
    herramienta   necesitan el sistema operativo (volumen, apps, brillo) y
                  pasan por el `ToolRegistry`. En la Fase 2 esas herramientas
                  no existen todavia: el manejador lo dice tal cual en vez de
                  fingir que lo hizo. En la Fase 6 se registran y funcionan
                  sin tocar este archivo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from jarvis.cognition.demo_tools import safe_arithmetic
from jarvis.cognition.spoken import (
    spoken_date,
    spoken_duration,
    spoken_number,
    spoken_time,
)
from jarvis.contracts.enums import Emotion, PrivacyMode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime

    from jarvis.cognition.catalog import Entities
    from jarvis.cognition.confirm import PendingConfirmations
    from jarvis.cognition.phrases import PhrasePicker
    from jarvis.cognition.timers import TimerService
    from jarvis.cognition.tool_loop import ToolRegistry
    from jarvis.contracts.cognition import Intent
    from jarvis.contracts.tools import ToolCall

__all__ = ["CascadeControl", "FastContext", "FastOutcome", "FastPath"]


class CascadeControl(Protocol):
    """Lo que la ruta rapida necesita saber o cambiar del orquestador."""

    @property
    def user_name(self) -> str:
        """Como se llama el usuario."""
        ...

    def last_spoken(self) -> str | None:
        """La ultima respuesta dicha, para "repite"."""
        ...

    def privacy_mode(self) -> PrivacyMode:
        """El modo vigente."""
        ...

    def set_privacy_mode(self, mode: PrivacyMode) -> None:
        """Cambia el modo en caliente, sin tocar el TOML."""
        ...

    def spending(self) -> tuple[float, float]:
        """(gastado hoy, presupuesto del dia) en dolares."""
        ...

    async def health(self) -> str:
        """Una frase sobre el estado de los niveles 2 y 3."""
        ...


@dataclass(slots=True)
class FastOutcome:
    """Lo que el nivel 1 contesta."""

    text: str
    emotion: Emotion = Emotion.AFIRMATIVA
    should_speak: bool = True
    actions: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class FastContext:
    """Todo lo que un manejador puede usar."""

    correlation_id: str
    control: CascadeControl
    tools: ToolRegistry
    timers: TimerService
    phrases: PhrasePicker
    now: datetime
    step_percent: int
    cancelled_previous: bool = False
    """Si esta orden interrumpio otra que seguia en curso."""
    text: str = ""
    """La orden tal cual llego, con mayusculas y tildes (las entidades van normalizadas)."""
    confirmations: PendingConfirmations | None = None
    """La accion que espera un "si", si hay alguna."""


Handler = "Callable[[Entities, FastContext], Awaitable[FastOutcome]]"


def _capitalized(text: str) -> str:
    return text[:1].upper() + text[1:]


def _tool_handler(
    tool: str,
    phrase: str,
    what: str,
    arguments: Callable[[Entities, int], dict[str, Any] | None] = lambda e, s: {},
    ask: str = "ask_value",
) -> Callable[[Entities, FastContext], Awaitable[FastOutcome]]:
    """Construye un manejador que delega en una herramienta del sistema.

    Args:
        tool: `servidor.nombre` de la herramienta.
        phrase: grupo de frases de exito.
        what: la accion en infinitivo, para decir que no se pudo.
        arguments: arma los argumentos desde las entidades. None si falta un
            dato imprescindible: entonces se pregunta.
        ask: grupo de frases para preguntar el dato que falta.
    """

    async def handle(entities: Entities, ctx: FastContext) -> FastOutcome:
        args = arguments(entities, ctx.step_percent)
        if args is None:
            return FastOutcome(ctx.phrases.pick(ask), Emotion.INTERROGATIVA)
        subject = what.format(**args)
        call, result = await ctx.tools.call(tool, args, ctx.correlation_id)
        if call is None:
            text = _capitalized(ctx.phrases.pick("unavailable", what=subject))
            return FastOutcome(text, Emotion.NEUTRAL)
        if not result.success:
            text = ctx.phrases.pick("tool_failed", what=subject, error=result.error)
            return FastOutcome(text, Emotion.ERROR, actions=[call])
        return FastOutcome(ctx.phrases.pick(phrase, **args), actions=[call])

    return handle


def _value(entities: Entities, step: int) -> dict[str, Any] | None:
    value = entities.get("value")
    return None if value is None else {"value": value}


def _delta(sign: int) -> Callable[[Entities, int], dict[str, Any]]:
    def build(entities: Entities, step: int) -> dict[str, Any]:
        value = entities.get("value")
        amount = value if isinstance(value, int) and 0 < value <= 100 else step
        return {"delta": sign * amount}

    return build


def _app(entities: Entities, step: int) -> dict[str, Any] | None:
    app = entities.get("app")
    return {"app": app} if app else None


def _query(entities: Entities, step: int) -> dict[str, Any]:
    query = entities.get("query")
    return {"query": query} if query else {}


class FastPath:
    """Registro de manejadores del nivel 1."""

    def __init__(self) -> None:
        """Registra los manejadores por defecto."""
        self._handlers: dict[str, Callable[[Entities, FastContext], Awaitable[FastOutcome]]] = {}
        self._register_defaults()

    def register(
        self, intent: str, handler: Callable[[Entities, FastContext], Awaitable[FastOutcome]]
    ) -> None:
        """Agrega o reemplaza el manejador de una intencion."""
        self._handlers[intent] = handler

    def handles(self, intent: str) -> bool:
        """Si el nivel 1 sabe atender esta intencion."""
        return intent in self._handlers

    @property
    def intents(self) -> list[str]:
        """Las intenciones que atiende, ordenadas."""
        return sorted(self._handlers)

    async def execute(self, intent: Intent, ctx: FastContext) -> FastOutcome:
        """Atiende una intencion. Lanza KeyError si no hay manejador."""
        return await self._handlers[intent.name](dict(intent.entities), ctx)

    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        tool = self.register
        tool(
            "media.volume_up",
            _tool_handler("media.volume_change", "volume_up", "subir el volumen", _delta(+1)),
        )
        tool(
            "media.volume_down",
            _tool_handler("media.volume_change", "volume_down", "bajar el volumen", _delta(-1)),
        )
        tool(
            "media.volume_set",
            _tool_handler(
                "media.volume_set", "volume_set", "poner el volumen al {value} por ciento", _value
            ),
        )
        tool(
            "media.mute",
            _tool_handler("media.mute", "mute", "silenciar el audio", lambda e, s: {"muted": True}),
        )
        tool(
            "media.unmute",
            _tool_handler(
                "media.mute", "unmute", "quitar el silencio", lambda e, s: {"muted": False}
            ),
        )
        tool("media.play", _tool_handler("media.play", "play", "reproducir", _query))
        tool("media.pause", _tool_handler("media.pause", "pause", "pausar"))
        tool("media.next", _tool_handler("media.next", "next", "pasar a la siguiente"))
        tool("media.previous", _tool_handler("media.previous", "previous", "volver a la anterior"))
        tool(
            "windows.minimize_all",
            _tool_handler("windows.minimize_all", "minimize_all", "minimizar todo"),
        )
        tool(
            "windows.close_all",
            _tool_handler("windows.close_all", "close_all", "cerrar las ventanas"),
        )
        tool("windows.switch", _tool_handler("windows.switch", "switch", "cambiar de ventana"))
        tool("apps.open", _tool_handler("apps.open", "open", "abrir {app}", _app, ask="ask_app"))
        tool(
            "apps.close", _tool_handler("apps.close", "close", "cerrar {app}", _app, ask="ask_app")
        )
        tool(
            "system.screenshot",
            _tool_handler("vision.screenshot", "screenshot", "tomar la captura"),
        )
        tool("system.lock", _tool_handler("system.lock", "lock", "bloquear la sesión"))
        tool(
            "system.brightness_up",
            _tool_handler(
                "system.brightness_change", "brightness_up", "subir el brillo", _delta(+1)
            ),
        )
        tool(
            "system.brightness_down",
            _tool_handler(
                "system.brightness_change", "brightness_down", "bajar el brillo", _delta(-1)
            ),
        )
        tool(
            "system.brightness_set",
            _tool_handler(
                "system.brightness_set",
                "brightness_set",
                "poner el brillo al {value} por ciento",
                _value,
            ),
        )

        self.register("query.arithmetic", self._arithmetic)
        self.register("system.time", self._time)
        self.register("system.date", self._date)
        self.register("timer.set", self._timer_set)
        self.register("timer.cancel", self._timer_cancel)
        self.register("timer.query", self._timer_query)
        self.register("jarvis.stop_talking", self._stop_talking)
        self.register("jarvis.cancel", self._cancel)
        self.register("jarvis.confirm", self._confirm)
        self.register("jarvis.deny", self._deny)
        self.register("jarvis.repeat", self._repeat)
        self.register(
            "jarvis.privacy_local", self._privacy(PrivacyMode.LOCAL_ONLY, "privacy_local")
        )
        self.register("jarvis.privacy_hybrid", self._privacy(PrivacyMode.HYBRID, "privacy_hybrid"))
        self.register("jarvis.budget", self._budget)
        self.register("jarvis.status", self._status)
        self.register("chat.greeting", self._polite("greeting"))
        self.register("chat.thanks", self._polite("thanks"))
        self.register("chat.goodbye", self._polite("goodbye"))  # fmt: skip

    @staticmethod
    async def _arithmetic(entities: Entities, ctx: FastContext) -> FastOutcome:
        expression = entities.get("expression")
        try:
            value = safe_arithmetic(str(expression))
        except (ValueError, ZeroDivisionError, OverflowError):
            return FastOutcome("Esa cuenta no la puedo hacer así.", Emotion.ERROR)
        return FastOutcome(f"Son {spoken_number(value)}.", Emotion.NEUTRAL)

    @staticmethod
    async def _time(entities: Entities, ctx: FastContext) -> FastOutcome:
        return FastOutcome(spoken_time(ctx.now), Emotion.NEUTRAL)

    @staticmethod
    async def _date(entities: Entities, ctx: FastContext) -> FastOutcome:
        return FastOutcome(spoken_date(ctx.now), Emotion.NEUTRAL)

    @staticmethod
    async def _timer_set(entities: Entities, ctx: FastContext) -> FastOutcome:
        seconds = entities.get("seconds")
        if not isinstance(seconds, int) or seconds <= 0:
            return FastOutcome(ctx.phrases.pick("ask_duration"), Emotion.INTERROGATIVA)
        label = entities.get("label")
        label_text = str(label) if label else None
        ctx.timers.start(seconds, label_text)
        duration = spoken_duration(seconds)
        if label_text:
            return FastOutcome(
                ctx.phrases.pick("timer_set_label", duration=duration, label=label_text)
            )
        return FastOutcome(ctx.phrases.pick("timer_set", duration=duration))

    @staticmethod
    async def _timer_cancel(entities: Entities, ctx: FastContext) -> FastOutcome:
        count = ctx.timers.cancel_all()
        if count == 0:
            return FastOutcome(ctx.phrases.pick("timer_none"), Emotion.NEUTRAL)
        key = "timer_cancel" if count == 1 else "timer_cancel_many"
        return FastOutcome(ctx.phrases.pick(key, count=count))

    @staticmethod
    async def _timer_query(entities: Entities, ctx: FastContext) -> FastOutcome:
        active = ctx.timers.active
        if not active:
            return FastOutcome(ctx.phrases.pick("timer_none"), Emotion.NEUTRAL)
        left = spoken_duration(active[0].remaining())
        return FastOutcome(ctx.phrases.pick("timer_left", duration=left), Emotion.NEUTRAL)

    @staticmethod
    async def _stop_talking(entities: Entities, ctx: FastContext) -> FastOutcome:
        # La palabra clave ya lo callo (barge-in). Contestar seria hablar otra vez.
        return FastOutcome("", Emotion.NEUTRAL, should_speak=False)

    @staticmethod
    async def _cancel(entities: Entities, ctx: FastContext) -> FastOutcome:
        # "Cancela" tambien es un no a la pregunta de confirmacion pendiente.
        denied = ctx.confirmations.deny() if ctx.confirmations is not None else None
        key = "cancel" if ctx.cancelled_previous or denied is not None else "cancel_nothing"
        return FastOutcome(ctx.phrases.pick(key), Emotion.NEUTRAL)

    @staticmethod
    async def _confirm(entities: Entities, ctx: FastContext) -> FastOutcome:
        done = await ctx.confirmations.confirm() if ctx.confirmations is not None else None
        if done is None:
            return FastOutcome("No hay nada pendiente de confirmar.", Emotion.NEUTRAL)
        return FastOutcome(done)

    @staticmethod
    async def _deny(entities: Entities, ctx: FastContext) -> FastOutcome:
        denied = ctx.confirmations.deny() if ctx.confirmations is not None else None
        if denied is None:
            return FastOutcome("No hay nada pendiente de confirmar.", Emotion.NEUTRAL)
        return FastOutcome("Entendido, no toco nada.", Emotion.NEUTRAL)

    @staticmethod
    async def _repeat(entities: Entities, ctx: FastContext) -> FastOutcome:
        last = ctx.control.last_spoken()
        if not last:
            return FastOutcome(ctx.phrases.pick("repeat_nothing"), Emotion.NEUTRAL)
        return FastOutcome(last, Emotion.NEUTRAL)

    @staticmethod
    def _privacy(
        mode: PrivacyMode, key: str
    ) -> Callable[[Entities, FastContext], Awaitable[FastOutcome]]:
        async def handle(entities: Entities, ctx: FastContext) -> FastOutcome:
            if ctx.control.privacy_mode() is mode:
                return FastOutcome(ctx.phrases.pick("privacy_same"), Emotion.NEUTRAL)
            ctx.control.set_privacy_mode(mode)
            return FastOutcome(ctx.phrases.pick(key))

        return handle

    @staticmethod
    async def _budget(entities: Entities, ctx: FastContext) -> FastOutcome:
        spent, budget = ctx.control.spending()
        if budget <= 0:
            text = f"Hoy van {spent:.2f} dólares, sin tope configurado."
        else:
            text = f"Hoy van {spent:.2f} de {budget:.2f} dólares."
        return FastOutcome(text, Emotion.NEUTRAL)

    @staticmethod
    async def _status(entities: Entities, ctx: FastContext) -> FastOutcome:
        return FastOutcome(await ctx.control.health(), Emotion.NEUTRAL)

    @staticmethod
    def _polite(key: str) -> Callable[[Entities, FastContext], Awaitable[FastOutcome]]:
        async def handle(entities: Entities, ctx: FastContext) -> FastOutcome:
            return FastOutcome(ctx.phrases.pick(key, name=ctx.control.user_name), Emotion.NEUTRAL)

        return handle
