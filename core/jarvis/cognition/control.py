"""Lo que la ruta rapida puede consultar o cambiar de la cascada.

Implementa `fastpath.CascadeControl`: el nombre del usuario, lo ultimo que se
dijo, el modo de privacidad, el gasto y la salud de los niveles 2 y 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.contracts.enums import CascadeLevel
from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.cognition.budget import SpendLedger
    from jarvis.cognition.levels import CloudModel
    from jarvis.cognition.llm import LocalModel
    from jarvis.cognition.router import PrivacySwitch, Route
    from jarvis.contracts.enums import PrivacyMode

__all__ = ["SessionControl"]

_log = get_logger(__name__)


class SessionControl:
    """El estado de la sesion que comparten el orquestador y la ruta rapida."""

    def __init__(
        self,
        *,
        user_name: Callable[[], str],
        privacy: PrivacySwitch,
        ledger: SpendLedger,
        local: LocalModel | None,
        cloud: CloudModel | None,
    ) -> None:
        """Args: las piezas cuyo estado se expone."""
        self._user_name = user_name
        self._privacy = privacy
        self._ledger = ledger
        self._local = local
        self.cloud = cloud
        self.spoken: str | None = None
        """Lo ultimo que se dijo; el orquestador lo actualiza en cada respuesta."""

    @property
    def user_name(self) -> str:
        """Como se llama el usuario."""
        return self._user_name()

    def last_spoken(self) -> str | None:
        """La ultima respuesta dicha."""
        return self.spoken

    def privacy_mode(self) -> PrivacyMode:
        """El modo efectivo."""
        return self._privacy.mode

    def set_privacy_mode(self, mode: PrivacyMode) -> None:
        """Cambia el modo hasta reiniciar."""
        self._privacy.set(mode)
        _log.info("cognition.privacy_changed", mode=mode.value)

    def spending(self) -> tuple[float, float]:
        """(gastado hoy, presupuesto)."""
        return self._ledger.spent_today, self._ledger.budget

    def cloud_blocker(self) -> str | None:
        """Por que no se puede usar la nube ahora, o None si se puede."""
        if self.cloud is None:
            return "no hay cliente de la nube"
        return self.cloud.unavailable_reason()

    def model_for(self, route: Route) -> str | None:
        """Que modelo concreto atiende una ruta, para el registro."""
        if route.level is CascadeLevel.LOCAL_LLM and self._local is not None:
            return self._local.model
        if route.level is CascadeLevel.CLOUD_LLM and self.cloud is not None:
            model_for = getattr(self.cloud, "model_for", None)
            return model_for(route.complexity) if callable(model_for) else None
        return None

    async def health(self) -> str:
        """Estado de los niveles 2 y 3 en una frase."""
        local_ok = self._local is not None and await self._local.available()
        blocker = self.cloud_blocker()
        local = "Modelo local listo" if local_ok else "Modelo local apagado"
        cloud = "nube disponible" if blocker is None else f"nube no disponible: {blocker}"
        return f"{local}; {cloud}."
