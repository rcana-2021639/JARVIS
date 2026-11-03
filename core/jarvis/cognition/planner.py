"""De una intencion clasificada a una ruta y su contexto.

Decide el nivel con el enrutador y, si la orden va a un modelo, le pide a la
memoria lo que viene al caso y arma el contexto de cada nivel con ello. Es el
paso entre clasificar y resolver; separado del orquestador para que este se
lea como el flujo de una orden.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.cognition.context import Recollection
from jarvis.cognition.levels import Interaction
from jarvis.cognition.router import Route, profile_task
from jarvis.contracts.enums import CascadeLevel

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.cognition.context import MemoryAccess, PromptAssembler
    from jarvis.cognition.control import SessionControl
    from jarvis.cognition.fastpath import FastPath
    from jarvis.cognition.levels import Attempt
    from jarvis.cognition.router import PrivacySwitch, Router
    from jarvis.cognition.speaker import ResponseStream
    from jarvis.config import CognitionConfig
    from jarvis.contracts.cognition import Intent

__all__ = ["CascadePlanner"]


class CascadePlanner:
    """Enruta una orden y le prepara el contexto, con memoria si la hay."""

    def __init__(
        self,
        *,
        config: Callable[[], CognitionConfig],
        router: Router,
        fast: FastPath,
        prompts: PromptAssembler,
        control: SessionControl,
        privacy: PrivacySwitch,
    ) -> None:
        """Conecta las piezas que deciden."""
        self._config = config
        self._router = router
        self._fast = fast
        self._prompts = prompts
        self._control = control
        self._privacy = privacy
        self.memory: MemoryAccess | None = None

    async def plan(
        self,
        text: str,
        intent: Intent,
        *,
        correlation_id: str,
        stream: ResponseStream,
        attempt: Attempt,
    ) -> tuple[Route, Interaction]:
        """La ruta elegida y la interaccion lista para resolverse."""
        cfg = self._config()
        build = self._prompts.build
        local_ctx = build(text, structured=True, budget=cfg.local_context_tokens)
        route = self._router.decide(
            intent,
            profile_task(text, local_ctx.tokens),
            privacy=self._privacy.mode,
            cloud_blocker=self._control.cloud_blocker(),
            fast_path_handles=self._fast.handles(intent.name),
        )
        recalled = Recollection()
        # El nivel 1 no lee contexto: recuperar para el seria latencia tirada.
        if route.level is not CascadeLevel.FAST_PATH and self.memory is not None:
            recalled = await self.memory.recollect(text)
            if recalled.local:
                local_ctx = build(
                    text, structured=True, budget=cfg.local_context_tokens, memory=recalled.local
                )
        interaction = Interaction(
            text=text,
            intent=intent,
            correlation_id=correlation_id,
            stream=stream,
            attempt=attempt,
            local_context=lambda: local_ctx,
            cloud_context=lambda: build(
                text, structured=False, budget=cfg.cloud_context_tokens, memory=recalled.cloud
            ),
            cloud_blocker=self._control.cloud_blocker,
        )
        return route, interaction
