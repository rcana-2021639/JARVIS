"""El enrutador: decide que nivel atiende cada orden.

Entradas de la decision:

    · nivel sugerido por el clasificador y su confianza
    · complejidad estimada: pasos encadenados, codigo, contexto necesario
    · si el usuario pidio la nube o razonamiento profundo
    · modo de privacidad (LOCAL_ONLY corta el nivel 3)
    · presupuesto restante y si hay clave (lo resume `cloud_blocker`)
    · despues, la confianza que reporta el nivel 2

Regla permanente: nunca llamar a la nube si el nivel inferior resuelve. Cada
decision lleva su motivo; el orquestador lo publica y lo registra siempre.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jarvis.cognition.llm import Complexity, GateDecision
from jarvis.cognition.text import normalize
from jarvis.contracts.enums import CascadeLevel, PrivacyMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from jarvis.config import CognitionConfig
    from jarvis.contracts.cognition import Intent

__all__ = ["PrivacySwitch", "Route", "Router", "TaskProfile", "profile_task"]

_STEP_SPLIT = re.compile(
    r"\b(?:y luego|luego|despues|y despues|y entonces|entonces|ademas|tambien|al final|por ultimo"
    r"|primero|segundo|tercero)\b|;|\n\s*(?:\d+[.)]|-)\s"
)
_CODE_WORDS = re.compile(
    r"\b(?:codigo|script|funcion|clase|refactoriza\w*|depura\w*|compila\w*|python|javascript"
    r"|typescript|java|sql|regex|bug|stack trace|api|endpoint|json|html|css)\b"
)
_CODE_SYMBOLS = re.compile(r"```|\bdef \w+\(|\bfunction \w*\(|=>|\bimport \w+|[{};]\s*$", re.M)
_LINES = re.compile(r"\b(\d+)\s*lineas\b")
_DEEP = re.compile(
    r"\b(?:opus|a fondo|profundamente|con calma|razonamiento|demuestra|paso a paso|exhaustiv\w*)\b"
)
_TOKENS_PER_LINE = 12
_DEEP_INTENTS = {"task.plan"}


class PrivacySwitch:
    """El modo de privacidad vigente: el del TOML, salvo que se cambie por voz.

    Lo comparten el orquestador y el cliente de la nube. Que el cliente lo
    consulte por su cuenta es defensa en profundidad: LOCAL_ONLY corta la red
    aunque el enrutador tuviera un error.
    """

    def __init__(self, configured: Callable[[], PrivacyMode]) -> None:
        """Args: configured, el modo del TOML (se lee en caliente)."""
        self._configured = configured
        self._override: PrivacyMode | None = None

    @property
    def mode(self) -> PrivacyMode:
        """El modo efectivo."""
        return self._override or self._configured()

    def set(self, mode: PrivacyMode) -> None:
        """Cambia el modo hasta el proximo reinicio."""
        self._override = mode

    def network_allowed(self) -> bool:
        """Si el nivel 3 puede salir a la red."""
        return self.mode is not PrivacyMode.LOCAL_ONLY


@dataclass(frozen=True, slots=True)
class TaskProfile:
    """Lo que se puede estimar de una orden sin llamar a ningun modelo."""

    steps: int
    has_code: bool
    context_tokens: int
    """Contexto que la tarea necesitaria: historial + lo que menciona (archivos)."""
    deep: bool


def profile_task(text: str, context_tokens: int) -> TaskProfile:
    """Estima pasos, codigo, contexto y profundidad de una orden."""
    normalized = normalize(text)
    steps = 1 + len(_STEP_SPLIT.findall(normalized))
    has_code = bool(_CODE_WORDS.search(normalized) or _CODE_SYMBOLS.search(text))
    extra = sum(int(n) * _TOKENS_PER_LINE for n in _LINES.findall(normalized))
    return TaskProfile(
        steps=steps,
        has_code=has_code,
        context_tokens=context_tokens + extra,
        deep=bool(_DEEP.search(normalized)),
    )


@dataclass(frozen=True, slots=True)
class Route:
    """Una decision de enrutamiento."""

    level: CascadeLevel
    reason: str
    complexity: Complexity = Complexity.LIGHT
    degraded: bool = False
    """Si la orden pedia la nube y no se pudo: se contesta en local y se avisa."""
    wanted_cloud_because: str | None = None


class Router:
    """Las reglas de la cascada, puras y sin estado."""

    def __init__(self, config: Callable[[], CognitionConfig]) -> None:
        """Args: config, la seccion vigente (umbrales en caliente)."""
        self._config = config

    def decide(
        self,
        intent: Intent,
        profile: TaskProfile,
        *,
        privacy: PrivacyMode,
        cloud_blocker: str | None,
        fast_path_handles: bool,
    ) -> Route:
        """Primera decision para una orden.

        Args:
            intent: lo que entendio el clasificador.
            profile: la complejidad estimada.
            privacy: el modo vigente.
            cloud_blocker: por que no se puede usar la nube, o None si se puede.
            fast_path_handles: si el nivel 1 tiene manejador para la intencion.
        """
        cfg = self._config()
        if (
            intent.suggested_level is CascadeLevel.FAST_PATH
            and fast_path_handles
            and intent.confidence >= cfg.fast_path_min_confidence
        ):
            return Route(CascadeLevel.FAST_PATH, "forma_fija")

        why = self._cloud_reason(intent, profile, privacy)
        complexity = self._complexity(intent, profile, why)
        if why is None:
            return Route(CascadeLevel.LOCAL_LLM, "conversacion")
        if cloud_blocker is None:
            return Route(CascadeLevel.CLOUD_LLM, why, complexity)
        return Route(
            CascadeLevel.LOCAL_LLM,
            f"{why}; nube no disponible: {cloud_blocker}",
            complexity,
            degraded=privacy is not PrivacyMode.CLOUD_PREFERRED,
            wanted_cloud_because=why,
        )

    def escalate_after_local(
        self, confidence: float | None, *, cloud_blocker: str | None
    ) -> Route | None:
        """Tras el nivel 2: devuelve la escalada si la confianza no alcanza."""
        cfg = self._config()
        if confidence is None or confidence >= cfg.local_confidence_threshold:
            return None
        if cloud_blocker is not None:
            return None
        return Route(
            CascadeLevel.CLOUD_LLM,
            f"confianza_local_{confidence:.2f}<{cfg.local_confidence_threshold:.2f}",
            Complexity.LIGHT,
        )

    def gate(self, *, cloud_blocker: str | None, margin: float) -> Callable[[float], GateDecision]:
        """La funcion que el nivel 2 consulta con su confianza temprana.

        Holgada: hablar ya. Al limite: generar entera y medir. Baja: cortar y
        escalar, pero solo si hay nube a donde escalar; si no, se genera entera
        y se contesta con la advertencia de que no hay certeza.
        """
        threshold = self._config().local_confidence_threshold

        def decide(early: float) -> GateDecision:
            if early >= threshold + margin:
                return GateDecision.STREAM
            if early < threshold and cloud_blocker is None:
                return GateDecision.ABORT
            return GateDecision.BUFFER

        return decide

    # ------------------------------------------------------------------

    def _cloud_reason(
        self, intent: Intent, profile: TaskProfile, privacy: PrivacyMode
    ) -> str | None:
        cfg = self._config()
        if intent.name == "cloud.explicit":
            return "pedido_explicito"
        if profile.has_code:
            return "codigo"
        if profile.steps > cfg.max_chained_steps:
            return f"pasos_{profile.steps}>{cfg.max_chained_steps}"
        if profile.context_tokens > cfg.max_context_tokens:
            return f"contexto_{profile.context_tokens}>{cfg.max_context_tokens}"
        if intent.suggested_level is CascadeLevel.CLOUD_LLM:
            return f"intencion_{intent.name}"
        if privacy is PrivacyMode.CLOUD_PREFERRED:
            return "modo_nube_preferida"
        return None

    @staticmethod
    def _complexity(intent: Intent, profile: TaskProfile, why: str | None) -> Complexity:
        if profile.deep and intent.name == "cloud.explicit":
            return Complexity.DEEP
        if why is None or why == "modo_nube_preferida":
            return Complexity.LIGHT
        if (
            profile.has_code
            or intent.name in _DEEP_INTENTS
            or why.startswith(("pasos", "contexto"))
        ):
            return Complexity.STANDARD
        return Complexity.STANDARD if profile.deep else Complexity.LIGHT
