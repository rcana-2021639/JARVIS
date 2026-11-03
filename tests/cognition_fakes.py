"""Piezas falsas y deterministas para probar la cascada sin modelos ni red.

FakeEmbedder   bolsa de palabras con hash: identicos dan 1.0, ajenos ~0
FakeLocal      un "Ollama" con guion: confianza, texto, fallos a media frase
FakeCloud      una "API de Claude" con guion, costo y bloqueos
build()        el orquestador completo armado con estos falsos
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from jarvis.cognition.budget import SpendLedger
from jarvis.cognition.catalog import build_catalog
from jarvis.cognition.classifier import IntentClassifier
from jarvis.cognition.llm import GateDecision, LLMReply, LLMUnavailableError, ToolUse
from jarvis.cognition.orchestrator import CognitiveOrchestrator
from jarvis.cognition.phrases import PhrasePicker
from jarvis.cognition.router import PrivacySwitch
from jarvis.cognition.tool_loop import ToolRegistry
from jarvis.config import CognitionConfig
from jarvis.contracts import CascadeLevel, PrivacyMode

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from jarvis.bus.base import EventBus
    from jarvis.cognition.llm import LLMRequest

FIXED_NOW = datetime(2026, 11, 2, 15, 25).astimezone()


class FakeEmbedder:
    """Vectores de bolsa de palabras. Suficiente para ejercitar la etapa (b)."""

    identity = "fake-bow"
    dims = 512

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls += 1
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.split():
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


@dataclass
class Script:
    """Lo que un modelo falso hara en una llamada."""

    text: str = "Respuesta de prueba."
    confidence: float | None = 0.9
    early: float | None = None
    """Confianza temprana; por defecto igual a la final."""
    fail: str | None = None
    """Si existe, la llamada falla con este motivo."""
    fail_after_words: int | None = None
    """Con `fail`: falla despues de decir tantas palabras (Ollama muriendo a media frase)."""
    tool_calls: list[ToolUse] = field(default_factory=list)
    cost_usd: float = 0.0


class _ScriptedModel:
    level: CascadeLevel
    model = "fake"

    def __init__(self, scripts: Sequence[Script] | None = None) -> None:
        self._scripts = list(scripts or [Script()])
        self.requests: list[LLMRequest] = []

    def _next(self) -> Script:
        return self._scripts.pop(0) if len(self._scripts) > 1 else self._scripts[0]

    async def available(self) -> bool:
        return True

    async def _stream(self, script: Script, request: LLMRequest) -> bool:
        if request.on_text is None:
            return False
        words = script.text.split(" ")
        for index, word in enumerate(words):
            if (
                script.fail
                and script.fail_after_words is not None
                and index >= script.fail_after_words
            ):
                raise LLMUnavailableError(self.level, script.fail, spoke=index > 0)
            await request.on_text(word if index == 0 else f" {word}")
        return True


class FakeLocal(_ScriptedModel):
    """Nivel 2 falso. Respeta el `gate` como el cliente real de Ollama."""

    level = CascadeLevel.LOCAL_LLM
    model = "fake-local"

    def __init__(self, scripts: Sequence[Script] | None = None) -> None:
        super().__init__(scripts)
        self.warmed = False

    async def warm_up(self, system: str) -> bool:
        self.warmed = True
        return True

    async def generate(self, request: LLMRequest) -> LLMReply:
        self.requests.append(request)
        script = self._next()
        if script.fail and script.fail_after_words is None:
            raise LLMUnavailableError(self.level, script.fail)
        early = script.early if script.early is not None else script.confidence
        decision = GateDecision.STREAM
        if request.gate is not None and early is not None:
            decision = request.gate(early)
        if decision is GateDecision.ABORT:
            return LLMReply(
                text="", level=self.level, model=self.model, confidence=early,
                early_confidence=early, gate=decision, stop_reason="aborted",
            )  # fmt: skip
        streamed = False
        if decision is GateDecision.STREAM:
            streamed = await self._stream(script, request)
        return LLMReply(
            text=script.text, level=self.level, model=self.model,
            tool_calls=list(script.tool_calls), confidence=script.confidence,
            early_confidence=early, gate=decision, streamed=streamed,
        )  # fmt: skip


class FakeCloud(_ScriptedModel):
    """Nivel 3 falso, con motivo de bloqueo y costo por llamada."""

    level = CascadeLevel.CLOUD_LLM
    model = "fake-cloud"

    def __init__(
        self,
        scripts: Sequence[Script] | None = None,
        *,
        blocked: str | None = None,
        ledger: SpendLedger | None = None,
        privacy: PrivacySwitch | None = None,
    ) -> None:
        super().__init__(scripts)
        self.blocked = blocked
        self.ledger = ledger
        self.privacy = privacy

    def unavailable_reason(self) -> str | None:
        if self.privacy is not None and not self.privacy.network_allowed():
            return "modo solo-local"
        if self.ledger is not None and not self.ledger.can_spend():
            return "presupuesto del dia agotado"
        return self.blocked

    def model_for(self, complexity: Any) -> str:
        return f"fake-cloud-{complexity.value.lower()}"

    async def generate(self, request: LLMRequest) -> LLMReply:
        reason = self.unavailable_reason()
        if reason is not None:
            raise AssertionError(f"Se llamo a la nube bloqueada: {reason}")
        self.requests.append(request)
        script = self._next()
        if script.fail and script.fail_after_words is None:
            raise LLMUnavailableError(self.level, script.fail)
        streamed = await self._stream(script, request)
        if self.ledger is not None:
            self.ledger.record(script.cost_usd)
        return LLMReply(
            text=script.text, level=self.level, model=self.model_for(request.complexity),
            tool_calls=list(script.tool_calls), cost_usd=script.cost_usd, streamed=streamed,
        )  # fmt: skip


@dataclass
class Rig:
    """El orquestador armado y sus piezas, para inspeccionarlas en la prueba."""

    orchestrator: CognitiveOrchestrator
    config: CognitionConfig
    ledger: SpendLedger
    privacy: PrivacySwitch
    local: FakeLocal | None
    cloud: FakeCloud | None
    tools: ToolRegistry


def build(
    bus: EventBus,
    *,
    local: FakeLocal | None = None,
    cloud: FakeCloud | None = None,
    privacy: PrivacyMode = PrivacyMode.HYBRID,
    budget: float = 0.5,
    embedder: FakeEmbedder | None = None,
    rng: random.Random | None = None,
    **overrides: Any,
) -> Rig:
    """Arma un orquestador con falsos. `overrides` va a `CognitionConfig`."""
    config = CognitionConfig(privacy_mode=privacy, daily_budget_usd=budget, **overrides)
    switch = PrivacySwitch(lambda: config.privacy_mode)
    ledger = SpendLedger(None, lambda: config.daily_budget_usd, today=FIXED_NOW.date)
    if cloud is not None:
        cloud.ledger = cloud.ledger or ledger
        cloud.privacy = cloud.privacy or switch
    tools = ToolRegistry(bus, lambda: config.tool_timeout_s)
    classifier = IntentClassifier(build_catalog(), lambda: config, embedder)
    orchestrator = CognitiveOrchestrator(
        bus,
        lambda: config,
        classifier,
        ledger=ledger,
        privacy=switch,
        local=local,
        cloud=cloud,
        tools=tools,
        user_name=lambda: "Rhandy",
        clock=lambda: FIXED_NOW,
        phrases=PhrasePicker(rng),
    )
    return Rig(orchestrator, config, ledger, switch, local, cloud, tools)
