"""Fixtures compartidas por todas las pruebas."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.bus import InMemoryEventBus
from jarvis.contracts import (
    CascadeLevel,
    Command,
    CommandOrigin,
    Intent,
    Language,
    Response,
    RiskLevel,
    ToolCall,
    Utterance,
    new_correlation_id,
)
from jarvis.logging import bind_bus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator


@pytest.fixture
def correlation_id() -> str:
    """Un id de correlacion nuevo para la prueba."""
    return new_correlation_id()


@pytest.fixture
async def bus() -> AsyncIterator[InMemoryEventBus]:
    """Un bus arrancado y limpio, que se cierra al terminar la prueba."""
    instance = InMemoryEventBus()
    await instance.start()
    try:
        yield instance
    finally:
        await instance.stop(drain=False)


@pytest.fixture(autouse=True)
def _unbind_bus() -> Iterator[None]:
    """Evita que el bus de una prueba se filtre a la siguiente via `@traced`."""
    yield
    bind_bus(None)


@pytest.fixture
def utterance(correlation_id: str) -> Utterance:
    """Una transcripcion final valida."""
    return Utterance(
        text="sube el volumen",
        confidence=0.94,
        language=Language.ES,
        duration_ms=820.0,
        is_final=True,
        correlation_id=correlation_id,
    )


@pytest.fixture
def intent() -> Intent:
    """Una intencion de ruta rapida."""
    return Intent(
        name="media.volume_up",
        entities={"delta": 10},
        confidence=0.98,
        suggested_level=CascadeLevel.FAST_PATH,
    )


@pytest.fixture
def command(intent: Intent, correlation_id: str) -> Command:
    """Una orden lista para ejecutarse."""
    return Command(
        intent=intent,
        raw_text="Jarvis, sube el volumen",
        origin=CommandOrigin.VOZ,
        correlation_id=correlation_id,
    )


@pytest.fixture
def response(correlation_id: str) -> Response:
    """Una respuesta de nivel 1, sin costo."""
    return Response(
        text="Listo.",
        handled_by_level=CascadeLevel.FAST_PATH,
        correlation_id=correlation_id,
    )


@pytest.fixture
def tool_call(correlation_id: str) -> ToolCall:
    """Una invocacion de herramienta de nivel VERDE."""
    return ToolCall(
        server="media",
        tool="volume_up",
        arguments={"delta": 10},
        risk_level=RiskLevel.VERDE,
        correlation_id=correlation_id,
    )
