"""Contratos de datos del Proyecto JARVIS -- el idioma comun de todos los modulos.

Todo lo que cruza una frontera (el bus, un WebSocket, el disco) es uno de estos
modelos. Ningun modulo pasa diccionarios sueltos a otro.

Los diez contratos y de que hablan:

    Utterance     lo que el usuario dijo, ya transcrito
    Intent        lo que el clasificador entendio que quiere
    Command       la orden lista para ejecutarse
    ToolCall      la invocacion de una herramienta
    ToolResult    lo que esa herramienta devolvio
    Response      lo que Jarvis contesta, y cuanto costo producirlo
    MemoryRecord  una unidad de memoria de cualquiera de las cinco capas
    SystemState   el estado global en un instante
    AuditEntry    una linea de la bitacora de solo-anadir
    Event         el sobre que viaja por el bus

Cada uno se exporta como esquema JSON a `shared/protocol/` para que el HUD y la
app movil compartan exactamente los mismos tipos. Ver `scripts/export_schemas.py`.
"""

from jarvis.contracts.audit import AuditEntry
from jarvis.contracts.base import (
    BaseContract,
    CorrelationId,
    new_correlation_id,
    new_id,
    utc_now,
)
from jarvis.contracts.cascade import CascadeRecord, CostUpdate, ResponseChunk, RouteDecision
from jarvis.contracts.cognition import Command, Intent, Response
from jarvis.contracts.enums import (
    CascadeLevel,
    CommandOrigin,
    CoreState,
    Emotion,
    EventType,
    Language,
    MemoryKind,
    PrivacyMode,
    RiskLevel,
    SpeechEndReason,
)
from jarvis.contracts.events import Event
from jarvis.contracts.memory import MemoryRecord
from jarvis.contracts.system import StateChange, SystemState, Telemetry
from jarvis.contracts.tools import ToolCall, ToolResult
from jarvis.contracts.voice import (
    AudioLevel,
    SpeechActivity,
    SpeechOutput,
    TtsAmplitude,
    Utterance,
    WakeDetection,
)

#: Los contratos que se exportan como esquema JSON a shared/protocol/.
#: Si agregas un contrato nuevo, agregalo aqui o el HUD no lo vera nunca.
EXPORTED_CONTRACTS: tuple[type[BaseContract], ...] = (
    Utterance,
    Intent,
    Command,
    ToolCall,
    ToolResult,
    Response,
    MemoryRecord,
    Telemetry,
    SystemState,
    AuditEntry,
    Event,
    # Fase 1: lo que el pipeline de voz publica para el HUD.
    AudioLevel,
    WakeDetection,
    SpeechActivity,
    SpeechOutput,
    TtsAmplitude,
    StateChange,
    # Fase 2: la contabilidad de la cascada.
    ResponseChunk,
    RouteDecision,
    CostUpdate,
    CascadeRecord,
)

__all__ = [
    "EXPORTED_CONTRACTS",
    "AudioLevel",
    "AuditEntry",
    "BaseContract",
    "CascadeLevel",
    "CascadeRecord",
    "Command",
    "CommandOrigin",
    "CoreState",
    "CorrelationId",
    "CostUpdate",
    "Emotion",
    "Event",
    "EventType",
    "Intent",
    "Language",
    "MemoryKind",
    "MemoryRecord",
    "PrivacyMode",
    "Response",
    "ResponseChunk",
    "RiskLevel",
    "RouteDecision",
    "SpeechActivity",
    "SpeechEndReason",
    "SpeechOutput",
    "StateChange",
    "SystemState",
    "Telemetry",
    "ToolCall",
    "ToolResult",
    "TtsAmplitude",
    "Utterance",
    "WakeDetection",
    "new_correlation_id",
    "new_id",
    "utc_now",
]
