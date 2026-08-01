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
)
from jarvis.contracts.events import Event
from jarvis.contracts.memory import MemoryRecord
from jarvis.contracts.system import SystemState, Telemetry
from jarvis.contracts.tools import ToolCall, ToolResult
from jarvis.contracts.voice import Utterance

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
)

__all__ = [
    "EXPORTED_CONTRACTS",
    "AuditEntry",
    "BaseContract",
    "CascadeLevel",
    "Command",
    "CommandOrigin",
    "CoreState",
    "CorrelationId",
    "Emotion",
    "Event",
    "EventType",
    "Intent",
    "Language",
    "MemoryKind",
    "MemoryRecord",
    "PrivacyMode",
    "Response",
    "RiskLevel",
    "SystemState",
    "Telemetry",
    "ToolCall",
    "ToolResult",
    "Utterance",
    "new_correlation_id",
    "new_id",
    "utc_now",
]
