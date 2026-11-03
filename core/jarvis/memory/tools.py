"""Las herramientas de memoria que ve el orquestador (servidor `memory`).

    recordar(consulta)               recuperacion hibrida
    guardar_hecho(hecho, importancia)
    olvidar(consulta)                NO borra: deja el borrado esperando un
                                     "si" hablado del usuario
    listar_entidades(tipo)
    relacionar(a, relacion, b)
    linea_de_tiempo(desde, hasta)    episodios de un periodo

Tienen la forma de una herramienta MCP (nombre, descripcion, esquema), igual
que las de la Fase 6: el modelo las usa con el mismo bucle.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from jarvis.cognition.tool_loop import ToolSpec
from jarvis.contracts.enums import RiskLevel
from jarvis.memory.graph import ENTITY_TYPES
from jarvis.memory.handlers import stage_forget
from jarvis.memory.text import clip

if TYPE_CHECKING:
    from pydantic import JsonValue

    from jarvis.cognition.confirm import PendingConfirmations
    from jarvis.memory.service import MemorySystem

__all__ = ["memory_tools"]

_SERVER = "memory"
_TIMELINE_LIMIT = 50


def _string(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


def _parse_moment(value: str, now: datetime) -> datetime:
    """Fecha u hora ISO; sin zona se entiende hora local."""
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=now.astimezone().tzinfo)


def memory_tools(memory: MemorySystem, confirmations: PendingConfirmations) -> list[ToolSpec]:
    """Las seis herramientas, ligadas a un sistema de memoria."""

    async def recordar(args: dict[str, Any]) -> JsonValue:
        retrieval = await memory.retriever.retrieve(str(args["consulta"]))
        texts: list[JsonValue] = list(retrieval.texts)
        return {"recuerdos": texts}

    async def guardar_hecho(args: dict[str, Any]) -> JsonValue:
        importance = float(args.get("importancia", 0.6))
        written = await memory.save_fact(
            str(args["hecho"]), source="modelo", importance=max(0.0, min(1.0, importance))
        )
        return {"resultado": written.outcome, "hecho": written.fact.content}

    async def olvidar(args: dict[str, Any]) -> JsonValue:
        topic = str(args["consulta"])
        found = await stage_forget(memory, topic, confirmations)
        if found == 0:
            return {"pendiente": False, "mensaje": f"No hay recuerdos sobre {topic}."}
        return {
            "pendiente": True,
            "recuerdos": found,
            "mensaje": (
                "No se borro nada todavia. Pidele al usuario que confirme en voz "
                "alta diciendo: Jarvis, si."
            ),
        }

    async def listar_entidades(args: dict[str, Any]) -> JsonValue:
        kind = args.get("tipo") or None
        entities = await asyncio.to_thread(memory.graph.entities, kind)
        return [{"nombre": e.name, "tipo": e.type, "menciones": e.mentions} for e in entities]

    async def relacionar(args: dict[str, Any]) -> JsonValue:
        triple = await asyncio.to_thread(
            memory.graph.relate, str(args["a"]), str(args["relacion"]), str(args["b"]),
            origin="modelo",
        )  # fmt: skip
        return {"relacion": triple.text}

    async def linea_de_tiempo(args: dict[str, Any]) -> JsonValue:
        now = memory.now()
        start = _parse_moment(str(args["desde"]), now)
        end = _parse_moment(str(args["hasta"]), now) if args.get("hasta") else now
        if end <= start:
            end = start + timedelta(days=1)
        episodes = await asyncio.to_thread(
            memory.episodic.between, start, end, limit=_TIMELINE_LIMIT
        )
        return [
            {"fecha": e.at.astimezone().isoformat(timespec="minutes"),
             "contenido": clip(e.content, 200)}
            for e in episodes
        ]  # fmt: skip

    return [
        ToolSpec(
            _SERVER, "recordar",
            "Busca en la memoria a largo plazo lo relacionado con una consulta.",
            {"type": "object", "properties": {"consulta": _string("Que buscar.")},
             "required": ["consulta"]},
            recordar,
        ),
        ToolSpec(
            _SERVER, "guardar_hecho",
            "Guarda un hecho duradero sobre el usuario, en segunda persona.",
            {"type": "object", "properties": {
                "hecho": _string("El hecho, una oracion: 'Tu hermana se llama Ana'."),
                "importancia": {"type": "number", "minimum": 0, "maximum": 1},
            }, "required": ["hecho"]},
            guardar_hecho,
        ),
        ToolSpec(
            _SERVER, "olvidar",
            "Prepara el borrado de lo que se sabe sobre un tema. Requiere que el "
            "usuario confirme en voz alta; esta herramienta sola no borra nada.",
            {"type": "object", "properties": {"consulta": _string("El tema a olvidar.")},
             "required": ["consulta"]},
            olvidar,
            risk=RiskLevel.AMBAR,
        ),
        ToolSpec(
            _SERVER, "listar_entidades",
            "Lista las personas, proyectos, archivos, apps, sitios o lugares conocidos.",
            {"type": "object", "properties": {
                "tipo": {"type": "string", "enum": list(ENTITY_TYPES)},
            }},
            listar_entidades,
        ),
        ToolSpec(
            _SERVER, "relacionar",
            "Anota una relacion entre dos entidades: (Maria, trabaja_en, Proyecto Aurora).",
            {"type": "object", "properties": {
                "a": _string("Entidad de origen."),
                "relacion": _string("Verbo corto: trabaja_en, usa_archivo, vive_en."),
                "b": _string("Entidad de destino."),
            }, "required": ["a", "relacion", "b"]},
            relacionar,
        ),
        ToolSpec(
            _SERVER, "linea_de_tiempo",
            "Lo que paso entre dos fechas (ISO 8601, hora local).",
            {"type": "object", "properties": {
                "desde": _string("Inicio, por ejemplo 2026-11-02 o 2026-11-02T08:00."),
                "hasta": _string("Fin. Si falta, ahora."),
            }, "required": ["desde"]},
            linea_de_tiempo,
        ),
    ]  # fmt: skip
