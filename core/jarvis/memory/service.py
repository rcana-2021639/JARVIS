"""El sistema de memoria completo: las cinco capas detras de una fachada.

    working       ultimos turnos + estado, en RAM          (working.py)
    episodic      todo lo ocurrido, SQLite + FTS5          (episodic.py)
    semantic      hechos con importancia, LanceDB          (semantic.py)
    graph         entidades y relaciones, SQLite           (graph.py)
    procedural    secuencias que funcionaron -> rutinas    (procedural.py)

Se conecta a la cascada de dos formas: le da a la cascada la recuperacion
hibrida (`recollect`) y la memoria de trabajo como historial al construirla, y
despues `attach` registra los manejadores de voz, las herramientas y las
rutinas aprendidas en el orquestador ya construido.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from jarvis.cognition.context import Recollection, estimate_tokens
from jarvis.cognition.fastpath import FastOutcome
from jarvis.cognition.spoken import local_now
from jarvis.contracts.enums import CoreState, Emotion, EventType
from jarvis.contracts.events import Event
from jarvis.logging import get_logger
from jarvis.memory.consolidation import ConsolidationScheduler, Consolidator
from jarvis.memory.episodic import Episode, EpisodicMemory
from jarvis.memory.extraction import Extraction, OllamaExtractor, RuleExtractor
from jarvis.memory.graph import Entity, EntityGraph
from jarvis.memory.handlers import MemoryVoice
from jarvis.memory.procedural import ProceduralMemory, Routine
from jarvis.memory.recorder import EpisodeRecorder
from jarvis.memory.retrieval import HybridRetriever
from jarvis.memory.semantic import Fact, FactWrite, SemanticMemory
from jarvis.memory.tools import memory_tools
from jarvis.memory.working import WorkingMemory

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime

    from jarvis.bus.base import EventBus
    from jarvis.cognition.catalog import Entities
    from jarvis.cognition.embeddings import Embedder
    from jarvis.cognition.fastpath import FastContext
    from jarvis.cognition.orchestrator import CognitiveOrchestrator
    from jarvis.config import CognitionConfig, MemoryConfig
    from jarvis.contracts.memory import ConsolidationReport
    from jarvis.memory.extraction import Extractor
    from jarvis.memory.store import MemoryStore

__all__ = ["Forgettable", "MemorySystem"]

_log = get_logger(__name__)
_SOURCE = "memory"
_ABOUT_LIMIT = 8


@dataclass(frozen=True, slots=True)
class Forgettable:
    """Todo lo que "olvida lo de X" borraria."""

    facts: list[Fact] = field(default_factory=list)
    episodes: list[Episode] = field(default_factory=list)
    entity: Entity | None = None

    @property
    def total(self) -> int:
        """Cuantos recuerdos son."""
        return len(self.facts) + len(self.episodes) + (1 if self.entity else 0)


class MemorySystem:
    """Las cinco capas, la recuperacion, la consolidacion y el grabador."""

    def __init__(
        self,
        bus: EventBus,
        config: Callable[[], MemoryConfig],
        store: MemoryStore,
        embedder: Embedder,
        *,
        cognition: Callable[[], CognitionConfig],
        user_name: Callable[[], str] = lambda: "Usuario",
        clock: Callable[[], datetime] | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        """Arma las capas sobre un almacenamiento ya abierto."""
        self._bus = bus
        self._config = config
        self.store = store
        self.embedder = embedder
        self.now: Callable[[], datetime] = clock or local_now
        self.working = WorkingMemory(lambda: config().working_turns, self.now)
        self.episodic = EpisodicMemory(store, self.now)
        self.semantic = SemanticMemory(store, embedder, config, self.now)
        self.graph = EntityGraph(store, self.now)
        self.procedural = ProceduralMemory(
            store, self.now, lambda: config().routine_promotion_threshold
        )
        self.rules = RuleExtractor(user_name)
        if extractor is None:
            local = config().consolidation_engine == "local_llm"
            extractor = OllamaExtractor(cognition, user_name) if local else self.rules
        self.retriever = HybridRetriever(self.semantic, self.episodic, self.graph, config, self.now)
        self.consolidator = Consolidator(
            episodic=self.episodic, semantic=self.semantic, graph=self.graph,
            procedural=self.procedural, extractor=extractor, rules=self.rules,
            config=config, clock=self.now, busy=self._user_active,
        )  # fmt: skip
        self.scheduler = ConsolidationScheduler(
            self.consolidator, store, config, self.now, self._publish_report
        )
        self.recorder = EpisodeRecorder(
            bus, episodic=self.episodic, procedural=self.procedural, working=self.working,
            on_routine=self.activate_routine,
        )  # fmt: skip
        self._orchestrator: CognitiveOrchestrator | None = None

    def _user_active(self) -> bool:
        """Si hubo una orden hace poco o Jarvis esta en medio de algo.

        Un estado que no es DORMIDO cuenta solo si es reciente: sin el pipeline
        de voz (jarvis_chat sin --voz) nadie devuelve el nucleo a DORMIDO.
        """
        snapshot = self.working.snapshot
        idle = timedelta(minutes=self._config().consolidation_idle_minutes)
        now = self.now()
        recent = [snapshot.last_interaction_at]
        if snapshot.state is not CoreState.DORMIDO:
            recent.append(snapshot.state_since)
        return any(moment is not None and now - moment < idle for moment in recent)

    # ------------------------------------------------------------------
    # Ciclo de vida

    def attach(self, orchestrator: CognitiveOrchestrator) -> None:
        """Registra voz, herramientas y rutinas en un orquestador ya construido."""
        self._orchestrator = orchestrator
        MemoryVoice(self).register(orchestrator.fast_path)
        for tool in memory_tools(self, orchestrator.confirmations):
            orchestrator.tools.register(tool)
        for routine in self.procedural.routines("activa"):
            self.activate_routine(routine)

    async def start(self) -> ConsolidationReport | None:
        """Empieza a grabar y programa la consolidacion.

        Devuelve el reporte de la ultima consolidacion si nadie lo ha oido
        todavia ("que aprendi anoche"), y lo publica en el bus.
        """
        self.recorder.start()
        self.scheduler.start()
        report = await asyncio.to_thread(self.scheduler.unreported)
        if report is not None:
            _log.info("memory.report", text=report.spoken())
            await self._emit_report(report)
        return report

    async def stop(self) -> None:
        """Deja de grabar y detiene la consolidacion programada."""
        await self.recorder.stop()
        await self.scheduler.stop()

    def close(self) -> None:
        """Cierra el almacenamiento. Despues de esto el objeto no sirve."""
        self.store.close()

    # ------------------------------------------------------------------
    # Lo que usa la cascada

    async def recollect(self, text: str) -> Recollection:
        """Recupera lo relevante, ya cortado al presupuesto de cada nivel. Nunca lanza."""
        cfg = self._config()
        try:
            retrieval = await self.retriever.retrieve(
                text, budget_tokens=cfg.retrieval_budget_tokens
            )
        except Exception:
            # Sin memoria se contesta igual: peor, pero se contesta.
            _log.exception("memory.recollect_failed")
            return Recollection()
        local: list[str] = []
        used = 0
        for item in retrieval.texts:
            cost = estimate_tokens(item) + 2
            if used + cost <= cfg.local_retrieval_tokens:
                local.append(item)
                used += cost
        _log.info(
            "memory.recollected", items=len(retrieval.items), local=len(local),
            elapsed_ms=retrieval.elapsed_ms,
        )  # fmt: skip
        return Recollection(local=tuple(local), cloud=tuple(retrieval.texts))

    async def save_fact(
        self,
        content: str,
        *,
        source: str,
        text: str | None = None,
        importance: float | None = None,
    ) -> FactWrite:
        """Guarda un hecho y, de la frase original, las entidades que se reconozcan."""
        weight = importance if importance is not None else self._config().remember_importance
        written = await asyncio.to_thread(
            self.semantic.add, content, importance=weight, source=source
        )
        if text:
            await asyncio.to_thread(self._link, self.rules.from_text(text))
        await self._bus.publish(
            Event.of(EventType.MEMORY_RECORD_STORED, written.fact.record(), source=_SOURCE)
        )
        return written

    def _link(self, extraction: Extraction) -> None:
        for entity in extraction.entities:
            self.graph.upsert(entity.name, entity.type)
        for rel in extraction.relations:
            self.graph.relate(
                rel.source, rel.relation, rel.target,
                source_type=rel.source_type, target_type=rel.target_type,
            )  # fmt: skip

    async def about(self, topic: str) -> list[str]:
        """Lo que se sabe de un tema: relaciones del grafo primero, luego hechos."""
        cfg = self._config()

        def collect() -> list[str]:
            items: list[str] = []
            vector = self.semantic.embed([topic])[0]
            facts = [f for f, s in self.semantic.search_vector(vector, _ABOUT_LIMIT)
                     if s >= cfg.vector_min_similarity]  # fmt: skip
            facts += [f for f, _ in self.semantic.search_text(topic, _ABOUT_LIMIT)]
            seen = {f.id: f for f in facts}
            self.semantic.touch(list(seen))
            items.extend(f.content for f in seen.values())
            # Despues de los hechos, las relaciones directas: lo que esta a dos
            # saltos sirve para el contexto de un modelo, no para decirlo en voz alta.
            entity = self.graph.find(topic)
            if entity is not None:
                items.extend(t.text for t in self.graph.traverse([entity.id], 1))
            return list(dict.fromkeys(items))[:_ABOUT_LIMIT]

        return await asyncio.to_thread(collect)

    async def find_forgettable(self, topic: str) -> Forgettable:
        """Todo lo que hay guardado sobre un tema."""

        def collect() -> Forgettable:
            facts = self.semantic.matching(topic)
            episodes = [e for e, _ in self.episodic.search(topic, 200)]
            return Forgettable(facts, episodes, self.graph.find(topic))

        return await asyncio.to_thread(collect)

    async def forget(self, found: Forgettable, words: list[str]) -> int:
        """Borra de verdad: hechos, episodios, la entidad y los turnos en RAM."""

        def remove() -> int:
            removed = self.semantic.delete([f.id for f in found.facts])
            removed += self.episodic.delete([e.id for e in found.episodes])
            if found.entity is not None:
                removed += self.graph.delete([found.entity.id])
            return removed

        removed = await asyncio.to_thread(remove)
        self.working.forget(words)
        _log.info("memory.forgotten", records=removed)
        return removed

    # ------------------------------------------------------------------
    # Rutinas

    def activate_routine(self, routine: Routine) -> None:
        """Una rutina promovida pasa a la ruta rapida, sin reiniciar nada."""
        orchestrator = self._orchestrator
        if orchestrator is None:
            return
        orchestrator.classifier.add_routine(routine.trigger, routine.intent_name)
        orchestrator.fast_path.register(routine.intent_name, self._routine_handler(routine))
        _log.info("memory.routine_active", routine=routine.intent_name, trigger=routine.trigger)

    def _routine_handler(
        self, routine: Routine
    ) -> Callable[[Entities, FastContext], Awaitable[FastOutcome]]:
        async def handle(entities: Entities, ctx: FastContext) -> FastOutcome:
            calls = []
            for step in routine.steps:
                call, result = await ctx.tools.call(step.tool, step.arguments, ctx.correlation_id)
                if call is not None:
                    calls.append(call)
                if not result.success:
                    await asyncio.to_thread(self.procedural.record_use, routine.id, success=False)
                    return FastOutcome(
                        f"La rutina falló en {step.tool}: {result.error}.", Emotion.ERROR,
                        actions=calls,
                    )  # fmt: skip
            await asyncio.to_thread(self.procedural.record_use, routine.id, success=True)
            return FastOutcome("Hecho.", actions=calls)

        return handle

    # ------------------------------------------------------------------

    async def _publish_report(self, report: ConsolidationReport) -> None:
        await self._emit_report(report)

    async def _emit_report(self, report: ConsolidationReport) -> None:
        await self._bus.publish(
            Event.of(EventType.MEMORY_CONSOLIDATION_FINISHED, report, source=_SOURCE)
        )

    def stats(self) -> dict[str, int]:
        """Cuantos registros hay por capa, para el inspector."""
        entities, relations = self.graph.counts()
        return {
            "trabajo (turnos en RAM)": len(self.working.turns),
            "episodios": self.episodic.count(),
            "hechos activos": self.semantic.count("activo"),
            "hechos obsoletos": self.semantic.count("obsoleto"),
            "entidades": entities,
            "relaciones": relations,
            "patrones de acciones": len(self.procedural.patterns()),
            "rutinas activas": len(self.procedural.routines("activa")),
            "rutinas propuestas": len(self.procedural.routines("propuesta")),
        }
