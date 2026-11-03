"""La consolidacion nocturna: lo que hace un cerebro dormido.

Cada noche, a `memory.consolidation_hour`:

    1. lee los episodios que todavia no consolido y los agrupa en
       conversaciones (las separa un silencio de `session_gap_minutes`)
    2. extrae hechos, entidades y relaciones de cada tramo con el modelo local
       (y siempre tambien con reglas, que no se equivocan con lo explicito)
    3. guarda los hechos: nuevos, reforzados o reemplazando a los que cambiaron
    4. enlaza las entidades en el grafo
    5. resume las conversaciones largas en un episodio `resumen`
    6. aplica el decaimiento por desuso, respetando lo permanente
    7. detecta ordenes repetidas y propone rutinas
    8. comprime los episodios de mas de `episodic_compression_days`

Usa el modelo LOCAL: es trabajo por lotes sin prisa y no tiene sentido pagar
la nube por el. Si Ollama no responde, sigue con reglas y lo anota.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from jarvis.contracts.enums import CascadeLevel
from jarvis.contracts.memory import ConsolidationReport
from jarvis.logging import get_logger
from jarvis.memory.extraction import Extraction, ExtractionError
from jarvis.memory.procedural import routine_trigger
from jarvis.memory.text import clip, from_epoch, to_epoch

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from jarvis.config import MemoryConfig
    from jarvis.memory.episodic import Episode, EpisodicMemory
    from jarvis.memory.extraction import Extractor, RuleExtractor
    from jarvis.memory.graph import EntityGraph
    from jarvis.memory.procedural import ProceduralMemory
    from jarvis.memory.semantic import SemanticMemory
    from jarvis.memory.store import MemoryStore

__all__ = ["ConsolidationScheduler", "Consolidator", "split_conversations"]

_log = get_logger(__name__)
_PATTERN_WINDOW_DAYS = 30
_YIELD_S = 15.0
_HABIT_REGULARITY = 0.75
"""Que fraccion de los dias en que se pidio algo cae en su ventana de dos horas."""
_SUMMARY_LINES = 4
_MAX_LEARNED = 10


def split_conversations(episodes: Sequence[Episode], gap: timedelta) -> list[list[Episode]]:
    """Agrupa episodios ordenados en conversaciones separadas por silencios."""
    groups: list[list[Episode]] = []
    for episode in episodes:
        if groups and episode.at - groups[-1][-1].at <= gap:
            groups[-1].append(episode)
        else:
            groups.append([episode])
    return groups


@dataclass(slots=True)
class _Tally:
    started_at: datetime
    extractor: str
    episodes: int = 0
    conversations: int = 0
    facts_new: int = 0
    facts_updated: int = 0
    facts_reinforced: int = 0
    entities_new: int = 0
    relations_new: int = 0
    summaries: int = 0
    learned: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Consolidator:
    """Corre una consolidacion completa."""

    def __init__(
        self,
        *,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        graph: EntityGraph,
        procedural: ProceduralMemory,
        extractor: Extractor,
        rules: RuleExtractor,
        config: Callable[[], MemoryConfig],
        clock: Callable[[], datetime],
        busy: Callable[[], bool] = lambda: False,
    ) -> None:
        """Conecta las capas y los extractores.

        `busy` dice si el usuario esta usando a Jarvis. Mientras lo este, la
        consolidacion espera entre tramos.
        """
        self._episodic = episodic
        self._semantic = semantic
        self._graph = graph
        self._procedural = procedural
        self._extractor = extractor
        self._rules = rules
        self._config = config
        self._clock = clock
        self._busy = busy

    async def run(self, until: datetime | None = None) -> ConsolidationReport:
        """Consolida todo lo pendiente hasta `until` (por defecto, ahora)."""
        cfg = self._config()
        now = self._clock()
        extractor: Extractor = self._extractor
        tally = _Tally(started_at=now, extractor=extractor.name)
        pending = await asyncio.to_thread(self._episodic.unconsolidated, until or now)
        conversations = split_conversations(pending, timedelta(minutes=cfg.session_gap_minutes))
        tally.episodes, tally.conversations = len(pending), len(conversations)
        entities_before, relations_before = self._graph.counts()

        for conversation in conversations:
            summary_parts: list[str] = []
            for start in range(0, len(conversation), cfg.consolidation_batch):
                batch = conversation[start : start + cfg.consolidation_batch]
                # DECISION: si el usuario habla con Jarvis, la consolidacion le
                # cede el modelo local. Ollama atiende una peticion a la vez y
                # en esta CPU un tramo tarda decenas de segundos: la orden del
                # usuario esperaria detras. Es trabajo sin prisa.
                await self._wait_until_idle()
                extraction = await self._rules.extract(batch)
                try:
                    extraction.extend(await extractor.extract(batch))
                except ExtractionError as error:
                    # Nada falla en silencio: queda en el reporte y se sigue con reglas.
                    tally.errors.append(str(error))
                    _log.warning("memory.extraction_failed", error=str(error))
                    if extractor is not self._rules:
                        extractor = self._rules
                        tally.extractor = f"{tally.extractor} y luego reglas"
                await asyncio.to_thread(self._apply, extraction, tally)
                if extraction.summary:
                    summary_parts.append(extraction.summary)
            if len(conversation) >= cfg.summary_min_turns:
                await asyncio.to_thread(self._summarize, conversation, summary_parts)
                tally.summaries += 1
            await asyncio.to_thread(self._episodic.mark_consolidated, [e.id for e in conversation])

        entities_after, relations_after = self._graph.counts()
        decayed = await asyncio.to_thread(self._semantic.apply_decay)
        proposed = await asyncio.to_thread(self._propose_routines, now)
        cutoff = now - timedelta(days=cfg.episodic_compression_days)
        compressed = await asyncio.to_thread(self._episodic.compress_before, cutoff)
        report = ConsolidationReport(
            started_at=tally.started_at,
            finished_at=self._clock(),
            extractor=tally.extractor,
            episodes=tally.episodes,
            conversations=tally.conversations,
            facts_new=tally.facts_new,
            facts_updated=tally.facts_updated,
            facts_reinforced=tally.facts_reinforced,
            entities_new=max(0, entities_after - entities_before),
            relations_new=max(0, relations_after - relations_before),
            summaries=tally.summaries,
            decayed=decayed,
            routines_proposed=proposed,
            compressed=compressed,
            learned=tally.learned[:_MAX_LEARNED],
            errors=tally.errors,
        )
        _log.info(
            "memory.consolidated",
            episodes=report.episodes,
            facts_new=report.facts_new,
            entities_new=report.entities_new,
            extractor=report.extractor,
        )
        return report

    async def _wait_until_idle(self) -> None:
        # El "ocupado" depende de la hora, no de un evento: se consulta cada tanto.
        while True:
            if not self._busy():
                return
            await asyncio.sleep(_YIELD_S)

    def _apply(self, extraction: Extraction, tally: _Tally) -> None:
        for candidate in extraction.facts:
            written = self._semantic.add(
                candidate.content, importance=candidate.importance, source="consolidacion"
            )
            if written.outcome == "nuevo":
                tally.facts_new += 1
                tally.learned.append(written.fact.content)
            elif written.outcome == "actualizado":
                tally.facts_updated += 1
                tally.learned.append(written.fact.content)
            else:
                tally.facts_reinforced += 1
        for entity in extraction.entities:
            with contextlib.suppress(ValueError):  # nombre vacio: el modelo lo dejo en blanco
                self._graph.upsert(entity.name, entity.type)
        for rel in extraction.relations:
            with contextlib.suppress(ValueError):
                self._graph.relate(
                    rel.source, rel.relation, rel.target,
                    source_type=rel.source_type, target_type=rel.target_type,
                    origin="consolidacion",
                )  # fmt: skip

    def _summarize(self, conversation: list[Episode], parts: list[str]) -> None:
        if parts:
            content = " ".join(parts)
        else:
            topics = [clip(e.user_text or e.content, 60) for e in conversation if e.user_text]
            content = "Conversación sobre: " + "; ".join(topics[:_SUMMARY_LINES])
        self._episodic.add(
            "resumen",
            content,
            at=conversation[0].at,
            importance=0.5,
            metadata={"turns": len(conversation)},
        )

    def _propose_routines(self, now: datetime) -> int:
        """Habitos: la misma orden a un modelo, a la misma hora, en dias distintos.

        DECISION: cuenta DIAS en una ventana de dos horas, no repeticiones, y
        pide que esa ventana reuna casi todos los dias en que se pidio. "Que
        clima hace" cada manana a las 8 es un habito que vale una rutina;
        "cuentame un chiste" diez veces a cualquier hora no lo es.
        """
        threshold = self._config().routine_promotion_threshold
        turns = self._episodic.between(
            now - timedelta(days=_PATTERN_WINDOW_DAYS), now, kinds=("turno",), limit=5000
        )
        by_hour: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        sample: dict[str, str] = {}
        for turn in turns:
            if not turn.user_text or turn.level in (None, CascadeLevel.FAST_PATH):
                continue
            trigger = routine_trigger(turn.user_text)
            local = turn.at.astimezone()
            by_hour[trigger][local.hour].add(local.date().isoformat())
            sample.setdefault(trigger, turn.user_text)
        proposed = 0
        for trigger, hours in by_hour.items():
            all_days = set().union(*hours.values())
            start, best_days = max(
                ((hour, hours.get(hour, set()) | hours.get(hour + 1, set())) for hour in range(24)),
                key=lambda pair: len(pair[1]),
            )
            regular = len(best_days) >= _HABIT_REGULARITY * len(all_days)
            if len(best_days) < threshold or not regular:
                continue
            hour = max((start, start + 1), key=lambda h: len(hours.get(h, set())))
            note = f"La pediste {len(best_days)} días hacia las {hour}:00."
            if self._procedural.propose(sample[trigger], [], note) is not None:
                proposed += 1
        return proposed


class ConsolidationScheduler:
    """Programa la consolidacion a su hora y guarda cada reporte."""

    def __init__(
        self,
        consolidator: Consolidator,
        store: MemoryStore,
        config: Callable[[], MemoryConfig],
        clock: Callable[[], datetime],
        on_report: Callable[[ConsolidationReport], Awaitable[None]],
    ) -> None:
        """Args: on_report recibe cada reporte terminado (para publicarlo)."""
        self._consolidator = consolidator
        self._store = store
        self._config = config
        self._clock = clock
        self._on_report = on_report
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Arranca el ciclo en segundo plano."""
        if self._task is None and self._config().consolidation_enabled:
            self._task = asyncio.create_task(self._loop(), name="memory.consolidation")

    async def stop(self) -> None:
        """Detiene el ciclo. Una consolidacion a medias deja todo consistente:
        cada conversacion se marca al terminarla, no al final."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def run_now(self) -> ConsolidationReport:
        """Consolida ya, sin solaparse con otra corrida."""
        async with self._lock:
            report = await self._consolidator.run()
            self._store.execute(
                "INSERT INTO consolidation_runs (started_at, finished_at, report) VALUES (?, ?, ?)",
                (to_epoch(report.started_at), to_epoch(report.finished_at),
                 report.model_dump_json()),
            )  # fmt: skip
        await self._on_report(report)
        return report

    def last_scheduled(self) -> datetime:
        """La ultima vez que debio correr (hoy o ayer a la hora configurada)."""
        now = self._clock().astimezone()
        today = now.replace(hour=self._config().consolidation_hour, minute=0, second=0,
                            microsecond=0)  # fmt: skip
        return today if today <= now else today - timedelta(days=1)

    def missed(self) -> bool:
        """Si la ultima corrida programada no ocurrio y hay algo que consolidar."""
        rows = self._store.query("SELECT MAX(finished_at) FROM consolidation_runs")
        last = rows[0][0]
        due = self.last_scheduled()
        if last is not None and from_epoch(last) >= due:
            return False
        pending = self._store.query(
            "SELECT COUNT(*) FROM episodes WHERE consolidated = 0 AND at < ?", (to_epoch(due),)
        )
        return int(pending[0][0]) > 0

    def unreported(self) -> ConsolidationReport | None:
        """El ultimo reporte que nadie ha oido, y lo marca como oido."""
        rows = self._store.query(
            "SELECT id, report FROM consolidation_runs WHERE reported = 0 ORDER BY id DESC LIMIT 1"
        )
        if not rows:
            return None
        self._store.execute("UPDATE consolidation_runs SET reported = 1")
        return ConsolidationReport.model_validate_json(rows[0]["report"])

    def last_report(self) -> ConsolidationReport | None:
        """El ultimo reporte, oido o no."""
        rows = self._store.query("SELECT report FROM consolidation_runs ORDER BY id DESC LIMIT 1")
        return ConsolidationReport.model_validate_json(rows[0]["report"]) if rows else None

    async def _loop(self) -> None:
        if self._config().consolidation_catch_up and self.missed():
            await self._safe_run()
        while True:
            due = self.last_scheduled() + timedelta(days=1)
            delay = max(1.0, (due - self._clock().astimezone()).total_seconds())
            await asyncio.sleep(delay)
            if self._config().consolidation_enabled:
                await self._safe_run()

    async def _safe_run(self) -> None:
        try:
            await self.run_now()
        except Exception:
            # Una noche fallida no puede tumbar el ciclo: se registra y se
            # reintenta la noche siguiente con lo que siga pendiente.
            _log.exception("memory.consolidation_failed")
