"""Inspector de la memoria: que hay en cada capa y cuanto tarda recuperarla.

Uso:

    python scripts/memory_inspect.py                     # resumen de las cinco capas
    python scripts/memory_inspect.py --query "mi hermana"  # recuperacion con latencias
    python scripts/memory_inspect.py --bench 10000       # 10.000 registros sinteticos
    python scripts/memory_inspect.py --consolidate       # consolidar ya lo pendiente
    python scripts/memory_inspect.py --simulate          # 100 episodios simulados + consolidacion

`--bench` y `--simulate` trabajan en una carpeta temporal: no tocan la memoria
real de config/data/.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

from jarvis.bus import InMemoryEventBus
from jarvis.config import load_config
from jarvis.logging import configure_logging
from jarvis.memory import MemorySystem, build_memory
from jarvis.memory.factory import build_memory_embedder
from jarvis.memory.simulation import simulated_week
from jarvis.memory.store import dumps
from jarvis.memory.text import to_epoch

_TOPICS = (
    "la reunion de presupuesto", "el proyecto Aurora", "mi hermana Ana", "el gimnasio",
    "la cena del viernes", "el informe trimestral", "las vacaciones en la playa",
    "el cumpleanos de mama", "la factura de luz", "el curso de Rust", "el dentista",
    "la mudanza", "el coche", "la boda de Luis", "el servidor de casa", "la tesis",
)  # fmt: skip
_VERBS = (
    "quedo para", "cambio a", "se movio al", "es el", "necesita", "depende de",
    "se cancelo por", "empieza el", "termina el", "cuesta",
)  # fmt: skip
_OBJECTS = (
    "lunes", "martes", "jueves", "viernes por la tarde", "mes que viene", "200 dolares",
    "las 8 de la manana", "la oficina", "casa de Ana", "Zoom", "la sala 3", "marzo",
)  # fmt: skip


def _print_summary(memory: MemorySystem) -> None:
    print("\nREGISTROS POR CAPA")
    for name, count in memory.stats().items():
        print(f"  {name:<28} {count:>7}")
    print("\nHECHOS POR IMPORTANCIA EFECTIVA (los 20 primeros)")
    for fact in memory.semantic.active(20):
        flag = " permanente" if fact.importance >= 1.0 else ""
        print(f"  {fact.weight:.2f}  {fact.content}  [{fact.source}{flag}]")
    print("\nGRAFO")
    for entity in memory.graph.entities():
        print(f"  ({entity.type}) {entity.name}  x{entity.mentions}")
    for triple in memory.graph.relations():
        print(f"    {triple.source} --{triple.relation}--> {triple.target}")
    print("\nRUTINAS")
    for routine in memory.procedural.routines():
        steps = ", ".join(step.tool for step in routine.steps) or "sin pasos"
        print(f"  [{routine.status}] '{routine.trigger}' -> {steps}  {routine.note}")
    report = memory.scheduler.last_report()
    if report is not None:
        print(f"\nULTIMA CONSOLIDACION ({report.finished_at:%Y-%m-%d %H:%M}): {report.spoken()}")


async def _query(memory: MemorySystem, text: str) -> None:
    await memory.retriever.retrieve("calentar")
    result = await memory.retriever.retrieve(text)
    timings = ", ".join(f"{name} {ms:.1f} ms" for name, ms in result.timings_ms.items())
    print(f"\nRECUPERACION de '{text}': {result.elapsed_ms:.1f} ms ({timings})")
    if result.timed_out:
        print(f"  fuera de plazo: {result.timed_out}")
    for item in result.items:
        sources = "+".join(dict.fromkeys(item.sources))
        print(f"  {item.score:.4f}  {item.text}  [{sources}]")
    print(f"  {result.tokens} tokens de {memory._config().retrieval_budget_tokens}")


def _seed(memory: MemorySystem, count: int) -> None:
    rng = np.random.default_rng(11)
    now = to_epoch(memory.now())
    texts = [
        f"{rng.choice(_TOPICS).capitalize()} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} "
        f"(nota {index})."
        for index in range(count)
    ]
    print(f"Codificando {count} hechos con {memory.embedder.identity} ...", flush=True)
    started = time.perf_counter()
    vectors = np.concatenate([memory.embedder.embed(texts[i : i + 256])
                              for i in range(0, count, 256)])  # fmt: skip
    print(f"  {time.perf_counter() - started:.1f} s")
    ids = [f"bench{i}" for i in range(count)]
    with memory.store.transaction() as conn:
        conn.executemany(
            "INSERT INTO facts (id, content, fact_key, importance, created_at, last_accessed_at, "
            "source, metadata) VALUES (?, ?, '', 0.5, ?, ?, 'bench', ?)",
            [(i, t, now - n * 600, now - n * 600, dumps({})) for n, (i, t) in
             enumerate(zip(ids, texts, strict=True))],
        )  # fmt: skip
        conn.executemany(
            "INSERT INTO episodes (id, kind, content, user_text, at) VALUES (?, 'turno', ?, ?, ?)",
            [(f"ep{n}", f"Usuario: {t}\nJarvis: Anotado.", t, now - n * 300)
             for n, t in enumerate(texts)],
        )  # fmt: skip
    memory.store.vectors.add(ids, vectors)
    memory.graph.relate("María", "trabaja_en", "Proyecto Aurora", source_type="persona",
                        target_type="proyecto")  # fmt: skip
    memory.graph.relate("Proyecto Aurora", "usa_archivo", "informe.docx", target_type="archivo")


async def _bench(count: int) -> None:
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        config = config.model_copy(
            update={"memory": config.memory.model_copy(update={"data_dir": Path(folder)})}
        )
        memory = build_memory(InMemoryEventBus(), lambda: config)
        _seed(memory, count)
        queries = [
            "¿cuándo es la reunión de presupuesto?", "¿qué sabes de María?",
            "lo del dentista", "¿cuánto cuesta el curso de Rust?", "la boda",
            "¿dónde quedó la cena del viernes?", "el informe", "vacaciones",
        ] * 4  # fmt: skip
        await memory.retriever.retrieve("calentar")
        totals: list[float] = []
        per_search: dict[str, list[float]] = {}
        for query in queries:
            started = time.perf_counter()
            result = await memory.retriever.retrieve(query)
            totals.append((time.perf_counter() - started) * 1000)
            for name, ms in result.timings_ms.items():
                per_search.setdefault(name, []).append(ms)
        totals.sort()
        p95 = totals[int(len(totals) * 0.95) - 1]
        print(f"\nRECUPERACION con {count} hechos + {count} episodios ({len(queries)} consultas)")
        print(f"  punta a punta: p50 {statistics.median(totals):.1f} ms, p95 {p95:.1f} ms, "
              f"max {totals[-1]:.1f} ms  (meta < 100 ms)")  # fmt: skip
        for name, values in per_search.items():
            print(f"  {name:<7} p50 {statistics.median(values):.1f} ms")
        memory.close()


async def _simulate() -> None:
    config = load_config()
    with tempfile.TemporaryDirectory() as folder:
        config = config.model_copy(
            update={"memory": config.memory.model_copy(update={"data_dir": Path(folder)})}
        )
        memory = build_memory(InMemoryEventBus(), lambda: config)
        for turn in simulated_week(memory.now()):
            memory.episodic.add(
                "turno", f"Usuario: {turn.user}\nJarvis: {turn.answer}", user_text=turn.user,
                answer=turn.answer, intent=turn.intent, level=turn.level, at=turn.at,
            )  # fmt: skip
        print(f"100 episodios simulados. Consolidando con {config.memory.consolidation_engine} ...")
        started = time.perf_counter()
        report = await memory.consolidator.run()
        print(f"  {time.perf_counter() - started:.1f} s, extractor: {report.extractor}")
        for error in report.errors:
            print(f"  error: {error}")
        print(f"  {report.spoken()}")
        _print_summary(memory)
        memory.close()


async def run(args: argparse.Namespace) -> int:
    configure_logging(level="WARNING", development=True, log_file=None)
    if args.bench:
        await _bench(args.bench)
        return 0
    if args.simulate:
        await _simulate()
        return 0
    config = load_config()
    if args.data:
        config = config.model_copy(
            update={"memory": config.memory.model_copy(update={"data_dir": Path(args.data)})}
        )
    memory = build_memory(
        InMemoryEventBus(), lambda: config, embedder=build_memory_embedder(config)
    )
    print(f"Memoria en {config.resolved(config.memory.data_dir)}")
    if args.consolidate:
        report = await memory.consolidator.run()
        print(f"Consolidacion: {report.spoken()} ({report.episodes} episodios, {report.extractor})")
    _print_summary(memory)
    if args.query:
        await _query(memory, args.query)
    memory.close()
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", help="Prueba de recuperacion con latencias.")
    parser.add_argument("--bench", type=int, metavar="N", help="Banco con N registros sinteticos.")
    parser.add_argument("--consolidate", action="store_true", help="Consolidar lo pendiente ya.")
    parser.add_argument("--simulate", action="store_true", help="100 episodios simulados.")
    parser.add_argument("--data", help="Otra carpeta de datos en vez de memory.data_dir.")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
