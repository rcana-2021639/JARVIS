"""El almacenamiento de la memoria: SQLite con FTS5 y LanceDB, ambos en disco.

    config/data/memory.db     SQLite: episodios, hechos, grafo, rutinas,
                              corridas de consolidacion. Busqueda de texto
                              completo con FTS5 (BM25).
    config/data/vectors/      LanceDB embebido: un vector por hecho.

DECISION: SQLite es la fuente de verdad de los hechos (texto, importancia,
decaimiento, estado) y LanceDB es solo su indice vectorial (id + vector). Asi el
decaimiento nocturno es un UPDATE barato en SQLite y no reescribe la tabla de
vectores, y si cambia el modelo de embeddings el indice se reconstruye entero
desde SQLite sin perder nada.

Las migraciones son versionadas con `PRAGMA user_version`: cada una corre una
sola vez, en orden, dentro de una transaccion. Para cambiar el esquema se agrega
una migracion al final de `MIGRATIONS`; nunca se edita una que ya corrio.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from jarvis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from lancedb.query import LanceVectorQueryBuilder
    from numpy.typing import NDArray

__all__ = ["MIGRATIONS", "MemoryStore", "VectorIndex"]

_log = get_logger(__name__)

_FTS_TOKENIZER = "unicode61 remove_diacritics 2"
"""Sin acentos al indexar: "reunion" encuentra "reunión"."""

MIGRATIONS: tuple[tuple[str, str], ...] = (
    (
        "episodios, hechos, grafo, rutinas y consolidacion",
        f"""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            user_text TEXT,
            answer TEXT,
            intent TEXT,
            level INTEGER,
            success INTEGER NOT NULL DEFAULT 1,
            correlation_id TEXT,
            at REAL NOT NULL,
            importance REAL NOT NULL DEFAULT 0.3,
            compressed INTEGER NOT NULL DEFAULT 0,
            consolidated INTEGER NOT NULL DEFAULT 0,
            metadata TEXT NOT NULL DEFAULT '{{}}'
        );
        CREATE INDEX episodes_at ON episodes (at);
        CREATE INDEX episodes_pending ON episodes (consolidated, at);
        CREATE VIRTUAL TABLE episodes_fts USING fts5 (
            content, content='episodes', content_rowid='rowid', tokenize='{_FTS_TOKENIZER}'
        );
        CREATE TRIGGER episodes_ai AFTER INSERT ON episodes BEGIN
            INSERT INTO episodes_fts (rowid, content) VALUES (new.rowid, new.content);
        END;
        CREATE TRIGGER episodes_ad AFTER DELETE ON episodes BEGIN
            INSERT INTO episodes_fts (episodes_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
        END;
        CREATE TRIGGER episodes_au AFTER UPDATE OF content ON episodes BEGIN
            INSERT INTO episodes_fts (episodes_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
            INSERT INTO episodes_fts (rowid, content) VALUES (new.rowid, new.content);
        END;

        CREATE TABLE facts (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            fact_key TEXT NOT NULL DEFAULT '',
            importance REAL NOT NULL,
            decay REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            last_accessed_at REAL NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'activo',
            superseded_by TEXT,
            metadata TEXT NOT NULL DEFAULT '{{}}'
        );
        CREATE INDEX facts_status ON facts (status);
        CREATE INDEX facts_key ON facts (fact_key);
        CREATE VIRTUAL TABLE facts_fts USING fts5 (
            content, content='facts', content_rowid='rowid', tokenize='{_FTS_TOKENIZER}'
        );
        CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts (rowid, content) VALUES (new.rowid, new.content);
        END;
        CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
            INSERT INTO facts_fts (facts_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
        END;
        CREATE TRIGGER facts_au AFTER UPDATE OF content ON facts BEGIN
            INSERT INTO facts_fts (facts_fts, rowid, content)
            VALUES ('delete', old.rowid, old.content);
            INSERT INTO facts_fts (rowid, content) VALUES (new.rowid, new.content);
        END;

        CREATE TABLE entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            key TEXT NOT NULL,
            type TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_seen_at REAL NOT NULL,
            mentions INTEGER NOT NULL DEFAULT 1,
            UNIQUE (key, type)
        );
        CREATE INDEX entities_key ON entities (key);
        CREATE TABLE relations (
            id INTEGER PRIMARY KEY,
            src INTEGER NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
            rel TEXT NOT NULL,
            dst INTEGER NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
            weight REAL NOT NULL DEFAULT 1.0,
            created_at REAL NOT NULL,
            source TEXT NOT NULL,
            UNIQUE (src, rel, dst)
        );
        CREATE INDEX relations_src ON relations (src);
        CREATE INDEX relations_dst ON relations (dst);

        CREATE TABLE action_patterns (
            trigger TEXT NOT NULL,
            signature TEXT NOT NULL,
            steps TEXT NOT NULL,
            successes INTEGER NOT NULL DEFAULT 0,
            failures INTEGER NOT NULL DEFAULT 0,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            PRIMARY KEY (trigger, signature)
        );
        CREATE TABLE routines (
            id TEXT PRIMARY KEY,
            trigger TEXT NOT NULL UNIQUE,
            steps TEXT NOT NULL,
            status TEXT NOT NULL,
            origin TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            uses INTEGER NOT NULL DEFAULT 0,
            last_used_at REAL
        );

        CREATE TABLE consolidation_runs (
            id INTEGER PRIMARY KEY,
            started_at REAL NOT NULL,
            finished_at REAL NOT NULL,
            report TEXT NOT NULL,
            reported INTEGER NOT NULL DEFAULT 0
        );
        """,
    ),
)


class MemoryStore:
    """La conexion a SQLite y a LanceDB, con migraciones aplicadas.

    DECISION: una sola conexion de SQLite protegida por un candado, y no una por
    hilo. Las consultas de la memoria tardan fracciones de milisegundo; lo que
    corre en paralelo de verdad en la recuperacion es el modelo de embeddings y
    LanceDB, que no pasan por aqui. Un candado es mas simple y no deja
    conexiones colgadas en hilos del pool.
    """

    def __init__(self, data_dir: Path, *, vector_dim: int) -> None:
        """Abre (o crea) la base y aplica las migraciones pendientes.

        Args:
            data_dir: carpeta de datos; se crea si no existe.
            vector_dim: dimensiones del modelo de embeddings vigente.
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.db_path = data_dir / "memory.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()
        self.vectors = VectorIndex(data_dir / "vectors", vector_dim)

    @property
    def schema_version(self) -> int:
        """Cuantas migraciones se aplicaron."""
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def _migrate(self) -> None:
        current = self.schema_version
        for version, (description, script) in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            with self._lock:
                # executescript hace COMMIT antes de correr; por eso el BEGIN
                # va dentro del guion y el user_version al final del mismo.
                self._conn.executescript(
                    f"BEGIN;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;"
                )
            _log.info("memory.migrated", version=version, description=description)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """Una consulta de lectura."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Una escritura suelta. Devuelve las filas afectadas."""
        with self._lock:
            return self._conn.execute(sql, params).rowcount

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Varias escrituras atomicas: o entran todas o ninguna."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def get_meta(self, key: str) -> str | None:
        """Un valor de la tabla `meta`."""
        rows = self.query("SELECT value FROM meta WHERE key = ?", (key,))
        return str(rows[0]["value"]) if rows else None

    def set_meta(self, key: str, value: str) -> None:
        """Guarda un valor en la tabla `meta`."""
        self.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def close(self) -> None:
        """Cierra SQLite. LanceDB no guarda nada abierto entre operaciones."""
        with self._lock:
            self._conn.close()


def dumps(value: Any) -> str:
    """JSON compacto y estable para columnas de texto."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class VectorIndex:
    """La tabla `facts` de LanceDB: id del hecho y su vector normalizado."""

    TABLE = "facts"

    def __init__(self, path: Path, dim: int) -> None:
        """Abre o crea la tabla. Si cambio la dimension, la recrea vacia.

        Recrearla no pierde nada: los hechos viven en SQLite, y quien abre el
        indice (`SemanticMemory`) lo vuelve a llenar si lo encuentra vacio.
        """
        import lancedb
        import pyarrow as pa

        self.dim = dim
        self._lock = threading.Lock()
        self._schema = pa.schema(
            [pa.field("id", pa.string()), pa.field("vector", pa.list_(pa.float32(), dim))]
        )
        self._db = lancedb.connect(str(path))
        existing = set(self._db.list_tables().tables)
        if self.TABLE in existing:
            table = self._db.open_table(self.TABLE)
            if table.schema.field("vector").type.list_size != dim:
                _log.warning("memory.vectors_rebuilt", reason="cambio la dimension del modelo")
                table = self._db.create_table(self.TABLE, schema=self._schema, mode="overwrite")
        else:
            table = self._db.create_table(self.TABLE, schema=self._schema)
        self._table = table

    def count(self) -> int:
        """Vectores guardados."""
        with self._lock:
            return int(self._table.count_rows())

    def add(self, ids: Sequence[str], vectors: NDArray[np.float32]) -> None:
        """Agrega vectores (ya normalizados) en un solo lote."""
        import pyarrow as pa

        if not ids:
            return
        flat = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).ravel())
        batch = pa.table(
            {"id": list(ids), "vector": pa.FixedSizeListArray.from_arrays(flat, self.dim)},
            schema=self._schema,
        )
        with self._lock:
            self._table.add(batch)

    def search(self, vector: NDArray[np.float32], limit: int) -> list[tuple[str, float]]:
        """Los mas parecidos, como (id, coseno). Busqueda exacta, sin indice ANN.

        DECISION: sin indice aproximado. Con 10.000 hechos la busqueda exacta
        tarda ~25 ms en esta CPU y no se equivoca; un IVF necesita miles de
        filas para entrenarse y se desactualiza con cada hecho nuevo.
        """
        with self._lock:
            if self._table.count_rows() == 0:
                return []
            builder = cast(
                "LanceVectorQueryBuilder", self._table.search(np.asarray(vector, dtype=np.float32))
            )
            rows = (
                builder.distance_type("cosine").limit(limit).select(["id", "_distance"]).to_list()
            )
        return [(str(row["id"]), 1.0 - float(row["_distance"])) for row in rows]

    def delete(self, ids: Sequence[str]) -> None:
        """Borra vectores por id."""
        if not ids:
            return
        quoted = ", ".join("'" + item.replace("'", "''") + "'" for item in ids)
        with self._lock:
            self._table.delete(f"id IN ({quoted})")

    def clear(self) -> None:
        """Vacia la tabla (al cambiar de modelo de embeddings)."""
        with self._lock:
            self._table = self._db.create_table(self.TABLE, schema=self._schema, mode="overwrite")
