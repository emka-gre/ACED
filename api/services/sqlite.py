"""Persistence in one SQLite file.

The development store was a dictionary behind a lock, and that was right for a laptop: a
restart on a laptop is the lid closing. On a server a restart is a deployment, and a store
that forgets every analysis while the videos it named stay on disk turns each deploy into a
small leak of other people's footage. This is the store that remembers.

One file, one connection, one lock. SQLite serialises writes anyway, and the service's write
rate is a progress update every twenty frames from a couple of worker threads — a single
guarded connection is not the bottleneck and is far easier to reason about than a pool. The
day this matters is the day there are fifty concurrent analyses, and that day the store
changes again, to something with a server behind it.

Records are stored whole as JSON, with the fields the store has to search on — state, expiry,
and in the next section the owner — lifted into indexed columns. A schema that mirrored every
field would have to change every time the record did; this one changes when a *query* does,
which is rarer and is the thing a schema is actually for.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

from api.models import AnalysisRecord, AnalysisState, utcnow
from api.services.records import RecordNotFoundError

DATABASE_FILENAME = "aced.sqlite3"

#: Applied in order on open; each entry is one version. Adding a version is how the schema
#: changes — an existing file gets only the versions it has not seen, and a fresh file gets
#: them all. Never edit an entry that has shipped.
SCHEMA_VERSIONS: tuple[str, ...] = (
    """
    CREATE TABLE analyses (
        id          TEXT PRIMARY KEY,
        state       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL,
        video_key   TEXT NOT NULL,
        overlay_key TEXT,
        payload     TEXT NOT NULL
    );
    CREATE INDEX analyses_state ON analyses (state);
    CREATE INDEX analyses_expires_at ON analyses (expires_at);
    """,
)


class Database:
    """One SQLite file, opened once, shared by every store that lives in it."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # `check_same_thread=False` because worker threads write progress; the lock below is
        # what actually makes that safe. WAL lets the request threads read while a worker
        # writes instead of waiting on it.
        self._connection = sqlite3.connect(str(self._path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._migrate()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def _migrate(self) -> None:
        current = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        for version, statements in enumerate(SCHEMA_VERSIONS[current:], start=current + 1):
            self._connection.executescript(statements)
            self._connection.execute(f"PRAGMA user_version={version}")
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class SqliteAnalysisStore:
    """`AnalysisStore` on a `Database`. Same contract as the in-memory store, same locking
    discipline: `update` is a read-modify-write under the lock."""

    def __init__(self, database: Database) -> None:
        self._db = database

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AnalysisRecord:
        return AnalysisRecord.from_dict(json.loads(row["payload"]))

    def _write(self, record: AnalysisRecord) -> None:
        self._db.connection.execute(
            """
            INSERT INTO analyses
                (id, state, created_at, expires_at, video_key, overlay_key, payload)
            VALUES
                (:id, :state, :created_at, :expires_at, :video_key, :overlay_key, :payload)
            ON CONFLICT (id) DO UPDATE SET
                state = excluded.state,
                expires_at = excluded.expires_at,
                video_key = excluded.video_key,
                overlay_key = excluded.overlay_key,
                payload = excluded.payload
            """,
            {
                "id": record.id,
                "state": record.state.value,
                "created_at": record.created_at.isoformat(),
                "expires_at": record.expires_at.isoformat(),
                "video_key": record.video_key,
                "overlay_key": record.overlay_key,
                "payload": json.dumps(record.to_dict()),
            },
        )
        self._db.connection.commit()

    def add(self, record: AnalysisRecord) -> AnalysisRecord:
        with self._db.lock:
            self._write(record)
        return record

    def get(self, analysis_id: str) -> AnalysisRecord:
        with self._db.lock:
            row = self._db.connection.execute(
                "SELECT payload FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFoundError(analysis_id)
        return self._row_to_record(row)

    def update(
        self, analysis_id: str, change: Callable[[AnalysisRecord], AnalysisRecord]
    ) -> AnalysisRecord:
        with self._db.lock:
            current = self.get(analysis_id)
            updated = change(current)
            self._write(updated)
            return updated

    def delete(self, analysis_id: str) -> AnalysisRecord:
        with self._db.lock:
            record = self.get(analysis_id)
            self._db.connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            self._db.connection.commit()
            return record

    def expired(self, now: datetime | None = None) -> list[AnalysisRecord]:
        moment = (now or utcnow()).isoformat()
        with self._db.lock:
            rows = self._db.connection.execute(
                "SELECT payload FROM analyses WHERE expires_at <= ?", (moment,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def in_state(self, state: AnalysisState) -> list[AnalysisRecord]:
        with self._db.lock:
            rows = self._db.connection.execute(
                "SELECT payload FROM analyses WHERE state = ?", (state.value,)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def media_keys(self) -> set[str]:
        """Every storage key some record names — what reconciliation keeps."""
        with self._db.lock:
            rows = self._db.connection.execute(
                "SELECT video_key, overlay_key FROM analyses"
            ).fetchall()
        keys: set[str] = set()
        for row in rows:
            keys.add(row["video_key"])
            if row["overlay_key"] is not None:
                keys.add(row["overlay_key"])
        return keys

    def __iter__(self) -> Iterator[AnalysisRecord]:
        with self._db.lock:
            rows = self._db.connection.execute(
                "SELECT payload FROM analyses ORDER BY created_at DESC"
            ).fetchall()
        return iter([self._row_to_record(row) for row in rows])

    def __len__(self) -> int:
        with self._db.lock:
            return int(self._db.connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])
