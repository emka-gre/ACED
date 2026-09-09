"""Persistence for analysis records.

An in-process dictionary behind a lock, which is the development implementation the change
calls for. It is an interface rather than a bare dict because the swap to a real database is
a swap of this one class, and because the locking discipline — every mutation is a
read-modify-replace under the same lock — is the part that has to be right whatever the
backing store is.

`update` takes a function rather than fields so that a worker's progress write cannot clobber
a concurrent state change: the callback runs inside the lock, on the current record.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Protocol

from api.models import AnalysisRecord, utcnow


class RecordNotFoundError(KeyError):
    """No analysis exists with that id — or it has been deleted, which is the same answer."""


class AnalysisStore(Protocol):
    def add(self, record: AnalysisRecord) -> AnalysisRecord: ...

    def get(self, analysis_id: str) -> AnalysisRecord: ...

    def update(
        self, analysis_id: str, change: Callable[[AnalysisRecord], AnalysisRecord]
    ) -> AnalysisRecord: ...

    def delete(self, analysis_id: str) -> AnalysisRecord: ...

    def expired(self, now: datetime | None = None) -> list[AnalysisRecord]: ...

    def __iter__(self) -> Iterator[AnalysisRecord]: ...

    def __len__(self) -> int: ...


class InMemoryAnalysisStore:
    """Records held in this process, guarded by one lock."""

    def __init__(self) -> None:
        self._records: dict[str, AnalysisRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: AnalysisRecord) -> AnalysisRecord:
        with self._lock:
            self._records[record.id] = record
        return record

    def get(self, analysis_id: str) -> AnalysisRecord:
        with self._lock:
            try:
                return self._records[analysis_id]
            except KeyError:
                raise RecordNotFoundError(analysis_id) from None

    def update(
        self, analysis_id: str, change: Callable[[AnalysisRecord], AnalysisRecord]
    ) -> AnalysisRecord:
        """Apply `change` to the current record atomically.

        Read-modify-write under the lock, so a progress update computed from a stale copy
        cannot resurrect a state the record has already moved past.
        """
        with self._lock:
            try:
                current = self._records[analysis_id]
            except KeyError:
                raise RecordNotFoundError(analysis_id) from None
            updated = change(current)
            self._records[analysis_id] = updated
            return updated

    def delete(self, analysis_id: str) -> AnalysisRecord:
        with self._lock:
            try:
                return self._records.pop(analysis_id)
            except KeyError:
                raise RecordNotFoundError(analysis_id) from None

    def expired(self, now: datetime | None = None) -> list[AnalysisRecord]:
        moment = now or utcnow()
        with self._lock:
            return [record for record in self._records.values() if record.is_expired(moment)]

    def __iter__(self) -> Iterator[AnalysisRecord]:
        with self._lock:
            return iter(list(self._records.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
