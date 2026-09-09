"""Deleting analyses that have passed their expiry.

Retention is a field on the record from day one because these are videos of identifiable
people in gyms, and "we will add deletion later" never happens. This is the process that
makes the field mean something.

Deletion is idempotent and order-sensitive: media first, then the record. A crash between the
two leaves an orphaned record pointing at absent media, which the API reports as not found;
the reverse order would leave a video on disk that nothing remembers to delete.
"""

from __future__ import annotations

import threading
from datetime import datetime

from api.logging import get_logger
from api.models import AnalysisRecord
from api.services.records import AnalysisStore, RecordNotFoundError
from api.services.storage import Storage


def delete_media(storage: Storage, record: AnalysisRecord) -> None:
    """Remove everything on disk belonging to one analysis."""
    storage.delete(record.video_key)
    if record.overlay_key is not None:
        storage.delete(record.overlay_key)


def purge_expired(store: AnalysisStore, storage: Storage, now: datetime | None = None) -> list[str]:
    """Delete every expired analysis. Returns the ids removed."""
    removed: list[str] = []
    for record in store.expired(now):
        delete_media(storage, record)
        try:
            store.delete(record.id)
        except RecordNotFoundError:
            continue
        removed.append(record.id)

    if removed:
        get_logger().info("expired analyses purged", extra={"count": len(removed)})
    return removed


class CleanupScheduler:
    """Runs `purge_expired` on an interval, on a daemon thread.

    A timer rather than a cron entry or a separate process: retention here is measured in
    hours and the check is a dictionary scan, so the simplest thing that cannot be forgotten
    to deploy is the right one.
    """

    def __init__(
        self,
        store: AnalysisStore,
        storage: Storage,
        interval_seconds: float = 300.0,
    ) -> None:
        self._store = store
        self._storage = storage
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="aced-cleanup", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                purge_expired(self._store, self._storage)
            except Exception:
                get_logger().exception("cleanup sweep failed")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
