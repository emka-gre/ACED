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

from api.errors import INTERRUPTED
from api.logging import get_logger
from api.models import AnalysisRecord, AnalysisState, ArtifactState
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


def reconcile(store: AnalysisStore, storage: Storage) -> tuple[list[str], list[str]]:
    """Make the state on disk agree with itself after a restart.

    Two things can be wrong, and both come from the same fact: the worker threads and the
    queue live in the process, so when the process dies, so does every job it held.

    An analysis recorded as pending or running is one whose job died with the previous
    process. It will never be picked up — nothing survived to pick it up — so leaving it in
    that state means a status resource that never resolves, which to a client is a hang. It
    becomes failed, with a reason that says what happened. An overlay caught mid-render goes
    back to absent rather than failed: nothing is wrong with the analysis, and a client can
    simply ask again.

    Media that no record names is deleted. The cleanup job can only remove what a record
    points at; this is the pass that catches what fell between a media write and the record
    write that should have followed it, or what the previous store — in memory — forgot.

    Returns (ids failed, keys removed), for the log.
    """
    logger = get_logger()
    failed: list[str] = []
    for record in list(store):
        changes: dict[str, object] = {}
        if record.state in (AnalysisState.PENDING, AnalysisState.RUNNING):
            changes.update(state=AnalysisState.FAILED, failure_reason=INTERRUPTED)
        if record.overlay is ArtifactState.PREPARING:
            changes.update(overlay=ArtifactState.ABSENT)
        if not changes:
            continue

        def settle(current: AnalysisRecord, changes: dict[str, object] = changes) -> AnalysisRecord:
            return current.with_changes(**changes)

        try:
            store.update(record.id, settle)
        except RecordNotFoundError:
            continue
        if "state" in changes:
            failed.append(record.id)
            logger.info("interrupted analysis failed at startup", extra={"analysis_id": record.id})

    named: set[str] = set()
    for record in store:
        named.add(record.video_key)
        if record.overlay_key is not None:
            named.add(record.overlay_key)
    orphaned = sorted(storage.keys() - named)
    for key in orphaned:
        storage.delete(key)
    if orphaned:
        logger.info("orphaned media removed at startup", extra={"count": len(orphaned)})

    return failed, orphaned


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
