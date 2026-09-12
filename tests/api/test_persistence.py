"""Records that survive a restart, and the reconciliation that makes a restart safe."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from api.errors import INTERRUPTED
from api.models import AnalysisRecord, AnalysisState, ArtifactState
from api.services.cleanup import reconcile
from api.services.records import RecordNotFoundError
from api.services.sqlite import Database, SqliteAnalysisStore
from api.services.storage import LocalStorage
from tests.api.conftest import ManualJobQueue, stub_report, upload


def record(**overrides: Any) -> AnalysisRecord:
    base = AnalysisRecord.create(
        filename="climb.mp4", video_key="abc.mp4", size_bytes=10, retention=timedelta(hours=1)
    )
    return base.with_changes(**overrides) if overrides else base


class TestSqliteStore:
    def test_a_record_survives_reopening_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "aced.sqlite3"
        first = Database(path)
        stored = record(state=AnalysisState.SUCCEEDED, report=stub_report(), progress=1.0)
        SqliteAnalysisStore(first).add(stored)
        first.close()

        second = Database(path)
        try:
            back = SqliteAnalysisStore(second).get(stored.id)
        finally:
            second.close()

        assert back == stored
        assert back.report == stub_report()

    def test_a_fresh_file_gets_its_schema(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "nested" / "dir" / "aced.sqlite3")
        try:
            assert len(SqliteAnalysisStore(database)) == 0
        finally:
            database.close()

    def test_reopening_does_not_reapply_the_schema(self, tmp_path: Path) -> None:
        """The migration list is applied once per version. A second open of a populated file
        must find its rows, not a freshly created empty table."""
        path = tmp_path / "aced.sqlite3"
        first = Database(path)
        SqliteAnalysisStore(first).add(record())
        first.close()

        second = Database(path)
        try:
            assert len(SqliteAnalysisStore(second)) == 1
        finally:
            second.close()

    def test_update_is_read_modify_write(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "aced.sqlite3")
        try:
            store = SqliteAnalysisStore(database)
            stored = store.add(record())
            store.update(stored.id, lambda r: r.with_changes(progress=0.5))
            updated = store.update(stored.id, lambda r: r.with_changes(state=AnalysisState.RUNNING))

            # The second update saw the first's progress, not the original record.
            assert updated.progress == 0.5
            assert updated.state is AnalysisState.RUNNING
        finally:
            database.close()

    def test_delete_removes_and_returns(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "aced.sqlite3")
        try:
            store = SqliteAnalysisStore(database)
            stored = store.add(record())

            assert store.delete(stored.id) == stored
            with pytest.raises(RecordNotFoundError):
                store.get(stored.id)
        finally:
            database.close()

    def test_expired_is_a_query_not_a_scan(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "aced.sqlite3")
        try:
            store = SqliteAnalysisStore(database)
            fresh = store.add(record())
            stale = store.add(
                record().with_changes(expires_at=fresh.created_at - timedelta(days=1))
            )

            expired = store.expired()

            assert [r.id for r in expired] == [stale.id]
        finally:
            database.close()

    def test_iteration_is_newest_first(self, tmp_path: Path) -> None:
        database = Database(tmp_path / "aced.sqlite3")
        try:
            store = SqliteAnalysisStore(database)
            older = store.add(record())
            newer = store.add(
                record().with_changes(created_at=older.created_at + timedelta(seconds=1))
            )

            assert [r.id for r in store] == [newer.id, older.id]
        finally:
            database.close()


class TestReconciliation:
    def test_an_interrupted_running_analysis_is_failed(self, storage: LocalStorage) -> None:
        from api.services.records import InMemoryAnalysisStore

        store = InMemoryAnalysisStore()
        stuck = store.add(record(state=AnalysisState.RUNNING, progress=0.4))
        storage.store([b"video"], suffix=".mp4", max_bytes=100)  # unrelated, keeps things honest

        failed, _ = reconcile(store, storage)

        assert failed == [stuck.id]
        settled = store.get(stuck.id)
        assert settled.state is AnalysisState.FAILED
        assert settled.failure_reason == INTERRUPTED

    def test_a_pending_analysis_is_failed_too(self, storage: LocalStorage) -> None:
        """Its job was in an in-process queue that died with the process. Nothing survived to
        pick it up, so leaving it pending is leaving a client polling forever."""
        from api.services.records import InMemoryAnalysisStore

        store = InMemoryAnalysisStore()
        queued = store.add(record(state=AnalysisState.PENDING))

        failed, _ = reconcile(store, storage)

        assert failed == [queued.id]
        assert store.get(queued.id).state is AnalysisState.FAILED

    def test_finished_analyses_are_left_alone(self, storage: LocalStorage) -> None:
        from api.services.records import InMemoryAnalysisStore

        store = InMemoryAnalysisStore()
        done = store.add(record(state=AnalysisState.SUCCEEDED, report=stub_report()))
        broken = store.add(record(state=AnalysisState.FAILED, failure_reason="earlier"))

        failed, _ = reconcile(store, storage)

        assert failed == []
        assert store.get(done.id).state is AnalysisState.SUCCEEDED
        assert store.get(broken.id).failure_reason == "earlier"

    def test_an_overlay_caught_mid_render_goes_back_to_absent(self, storage: LocalStorage) -> None:
        """Not failed: nothing is wrong with the analysis, and a client can simply ask again."""
        from api.services.records import InMemoryAnalysisStore

        store = InMemoryAnalysisStore()
        mid = store.add(
            record(
                state=AnalysisState.SUCCEEDED, report=stub_report(), overlay=ArtifactState.PREPARING
            )
        )

        reconcile(store, storage)

        assert store.get(mid.id).overlay is ArtifactState.ABSENT
        assert store.get(mid.id).state is AnalysisState.SUCCEEDED

    def test_orphaned_media_is_removed_and_named_media_kept(self, storage: LocalStorage) -> None:
        from api.services.records import InMemoryAnalysisStore

        store = InMemoryAnalysisStore()
        kept = storage.store([b"kept"], suffix=".mp4", max_bytes=100)
        overlay = storage.reserve(".mp4")
        overlay.write_bytes(b"overlay")
        orphan = storage.store([b"orphan"], suffix=".mp4", max_bytes=100)
        store.add(record(video_key=kept.key, overlay_key=overlay.name))

        _, removed = reconcile(store, storage)

        assert removed == [orphan.key]
        assert storage.exists(kept.key)
        assert overlay.exists()
        assert not storage.exists(orphan.key)

    def test_the_database_is_never_an_orphan(self, tmp_path: Path) -> None:
        """The database lives under the storage root. Reconciliation must not eat it."""
        storage = LocalStorage(tmp_path / "storage")
        database = Database(tmp_path / "storage" / "db" / "aced.sqlite3")
        try:
            store = SqliteAnalysisStore(database)
            _, removed = reconcile(store, storage)

            assert removed == []
            assert database.path.exists()
        finally:
            database.close()


class TestAcrossARestart:
    """The service, not the store: what a client sees after the process comes back."""

    def test_an_analysis_is_still_addressable_after_a_restart(
        self, settings: Any, storage: LocalStorage, analyze_stub: Any, tmp_path: Path
    ) -> None:
        from fastapi.testclient import TestClient

        from api.app import create_app

        database_path = tmp_path / "db" / "aced.sqlite3"
        first_db = Database(database_path)
        queue = ManualJobQueue()
        app = create_app(
            settings,
            store=SqliteAnalysisStore(first_db),
            storage=storage,
            queue=queue,
            bundle_loader=lambda _: _bundle(),
            start_cleanup=False,
        )
        with TestClient(app) as client:
            analysis_id = upload(client).json()["id"]
            queue.run_all()
            assert client.get(f"/analyses/{analysis_id}").json()["state"] == "succeeded"
        first_db.close()

        second_db = Database(database_path)
        restarted = create_app(
            settings,
            store=SqliteAnalysisStore(second_db),
            storage=storage,
            queue=ManualJobQueue(),
            bundle_loader=lambda _: _bundle(),
            start_cleanup=False,
        )
        try:
            with TestClient(restarted) as client:
                status = client.get(f"/analyses/{analysis_id}")
                assert status.status_code == 200
                assert status.json()["state"] == "succeeded"
                assert client.get(f"/analyses/{analysis_id}/report").status_code == 200
                assert client.get(f"/analyses/{analysis_id}/video").status_code == 200
        finally:
            second_db.close()

    def test_a_restart_settles_what_was_running(
        self, settings: Any, storage: LocalStorage, analyze_stub: Any, tmp_path: Path
    ) -> None:
        from fastapi.testclient import TestClient

        from api.app import create_app

        database_path = tmp_path / "db" / "aced.sqlite3"
        first_db = Database(database_path)
        app = create_app(
            settings,
            store=SqliteAnalysisStore(first_db),
            storage=storage,
            queue=ManualJobQueue(),  # never run: the job "dies" with the process
            bundle_loader=lambda _: _bundle(),
            start_cleanup=False,
        )
        with TestClient(app) as client:
            analysis_id = upload(client).json()["id"]
            assert client.get(f"/analyses/{analysis_id}").json()["state"] == "pending"
        first_db.close()

        second_db = Database(database_path)
        restarted = create_app(
            settings,
            store=SqliteAnalysisStore(second_db),
            storage=storage,
            queue=ManualJobQueue(),
            bundle_loader=lambda _: _bundle(),
            start_cleanup=False,
        )
        try:
            with TestClient(restarted) as client:
                body = client.get(f"/analyses/{analysis_id}").json()
                assert body["state"] == "failed"
                assert body["failure_reason"] == INTERRUPTED
        finally:
            second_db.close()


def _bundle() -> Any:
    from tests.api.conftest import stub_bundle

    return stub_bundle()
