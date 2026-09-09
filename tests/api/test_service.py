"""Health, failure isolation, concurrency, retention, and what a client is told on failure."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aced.pose.extractor import UnreadableVideoError
from api.config import Settings
from api.errors import UNEXPECTED, UNREADABLE_VIDEO
from api.models import AnalysisState, utcnow
from api.services.cleanup import purge_expired
from api.services.queue import InProcessJobQueue
from tests.api.conftest import ManualJobQueue, stub_analysis, upload, wait_for


class TestHealth:
    def test_a_loaded_bundle_is_reported_with_its_provenance(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["bundle"]["bundle"] == "test-bundle"
        assert body["bundle"]["version"] == "1.0.0"
        assert body["bundle"]["metrics"] == {"f1": 0.81}

    def test_a_bundle_that_will_not_load_makes_the_service_unhealthy(
        self, make_client: Any
    ) -> None:
        def broken(_: Settings) -> Any:
            raise RuntimeError("model.keras is truncated")

        client = make_client(bundle_loader=broken)
        response = client.get("/health")

        assert response.status_code == 503
        assert response.json()["status"] == "unhealthy"

    def test_an_unhealthy_service_refuses_analyses_it_cannot_run(
        self, make_client: Any, storage: Any
    ) -> None:
        """Storing someone's video only to fail it is worse than refusing it now."""

        def broken(_: Settings) -> Any:
            raise RuntimeError("no bundle")

        client = make_client(bundle_loader=broken)

        response = upload(client)

        assert response.status_code == 503
        assert not list(storage.root.glob("*.mp4"))

    def test_the_service_still_answers_when_it_is_unhealthy(self, make_client: Any) -> None:
        """Refusing to start would take the health endpoint down with it, leaving an
        operator with a container that exits and no statement of why."""

        def broken(_: Settings) -> Any:
            raise RuntimeError("no bundle")

        client = make_client(bundle_loader=broken)

        assert client.get("/health").status_code == 503
        assert client.get("/analyses/nope").status_code == 404


class TestFailureIsolation:
    def test_a_failing_analysis_reaches_a_terminal_state(
        self, client: TestClient, queue: ManualJobQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def exploding(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("keras exploded")

        monkeypatch.setattr("api.services.worker.analyze", exploding)
        analysis_id = upload(client).json()["id"]

        queue.run_all()

        body = client.get(f"/analyses/{analysis_id}").json()
        assert body["state"] == "failed"
        assert body["failure_reason"] == UNEXPECTED

    def test_a_worker_crash_does_not_stop_the_service(
        self, client: TestClient, queue: ManualJobQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that dies hard — not an exception the pipeline anticipated — still leaves a
        terminal record, and the next analysis runs normally."""
        crashed = upload(client).json()["id"]

        def crashing(*args: Any, **kwargs: Any) -> Any:
            raise MemoryError("out of memory")

        monkeypatch.setattr("api.services.worker.analyze", crashing)
        queue.run_all()

        monkeypatch.setattr("api.services.worker.analyze", lambda *a, **k: stub_analysis())
        healthy = upload(client).json()["id"]
        queue.run_all()

        assert client.get(f"/analyses/{crashed}").json()["state"] == "failed"
        assert client.get(f"/analyses/{healthy}").json()["state"] == "succeeded"

    def test_nothing_is_left_running_forever(
        self, client: TestClient, queue: ManualJobQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A status resource that never resolves is indistinguishable from a hung service."""
        monkeypatch.setattr(
            "api.services.worker.analyze",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        analysis_id = upload(client).json()["id"]

        queue.run_all()

        assert client.app.state.context.store.get(analysis_id).state.is_terminal

    def test_deleting_an_analysis_mid_run_does_not_resurrect_it(
        self, client: TestClient, queue: ManualJobQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        analysis_id = upload(client).json()["id"]

        def deleting_analyze(source: Any, bundle: Any, **kwargs: Any) -> Any:
            client.delete(f"/analyses/{analysis_id}")
            return stub_analysis()

        monkeypatch.setattr("api.services.worker.analyze", deleting_analyze)
        queue.run_all()

        assert client.get(f"/analyses/{analysis_id}").status_code == 404


class TestClientSafeFailures:
    def test_an_unreadable_video_is_explained_in_the_client_s_terms(
        self, client: TestClient, queue: ManualJobQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "api.services.worker.analyze",
            lambda *a, **k: (_ for _ in ()).throw(
                UnreadableVideoError("could not open /srv/storage/abc123.mp4")
            ),
        )
        analysis_id = upload(client).json()["id"]
        queue.run_all()

        reason = client.get(f"/analyses/{analysis_id}").json()["failure_reason"]
        assert reason == UNREADABLE_VIDEO

    def test_internal_detail_never_reaches_the_client(
        self, client: TestClient, queue: ManualJobQueue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A traceback can name filesystem paths, model internals and library versions.
        None of it helps the climber, and some of it a public service should not volunteer."""
        secret = "/srv/models/lstm/v1/model.keras line 402"
        monkeypatch.setattr(
            "api.services.worker.analyze",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError(secret)),
        )
        analysis_id = upload(client).json()["id"]
        queue.run_all()

        body = client.get(f"/analyses/{analysis_id}").text
        assert secret not in body
        assert "model.keras" not in body


class TestConcurrency:
    def test_analyses_are_tracked_independently(
        self, make_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slow(source: Any, bundle: Any, **kwargs: Any) -> Any:
            time.sleep(0.05)
            return stub_analysis()

        monkeypatch.setattr("api.services.worker.analyze", slow)
        pool = InProcessJobQueue(workers=2)
        client = make_client(job_queue=pool)

        ids = [
            upload(client, content=b"climb-%d" % index + b"\x00" * 64).json()["id"]
            for index in range(4)
        ]
        pool.drain(timeout=30)

        assert len(set(ids)) == 4
        states = {i: client.get(f"/analyses/{i}").json()["state"] for i in ids}
        assert set(states.values()) == {"succeeded"}

    def test_one_failing_analysis_does_not_affect_another(
        self, make_client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def selective(source: Any, bundle: Any, **kwargs: Any) -> Any:
            if b"bad" in Path(source).read_bytes():
                raise RuntimeError("this one is broken")
            return stub_analysis()

        monkeypatch.setattr("api.services.worker.analyze", selective)
        pool = InProcessJobQueue(workers=2)
        client = make_client(job_queue=pool)

        bad = upload(client, content=b"bad" + b"\x00" * 64).json()["id"]
        good = upload(client, content=b"good" + b"\x00" * 64).json()["id"]
        pool.drain(timeout=30)

        assert wait_for(lambda: client.get(f"/analyses/{bad}").json()["state"] == "failed")
        assert client.get(f"/analyses/{good}").json()["state"] == "succeeded"


class TestRetention:
    def test_an_expired_analysis_and_its_media_are_purged(
        self, client: TestClient, queue: ManualJobQueue, storage: Any
    ) -> None:
        analysis_id = upload(client).json()["id"]
        queue.run_all()
        context = client.app.state.context
        key = context.store.get(analysis_id).video_key

        context.store.update(
            analysis_id,
            lambda record: record.with_changes(expires_at=utcnow() - timedelta(seconds=1)),
        )
        removed = purge_expired(context.store, storage)

        assert removed == [analysis_id]
        assert not storage.exists(key)
        assert client.get(f"/analyses/{analysis_id}").status_code == 404

    def test_an_unexpired_analysis_survives_a_sweep(
        self, client: TestClient, queue: ManualJobQueue, storage: Any
    ) -> None:
        analysis_id = upload(client).json()["id"]
        queue.run_all()
        context = client.app.state.context

        assert purge_expired(context.store, storage) == []
        assert client.get(f"/analyses/{analysis_id}").status_code == 200

    def test_every_record_carries_an_expiry_from_the_moment_it_exists(
        self, client: TestClient, settings: Settings
    ) -> None:
        """Retention is a field on the record from day one, not something added later."""
        analysis_id = upload(client).json()["id"]
        record = client.app.state.context.store.get(analysis_id)

        assert record.expires_at - record.created_at == settings.retention
        assert record.state is AnalysisState.PENDING

    def test_a_sweep_removes_the_overlay_too(
        self, client: TestClient, queue: ManualJobQueue, storage: Any
    ) -> None:
        analysis_id = upload(client).json()["id"]
        queue.run_all()
        context = client.app.state.context

        overlay = storage.reserve(".mp4")
        overlay.write_bytes(b"overlay")
        context.store.update(
            analysis_id,
            lambda record: record.with_changes(
                overlay_key=overlay.name, expires_at=utcnow() - timedelta(seconds=1)
            ),
        )
        purge_expired(context.store, storage)

        assert not overlay.exists()
