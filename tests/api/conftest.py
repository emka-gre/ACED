"""Fixtures for the API suite.

The pipeline is stubbed throughout: what is under test is the service — states, codes,
storage, isolation — and running MediaPipe over a real video would make the suite slow
without exercising a single line of the code these tests are about. `aced.analysis.analyze`
has its own suite, and the parity suite proves the CLI and the library agree.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aced.analysis import Analysis
from aced.inference.bundle import (
    BundleManifest,
    Confidence,
    ModelBundle,
    ModelKind,
    Scaling,
)
from aced.pose.landmarks import QualitySignals, VideoMeta
from aced.report.build import build_report
from aced.report.types import AnalysisReport
from api.app import create_app
from api.config import Settings
from api.services.queue import Job
from api.services.records import InMemoryAnalysisStore
from api.services.storage import LocalStorage

VIDEO_BYTES = b"\x00\x00\x00\x20ftypmp42" + bytes(range(256)) * 8


def manifest() -> BundleManifest:
    return BundleManifest(
        id="test-bundle",
        version="1.0.0",
        model_kind=ModelKind.FLAT,
        model_file="model.joblib",
        features=("left_elbow_angle",),
        window_length=60,
        stride=20,
        landmarks=(),
        threshold=0.5,
        scaling=Scaling.INTERNAL,
        confidence=Confidence.NONE,
        trained_on="2026-04-07",
        metrics={"f1": 0.81},
    )


def stub_bundle() -> ModelBundle:
    return ModelBundle(manifest=manifest(), model=object(), scaler=None, path=Path("bundle"))


def stub_report(*, frames_detected: int = 600) -> AnalysisReport:
    return build_report(
        video=VideoMeta(fps=30.0, frame_count=600, width=1080, height=1920),
        quality=QualitySignals(
            frames_read=600, frames_detected=frames_detected, mean_visibility=0.9
        ),
        windows=[],
        results=[],
        manifest=manifest(),
    )


def stub_analysis() -> Analysis:
    return Analysis(report=stub_report(), landmarks=(), frame_classes={}, windows=())


class ManualJobQueue:
    """Holds jobs until a test runs them, so intermediate states can be observed.

    A real queue makes `pending` a race: with an immediate queue the analysis is finished
    before the upload response is read, and with a thread pool it may or may not be. Neither
    lets a test assert what a client sees while it waits.
    """

    def __init__(self) -> None:
        self.jobs: list[Job] = []
        self.shutdown_called = False

    def submit(self, job: Job) -> None:
        self.jobs.append(job)

    def run_next(self) -> None:
        self.jobs.pop(0)()

    def run_all(self) -> None:
        while self.jobs:
            self.run_next()

    def shutdown(self, *, wait: bool = True) -> None:
        self.shutdown_called = True


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "storage")


@pytest.fixture
def queue() -> ManualJobQueue:
    return ManualJobQueue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        storage_root=tmp_path / "storage",
        max_upload_bytes=4096,
        retention=timedelta(hours=1),
        worker_count=1,
    )


@pytest.fixture
def analyze_stub(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Analysis]:
    """Replace the pipeline with something instant and deterministic."""

    def fake_analyze(source: Any, bundle: Any, **kwargs: Any) -> Analysis:
        on_progress = kwargs.get("on_progress")
        if on_progress is not None:
            on_progress(0.5)
        return stub_analysis()

    monkeypatch.setattr("api.services.worker.analyze", fake_analyze)
    return fake_analyze


@pytest.fixture
def make_client(
    settings: Settings, storage: LocalStorage, queue: ManualJobQueue
) -> Iterator[Callable[..., TestClient]]:
    """Build a service against this test's settings, storage and queue."""
    clients: list[TestClient] = []

    def build(
        *,
        bundle_loader: Callable[[Settings], ModelBundle] | None = None,
        job_queue: Any = None,
        store: Any = None,
        web_root: Path | None = None,
    ) -> TestClient:
        app = create_app(
            replace(settings, web_root=web_root) if web_root is not None else settings,
            store=store if store is not None else InMemoryAnalysisStore(),
            storage=storage,
            queue=job_queue if job_queue is not None else queue,
            bundle_loader=bundle_loader or (lambda _: stub_bundle()),
            start_cleanup=False,
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient], analyze_stub: Any) -> TestClient:
    return make_client()


def upload(
    client: TestClient,
    *,
    name: str = "climb.mp4",
    content: bytes | None = None,
    content_type: str = "video/mp4",
) -> Any:
    return client.post(
        "/analyses",
        files={"file": (name, content if content is not None else VIDEO_BYTES, content_type)},
    )


def wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    """Poll a condition that a background thread is expected to satisfy."""
    deadline = threading.Event()
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        deadline.wait(0.02)
        waited += 0.02
    return predicate()
