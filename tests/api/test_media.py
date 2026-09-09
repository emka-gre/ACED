"""Serving video and overlay artifacts, including the range requests seeking depends on."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.models import ArtifactState
from api.routes.media import RangeNotSatisfiableError, parse_range
from tests.api.conftest import VIDEO_BYTES, ManualJobQueue, upload


@pytest.fixture
def succeeded(client: TestClient, queue: ManualJobQueue) -> str:
    analysis_id: str = upload(client).json()["id"]
    queue.run_all()
    return analysis_id


class TestRangeParsing:
    def test_a_closed_range_is_inclusive(self) -> None:
        assert parse_range("bytes=0-99", 1000) == (0, 99)

    def test_an_open_range_runs_to_the_end(self) -> None:
        assert parse_range("bytes=500-", 1000) == (500, 999)

    def test_a_suffix_range_counts_back_from_the_end(self) -> None:
        assert parse_range("bytes=-100", 1000) == (900, 999)

    def test_a_suffix_longer_than_the_file_starts_at_zero(self) -> None:
        assert parse_range("bytes=-5000", 1000) == (0, 999)

    def test_an_end_past_the_file_is_clamped(self) -> None:
        assert parse_range("bytes=900-99999", 1000) == (900, 999)

    def test_a_start_past_the_file_is_unsatisfiable(self) -> None:
        with pytest.raises(RangeNotSatisfiableError):
            parse_range("bytes=1000-", 1000)

    def test_a_reversed_range_is_unsatisfiable(self) -> None:
        with pytest.raises(RangeNotSatisfiableError):
            parse_range("bytes=500-100", 1000)

    @pytest.mark.parametrize(
        "header", ["items=0-10", "bytes=", "bytes=abc-def", "bytes=0-10, 20-30"]
    )
    def test_anything_else_is_not_a_range(self, header: str) -> None:
        """Unparseable and multi-range both fall back to serving the whole file, which is a
        valid answer to a range request and is what every video element can handle."""
        with pytest.raises(ValueError):
            parse_range(header, 1000)


class TestVideoServing:
    def test_the_whole_video_is_served_without_a_range(
        self, client: TestClient, succeeded: str
    ) -> None:
        response = client.get(f"/analyses/{succeeded}/video")

        assert response.status_code == 200
        assert response.content == VIDEO_BYTES
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["content-type"] == "video/mp4"

    def test_a_range_is_served_as_partial_content(self, client: TestClient, succeeded: str) -> None:
        response = client.get(f"/analyses/{succeeded}/video", headers={"Range": "bytes=10-19"})

        assert response.status_code == 206
        assert response.content == VIDEO_BYTES[10:20]
        assert response.headers["content-range"] == f"bytes 10-19/{len(VIDEO_BYTES)}"
        assert response.headers["content-length"] == "10"

    def test_an_unsatisfiable_range_says_how_long_the_file_is(
        self, client: TestClient, succeeded: str
    ) -> None:
        response = client.get(f"/analyses/{succeeded}/video", headers={"Range": "bytes=99999-"})

        assert response.status_code == 416
        assert response.headers["content-range"] == f"bytes */{len(VIDEO_BYTES)}"

    def test_an_unparseable_range_falls_back_to_the_whole_file(
        self, client: TestClient, succeeded: str
    ) -> None:
        response = client.get(f"/analyses/{succeeded}/video", headers={"Range": "pages=1-2"})

        assert response.status_code == 200
        assert response.content == VIDEO_BYTES

    def test_an_unknown_analysis_has_no_video(self, client: TestClient) -> None:
        assert client.get("/analyses/nope/video").status_code == 404

    def test_a_video_deleted_from_under_the_record_is_not_found(
        self, client: TestClient, succeeded: str, storage: Any
    ) -> None:
        storage.delete(client.app.state.context.store.get(succeeded).video_key)

        assert client.get(f"/analyses/{succeeded}/video").status_code == 404


class TestOverlay:
    def test_no_overlay_exists_until_one_is_asked_for(
        self, client: TestClient, succeeded: str, queue: ManualJobQueue
    ) -> None:
        """The spec's second overlay scenario: an analysis that completed with nobody asking
        for an overlay renders none."""
        assert client.get(f"/analyses/{succeeded}").json()["overlay"] == "absent"
        assert queue.jobs == []

    def test_the_first_request_starts_the_render_and_says_so(
        self, client: TestClient, succeeded: str, queue: ManualJobQueue
    ) -> None:
        response = client.get(f"/analyses/{succeeded}/overlay")

        assert response.status_code == 202
        assert response.json()["state"] == "preparing"
        assert len(queue.jobs) == 1
        assert client.get(f"/analyses/{succeeded}").json()["overlay"] == "preparing"

    def test_polling_while_preparing_does_not_queue_a_second_render(
        self, client: TestClient, succeeded: str, queue: ManualJobQueue
    ) -> None:
        client.get(f"/analyses/{succeeded}/overlay")
        client.get(f"/analyses/{succeeded}/overlay")

        assert len(queue.jobs) == 1

    def test_the_overlay_is_served_once_rendered(
        self,
        client: TestClient,
        succeeded: str,
        queue: ManualJobQueue,
        monkeypatch: pytest.MonkeyPatch,
        storage: Any,
    ) -> None:
        rendered = b"OVERLAY-VIDEO-BYTES"

        def fake_render(source: Any, landmarks: Any, classes: Any, destination: Path) -> Any:
            Path(destination).write_bytes(rendered)
            return None

        monkeypatch.setattr("aced.render.overlay.render_overlay", fake_render)
        monkeypatch.setattr("api.services.worker.analyze", lambda *a, **k: _stub())

        client.get(f"/analyses/{succeeded}/overlay")
        queue.run_all()

        assert client.get(f"/analyses/{succeeded}").json()["overlay"] == "ready"
        response = client.get(f"/analyses/{succeeded}/overlay")
        assert response.status_code == 200
        assert response.content == rendered

    def test_a_failed_render_is_reported_and_does_not_retry_forever(
        self,
        client: TestClient,
        succeeded: str,
        queue: ManualJobQueue,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def exploding(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no codec")

        monkeypatch.setattr("aced.render.overlay.render_overlay", exploding)
        monkeypatch.setattr("api.services.worker.analyze", lambda *a, **k: _stub())

        client.get(f"/analyses/{succeeded}/overlay")
        queue.run_all()

        assert client.app.state.context.store.get(succeeded).overlay is ArtifactState.FAILED
        assert client.get(f"/analyses/{succeeded}/overlay").status_code == 409

    def test_no_overlay_before_the_analysis_has_succeeded(self, client: TestClient) -> None:
        analysis_id = upload(client).json()["id"]

        response = client.get(f"/analyses/{analysis_id}/overlay")

        assert response.status_code == 409
        assert "pending" in response.json()["detail"]


def _stub() -> Any:
    from tests.api.conftest import stub_analysis

    return stub_analysis()
