"""Upload, status, report, and deletion."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.api.conftest import VIDEO_BYTES, ManualJobQueue, upload


class TestUpload:
    def test_an_accepted_upload_is_pending_and_addressable(
        self, client: TestClient, queue: ManualJobQueue
    ) -> None:
        response = upload(client)

        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "pending"
        assert response.headers["location"] == f"/analyses/{body['id']}"

        # Accepted, not analyzed: the work is queued and the response did not wait for it.
        assert len(queue.jobs) == 1
        assert client.get(f"/analyses/{body['id']}").json()["state"] == "pending"

    def test_the_video_is_stored_by_its_content(self, client: TestClient, storage: Any) -> None:
        import hashlib

        upload(client)
        digest = hashlib.sha256(VIDEO_BYTES).hexdigest()

        assert storage.exists(f"{digest}.mp4")

    def test_the_same_video_twice_is_stored_once(self, client: TestClient, storage: Any) -> None:
        first = upload(client).json()["id"]
        second = upload(client).json()["id"]

        assert first != second, "each upload is its own analysis"
        stored = [path for path in storage.root.iterdir() if path.is_file()]
        assert len(stored) == 1, "identical bytes are one object"

    @pytest.mark.parametrize(
        ("name", "content_type"),
        [("climb.txt", "text/plain"), ("climb.gif", "image/gif"), ("climb", "video/mp4")],
    )
    def test_an_unsupported_format_is_rejected(
        self, client: TestClient, storage: Any, name: str, content_type: str
    ) -> None:
        response = upload(client, name=name, content_type=content_type)

        assert response.status_code == 415
        assert ".mp4" in response.json()["detail"], "the message names what is supported"
        assert not list(storage.root.glob("*.mp4")), "no record, no stored file"
        assert len(client.app.state.context.store) == 0

    def test_a_declared_content_type_that_is_not_video_is_rejected(
        self, client: TestClient
    ) -> None:
        """The extension can be renamed; the browser's own declaration is a second opinion."""
        response = upload(client, name="climb.mp4", content_type="text/html")

        assert response.status_code == 415

    def test_an_oversized_upload_is_rejected_and_leaves_nothing_behind(
        self, client: TestClient, storage: Any, settings: Any
    ) -> None:
        oversized = b"\x00" * (settings.max_upload_bytes + 1024)

        response = upload(client, content=oversized)

        assert response.status_code == 413
        assert str(settings.max_upload_mb) in response.json()["detail"]
        assert not list(storage.root.glob("*.mp4"))
        assert not list((storage.root / "incoming").iterdir()), "the partial file is removed"
        assert len(client.app.state.context.store) == 0


class TestStatus:
    def test_the_states_move_pending_to_running_to_succeeded(
        self, client: TestClient, queue: ManualJobQueue
    ) -> None:
        analysis_id = upload(client).json()["id"]
        assert client.get(f"/analyses/{analysis_id}").json()["state"] == "pending"

        queue.run_all()

        body = client.get(f"/analyses/{analysis_id}").json()
        assert body["state"] == "succeeded"
        assert body["progress"] == 1.0
        assert body["model"]["bundle_id"] == "test-bundle"

    def test_progress_is_reported_between_zero_and_one(
        self, client: TestClient, queue: ManualJobQueue
    ) -> None:
        analysis_id = upload(client).json()["id"]
        queue.run_all()

        assert 0.0 <= client.get(f"/analyses/{analysis_id}").json()["progress"] <= 1.0

    def test_an_unknown_analysis_is_not_found(self, client: TestClient) -> None:
        assert client.get("/analyses/does-not-exist").status_code == 404

    def test_a_pending_analysis_carries_no_report_field(self, client: TestClient) -> None:
        """Not `report: null` — a client should not have a field to render at all."""
        analysis_id = upload(client).json()["id"]

        assert "report" not in client.get(f"/analyses/{analysis_id}").json()


class TestReport:
    def test_the_report_is_served_once_the_analysis_succeeds(
        self, client: TestClient, queue: ManualJobQueue
    ) -> None:
        analysis_id = upload(client).json()["id"]
        queue.run_all()

        response = client.get(f"/analyses/{analysis_id}/report")

        assert response.status_code == 200
        report = response.json()
        assert report["summary"]["message"]
        assert report["scope"] and report["limitations"]

    def test_no_partial_report_before_the_analysis_finishes(self, client: TestClient) -> None:
        analysis_id = upload(client).json()["id"]

        response = client.get(f"/analyses/{analysis_id}/report")

        assert response.status_code == 409
        assert "pending" in response.json()["detail"]

    def test_the_report_matches_the_one_in_the_status_resource(
        self, client: TestClient, queue: ManualJobQueue
    ) -> None:
        analysis_id = upload(client).json()["id"]
        queue.run_all()

        assert (
            client.get(f"/analyses/{analysis_id}/report").json()
            == client.get(f"/analyses/{analysis_id}").json()["report"]
        )

    def test_an_unknown_analysis_has_no_report(self, client: TestClient) -> None:
        assert client.get("/analyses/nope/report").status_code == 404


class TestDeletion:
    def test_deleting_removes_the_record_and_the_video(
        self, client: TestClient, queue: ManualJobQueue, storage: Any
    ) -> None:
        analysis_id = upload(client).json()["id"]
        queue.run_all()
        key = client.app.state.context.store.get(analysis_id).video_key

        assert client.delete(f"/analyses/{analysis_id}").status_code == 204

        assert not storage.exists(key)
        assert client.get(f"/analyses/{analysis_id}").status_code == 404
        assert client.get(f"/analyses/{analysis_id}/report").status_code == 404

    def test_deleting_something_absent_is_not_found(self, client: TestClient) -> None:
        assert client.delete("/analyses/nope").status_code == 404
