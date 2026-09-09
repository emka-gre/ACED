"""Task 11.1 against the reference climb: the API's report equals the CLI's.

The same comparison as `tests/e2e/test_api_cli_parity.py`, on a real climb with real faults in
it rather than a synthesised video. That is the difference that matters: on footage with no
climber both sides agree trivially, because both produce nothing. Here they have to agree on
merged fault spans, attributions, and rendered explanations.

Needs the thesis archive, which is not part of this repository — see THESIS_ARCHIVE.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aced.inference.bundle import DEFAULT_BUNDLE_ID, models_root
from api.app import create_app
from api.config import Settings
from api.services.queue import ImmediateJobQueue
from tests.e2e.conftest import run_cli

pytestmark = [pytest.mark.slow, pytest.mark.parity]

#: Same default as scripts/generate_reference.py, overridable for a different checkout.
ARCHIVE = Path(
    os.environ.get(
        "ACED_THESIS_ARCHIVE",
        Path.home() / "Desktop" / "aced-thesis-archive" / "extracted" / "bachelor-project-main",
    )
)
REFERENCE_VIDEO = ARCHIVE / "video_62_B_Ema.MOV"


@pytest.fixture(scope="module")
def reference_video() -> Path:
    if not REFERENCE_VIDEO.exists():
        pytest.skip(f"reference video not available at {REFERENCE_VIDEO} — see THESIS_ARCHIVE.md")
    pytest.importorskip("mediapipe", reason="the reference run needs MediaPipe")
    return REFERENCE_VIDEO


@pytest.fixture(scope="module")
def bundle_path() -> Path:
    path = models_root() / DEFAULT_BUNDLE_ID
    if not path.is_dir():
        pytest.skip(f"{DEFAULT_BUNDLE_ID} bundle not built — run scripts/build_bundles.py")
    return sorted(path.iterdir())[-1]


@pytest.fixture(scope="module")
def cli_report(
    reference_video: Path, bundle_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict:
    destination = tmp_path_factory.mktemp("cli") / "report.json"
    result = run_cli(
        "analyze", str(reference_video), "--bundle", str(bundle_path), "-o", str(destination)
    )
    assert result.returncode == 0, result.stderr
    return json.loads(destination.read_text())


@pytest.fixture(scope="module")
def api_report(
    reference_video: Path, bundle_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict:
    settings = Settings(
        storage_root=tmp_path_factory.mktemp("storage"),
        bundle_path=bundle_path,
        max_upload_bytes=512 * 1024 * 1024,
    )
    app = create_app(settings, queue=ImmediateJobQueue(), start_cleanup=False)

    with TestClient(app) as client:
        with reference_video.open("rb") as handle:
            created = client.post(
                "/analyses",
                files={"file": (reference_video.name, handle, "video/quicktime")},
            )
        assert created.status_code == 202, created.text
        analysis_id = created.json()["id"]

        status = client.get(f"/analyses/{analysis_id}").json()
        assert status["state"] == "succeeded", status

        report = client.get(f"/analyses/{analysis_id}/report")
        assert report.status_code == 200
        return report.json()


class TestReferenceClimb:
    def test_the_reports_are_identical(self, api_report: dict, cli_report: dict) -> None:
        assert api_report == cli_report

    def test_the_climb_actually_produced_findings(self, api_report: dict) -> None:
        """The comparison above is only worth something if there was something to compare."""
        assert api_report["segments"]
        assert api_report["summary"]["window_count"] > 0

    def test_every_fault_carries_an_explanation(self, api_report: dict) -> None:
        for fault in api_report["faults"]:
            assert fault["explanation"]
            assert fault["time_end"] >= fault["time_start"]
