"""The API and the CLI must produce the same report for the same video and bundle.

Task 11.1. This is the claim the whole layering rests on: `aced analyze` and the API worker
are two callers of one function, and a climber's report does not depend on which door they
came in by. If these ever diverge, something has grown a second copy of the pipeline.

The comparison is exact. Not "the same fault count" or "the same summary" — the same JSON.
Anything weaker would let a field drift on one side unnoticed.

Both sides run for real here: a real encoded file, real MediaPipe, the real shipped bundle,
the real HTTP handlers. The video is synthesised rather than the thesis reference climb, which
is not committed; `tests/parity/test_api_parity.py` is the same comparison against the
reference climb, gated on the archive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import Settings
from api.services.queue import ImmediateJobQueue
from tests.e2e.conftest import run_cli

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cli_report(
    synthetic_video: Path, default_bundle: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict:
    destination = tmp_path_factory.mktemp("cli") / "report.json"
    result = run_cli(
        "analyze",
        str(synthetic_video),
        "--bundle",
        str(default_bundle),
        "-o",
        str(destination),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(destination.read_text())


@pytest.fixture(scope="module")
def api_report(
    synthetic_video: Path, default_bundle: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict:
    """Upload through the API and poll to completion, as a client would.

    The queue runs the job on the calling thread, so by the time the upload response is read
    the analysis has finished. That is not a shortcut around the async design — the states are
    exercised in the API suite — it just removes a sleep from a test that is about the report.
    """
    settings = Settings(
        storage_root=tmp_path_factory.mktemp("storage"),
        bundle_path=default_bundle,
    )
    app = create_app(settings, queue=ImmediateJobQueue(), start_cleanup=False)

    with TestClient(app) as client:
        with synthetic_video.open("rb") as handle:
            created = client.post(
                "/analyses",
                files={"file": (synthetic_video.name, handle, "video/mp4")},
            )
        assert created.status_code == 202, created.text
        analysis_id = created.json()["id"]

        status = client.get(f"/analyses/{analysis_id}")
        assert status.status_code == 200
        assert status.json()["state"] == "succeeded", status.json()

        report = client.get(f"/analyses/{analysis_id}/report")
        assert report.status_code == 200
        return report.json()


class TestSameReportEitherWay:
    def test_the_reports_are_identical(self, api_report: dict, cli_report: dict) -> None:
        assert api_report == cli_report

    def test_the_comparison_is_not_vacuous(self, api_report: dict) -> None:
        """Guards against both sides being empty for the same uninteresting reason."""
        assert set(api_report) == {
            "video",
            "summary",
            "segments",
            "faults",
            "quality_flags",
            "model",
            "scope",
            "limitations",
        }
        assert api_report["video"]["frame_count"] > 0
        assert api_report["summary"]["message"]
        assert api_report["scope"] and api_report["limitations"]

    def test_both_name_the_same_bundle(self, api_report: dict, cli_report: dict) -> None:
        assert api_report["model"] == cli_report["model"]
        assert api_report["model"]["bundle_id"]
