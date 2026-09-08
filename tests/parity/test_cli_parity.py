"""The CLI on the reference climb: the same report the library produces, byte for byte.

Task 8.3, and the spec's "its content is equivalent to the report the service would produce
for the same video and bundle". The service is not built yet, but the thing the service will
call is — `aced.analysis.analyze` — so equivalence is asserted against that. When section 9
lands, its worker calls the same function, and section 11's cross-check becomes a matter of
transport rather than of analysis.

Needs the thesis archive, which is not part of this repository (see THESIS_ARCHIVE.md); the
reference video is 20 MB of an identifiable person and is deliberately not committed. The
CLI's behaviour on a video that *is* available is covered in tests/e2e.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from aced.analysis import analyze
from aced.inference.bundle import BUNDLE_ENV_VAR, DEFAULT_BUNDLE_ID, load_bundle, models_root

pytestmark = [pytest.mark.slow, pytest.mark.parity]

ROOT = Path(__file__).resolve().parents[2]

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
    pytest.importorskip("mediapipe", reason="the CLI end-to-end run needs MediaPipe")
    return REFERENCE_VIDEO


@pytest.fixture(scope="module")
def bundle_path() -> Path:
    path = models_root() / DEFAULT_BUNDLE_ID
    if not path.is_dir():
        pytest.skip(f"{DEFAULT_BUNDLE_ID} bundle not built — run scripts/build_bundles.py")
    return sorted(path.iterdir())[-1]


@pytest.fixture(scope="module")
def cli_report(reference_video: Path, bundle_path: Path, tmp_path_factory: Any) -> dict:
    destination = tmp_path_factory.mktemp("cli") / "report.json"
    environment = dict(os.environ)
    environment.pop(BUNDLE_ENV_VAR, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aced.cli.main",
            "analyze",
            str(reference_video),
            "--bundle",
            str(bundle_path),
            "-o",
            str(destination),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(destination.read_text())


@pytest.fixture(scope="module")
def library_report(reference_video: Path, bundle_path: Path) -> dict:
    analysis = analyze(reference_video, load_bundle(bundle_path))
    return json.loads(analysis.report.to_json())


class TestCliMatchesTheLibrary:
    def test_the_reports_are_identical(self, cli_report: dict, library_report: dict) -> None:
        """Not a field-by-field comparison: the CLI must add nothing, drop nothing, and
        reformat nothing on its way out."""
        assert cli_report == library_report

    def test_the_climb_produced_a_real_analysis(self, cli_report: dict) -> None:
        """Guards against the comparison above passing because both sides are empty."""
        assert cli_report["segments"]
        assert cli_report["summary"]["window_count"] == len(cli_report["segments"])
        assert cli_report["video"]["frame_count"] > 0

    def test_the_bundle_is_identified_in_the_report(self, cli_report: dict) -> None:
        assert cli_report["model"]["bundle_id"] == DEFAULT_BUNDLE_ID
