"""The `aced` command run as a process, against a real video file and a real bundle.

Everything else in the CLI suite stubs the pipeline out. This runs the whole thing: a video
encoded on disk, decoded by OpenCV, put through MediaPipe and the shipped model, serialised,
and read back — through a subprocess, so the exit codes are the ones a shell would actually
see rather than the return value of a Python call.

The video is synthesised rather than being the thesis reference climb, which is not committed
(see THESIS_ARCHIVE.md). What that costs is the ability to assert *which* faults come back;
what it keeps is every wiring failure between argparse and the JSON on disk, in a test that
runs anywhere. The reference climb's actual content is asserted in the parity suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aced.inference.bundle import BUNDLE_ENV_VAR, DEFAULT_BUNDLE_ID, models_root

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
FRAME_COUNT = 90
WIDTH, HEIGHT = 320, 240


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the command the way a shell would, with no inherited bundle override."""
    environment = dict(os.environ)
    environment.pop(BUNDLE_ENV_VAR, None)
    return subprocess.run(
        [sys.executable, "-m", "aced.cli.main", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
        timeout=900,
    )


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short, real, decodable video with no climber in it."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer.release()
        pytest.skip("no mp4v encoder available in this OpenCV build")

    try:
        for index in range(FRAME_COUNT):
            frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
            cv2.circle(frame, (index * 3 % WIDTH, HEIGHT // 2), 20, (200, 200, 200), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture(scope="module")
def default_bundle_exists() -> None:
    if not (models_root() / DEFAULT_BUNDLE_ID).is_dir():
        pytest.skip(f"{DEFAULT_BUNDLE_ID} bundle not built — run scripts/build_bundles.py")


class TestSuccessfulRun:
    def test_a_report_and_an_overlay_are_produced(
        self, synthetic_video: Path, default_bundle_exists: None, tmp_path: Path
    ) -> None:
        report_path = tmp_path / "report.json"
        overlay_path = tmp_path / "overlay.mp4"

        result = run_cli(
            "analyze",
            str(synthetic_video),
            "-o",
            str(report_path),
            "--overlay",
            str(overlay_path),
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout == ""

        report = json.loads(report_path.read_text())
        assert set(report) == {
            "video",
            "summary",
            "segments",
            "faults",
            "quality_flags",
            "model",
            "scope",
            "limitations",
        }
        assert report["model"]["bundle_id"] == DEFAULT_BUNDLE_ID
        assert report["video"]["frame_count"] == FRAME_COUNT
        assert report["scope"] and report["limitations"]

        assert overlay_path.exists() and overlay_path.stat().st_size > 0

    def test_the_report_reaches_stdout_when_no_output_is_named(
        self, synthetic_video: Path, default_bundle_exists: None
    ) -> None:
        result = run_cli("analyze", str(synthetic_video), "--quiet")

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["summary"]["message"]

        # `--quiet` silences the command's own summary. It cannot silence TensorFlow, which
        # writes CUDA and oneDNN notices to stderr as it initialises; that noise is exactly
        # why the report goes to stdout and the commentary does not.
        assert report["summary"]["message"] not in result.stderr
        assert "aced:" not in result.stderr


class TestFailureExitCodes:
    """Both failures resolve before the model is loaded, which is also why they are quick."""

    def test_an_unreadable_file_exits_three(self, tmp_path: Path) -> None:
        not_a_video = tmp_path / "notes.txt"
        not_a_video.write_text("this is not a video")
        svm = models_root() / "svm-arm-technique" / "v1"
        if not svm.is_dir():
            pytest.skip("svm bundle not built — run scripts/build_bundles.py")

        result = run_cli("analyze", str(not_a_video), "--bundle", str(svm))

        assert result.returncode == 3
        assert "aced:" in result.stderr
        assert result.stdout == ""

    def test_a_missing_bundle_exits_four(self, synthetic_video: Path, tmp_path: Path) -> None:
        result = run_cli(
            "analyze", str(synthetic_video), "--bundle", str(tmp_path / "no-such-bundle")
        )

        assert result.returncode == 4
        assert "aced:" in result.stderr
        assert result.stdout == ""
