"""Shared machinery for the end-to-end suite.

Every test here runs the real pipeline over a real encoded file. The video is synthesised
rather than the thesis reference climb, which is not committed (see THESIS_ARCHIVE.md) — what
that costs is the ability to assert *which* faults come back; what it keeps is every wiring
failure between the entry point and the report, in tests that run anywhere.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aced.inference.bundle import BUNDLE_ENV_VAR, DEFAULT_BUNDLE_ID, models_root

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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def default_bundle() -> Path:
    path = models_root() / DEFAULT_BUNDLE_ID
    if not path.is_dir():
        pytest.skip(f"{DEFAULT_BUNDLE_ID} bundle not built — run scripts/build_bundles.py")
    return sorted(path.iterdir())[-1]
