"""Pin the packaged extractor against the recorded research pose output.

This is the only test that runs MediaPipe over a real video, so it carries both the `slow`
and `parity` markers and is excluded from the CI run. It needs the thesis archive, which is
not part of this repository — see THESIS_ARCHIVE.md. The reference video is 20 MB of an
identifiable person and is deliberately not committed.

What it proves: `PoseExtractor` on `video_62_B_Ema.MOV` reproduces, landmark for landmark,
what `streamlit/pose_extraction.py` produced on the same file. Without it, the unit suite
only shows that the extractor is internally consistent with its own fakes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from aced.pose.extractor import extract_poses
from aced.pose.landmarks import MEDIAPIPE_INDEX, Joint
from tests.conftest import reference_file

pytestmark = [pytest.mark.slow, pytest.mark.parity]

#: Same default as scripts/generate_reference.py, overridable for a different checkout.
ARCHIVE = Path(
    os.environ.get(
        "ACED_THESIS_ARCHIVE",
        Path.home() / "Desktop" / "aced-thesis-archive" / "extracted" / "bachelor-project-main",
    )
)
REFERENCE_VIDEO = ARCHIVE / "video_62_B_Ema.MOV"

#: MediaPipe is deterministic for a fixed version and input, and both are pinned, so this is
#: a guard against drift rather than a tolerance for genuine numerical variation.
ABS_TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def recorded() -> pd.DataFrame:
    return pd.read_csv(reference_file("pose_reference.csv"))


@pytest.fixture(scope="module")
def extracted() -> list:
    if not REFERENCE_VIDEO.exists():
        pytest.skip(f"reference video not available at {REFERENCE_VIDEO} — see THESIS_ARCHIVE.md")
    pytest.importorskip("mediapipe", reason="pose parity needs MediaPipe")

    extraction = extract_poses(REFERENCE_VIDEO)
    return list(extraction)


def test_same_frames_are_detected(extracted: list, recorded: pd.DataFrame) -> None:
    """Detection has to agree frame for frame, including which frames were dropped."""
    assert [record.index for record in extracted] == recorded["frame"].tolist()


def test_retained_landmarks_match_the_reference(extracted: list, recorded: pd.DataFrame) -> None:
    rows = recorded.set_index("frame")

    for record in extracted:
        row = rows.loc[record.index]
        for joint, index in MEDIAPIPE_INDEX.items():
            landmark = record.joints[joint]
            assert landmark.x == pytest.approx(row[f"x_{index}"], abs=ABS_TOLERANCE), (
                f"frame {record.index} {joint.value} x"
            )
            assert landmark.y == pytest.approx(row[f"y_{index}"], abs=ABS_TOLERANCE), (
                f"frame {record.index} {joint.value} y"
            )
            assert landmark.z == pytest.approx(row[f"z_{index}"], abs=ABS_TOLERANCE), (
                f"frame {record.index} {joint.value} z"
            )
            assert landmark.visibility == pytest.approx(row[f"vis_{index}"], abs=ABS_TOLERANCE), (
                f"frame {record.index} {joint.value} visibility"
            )


def test_extraction_is_structurally_interchangeable_with_the_landmark_csv(
    extracted: list,
) -> None:
    """Video-sourced and CSV-sourced records must be the same shape for downstream stages.

    Deliberately not an equality check. `fixtures/landmarks/reference_62_B.csv` is the
    archive's labeling CSV (`labeling/csv_pose/mediapipe/bad/pose_data_essai_62_B_Ema.csv`),
    a *different extraction run* from the one this test reproduces:

        pose_reference.csv    620 rows, frames 0-619   research extractor over the video
        reference_62_B.csv    619 rows, frames 1-619   archive labeling pipeline

    The two disagree numerically — ~0.67 in normalized coordinates on the first frames,
    settling to 1e-2..1e-1 later, which is MediaPipe's tracker converging from a different
    start rather than an off-by-one (shifting either sequence by one frame does not align
    them). So the feature and prediction parity references, which are generated from the
    labeling CSV, are not reproducible from the video end to end. That gap belongs to
    integration (task 11); here we only assert the two sources produce the same structure.
    """
    from aced.pose.csv_source import read_landmark_csv
    from tests.conftest import LANDMARKS

    from_csv = read_landmark_csv(LANDMARKS / "reference_62_B.csv")

    assert extracted, "the video produced no records"
    for record in (extracted[0], from_csv[0]):
        assert set(record.joints) == set(Joint)
    assert [record.index for record in extracted] == sorted(r.index for r in extracted)
    # Frame indices are the video's own, and start at 0 where the labeling CSV starts at 1.
    assert extracted[0].index == 0
    assert from_csv[0].index == 1
