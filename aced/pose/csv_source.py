"""Read landmark records from a CSV instead of from a video.

Every stage after pose extraction consumes `FrameLandmarks`, so feeding it from a recorded
CSV lets features, windowing, scaling, and inference be exercised — and pinned against the
research pipeline — with no MediaPipe and no video file. That is what keeps the parity suite
runnable in CI, where neither is available.

The expected columns are the ones the research extractors wrote: `frame`, plus `x_<i>`,
`y_<i>`, `z_<i>`, and `vis_<i>` for each MediaPipe landmark index.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from aced.pose.landmarks import (
    DEFAULT_FPS,
    MEDIAPIPE_INDEX,
    FrameLandmarks,
    Landmark,
    QualitySignals,
)


class MalformedLandmarkCsvError(ValueError):
    """The CSV is missing columns the retained landmarks need."""


def _require_columns(frame: pd.DataFrame) -> None:
    required = ["frame"]
    for index in MEDIAPIPE_INDEX.values():
        required += [f"x_{index}", f"y_{index}"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MalformedLandmarkCsvError(f"landmark CSV is missing columns: {', '.join(missing)}")


def _column(table: pd.DataFrame, name: str, length: int) -> npt.NDArray[np.float64]:
    """A column as a float array, or zeros when the recording did not include it."""
    if name in table.columns:
        return table[name].to_numpy(dtype=float)
    return np.zeros(length, dtype=float)


def read_landmark_csv(path: Path | str, fps: float = DEFAULT_FPS) -> list[FrameLandmarks]:
    """Load every frame in the CSV, ordered by frame index."""
    return list(iter_landmark_csv(path, fps=fps))


def iter_landmark_csv(path: Path | str, fps: float = DEFAULT_FPS) -> Iterator[FrameLandmarks]:
    """Yield one `FrameLandmarks` per row, in frame order.

    Frame indices are taken from the file as-is: gaps are preserved, because a gap means
    frames where no pose was detected and the windower must split the run there.
    """
    table = pd.read_csv(path)
    _require_columns(table)
    table = table.sort_values("frame")

    frame_numbers = table["frame"].to_numpy(dtype=int)
    # Pull each column out once as an array. Per-row DataFrame access is the slow path, and
    # it is exactly the `df.apply(lambda row: ...)` pattern this port set out to remove.
    columns = {
        f"{prefix}_{index}": _column(table, f"{prefix}_{index}", len(table))
        for index in MEDIAPIPE_INDEX.values()
        for prefix in ("x", "y", "z", "vis")
    }

    for position, frame_index in enumerate(frame_numbers):
        joints = {
            joint: Landmark(
                x=float(columns[f"x_{index}"][position]),
                y=float(columns[f"y_{index}"][position]),
                # Some recorded CSVs carry only x/y; treat the rest as unknown rather than
                # refusing to load, since the features never touch them.
                z=float(columns[f"z_{index}"][position]),
                visibility=float(columns[f"vis_{index}"][position]),
            )
            for joint, index in MEDIAPIPE_INDEX.items()
        }
        yield FrameLandmarks(
            index=int(frame_index), timestamp=int(frame_index) / fps, joints=joints
        )


def quality_from_frames(frames: list[FrameLandmarks]) -> QualitySignals:
    """Quality signals for a recorded sequence.

    `frames_read` is inferred from the index span, so a CSV with gaps reports the undetected
    proportion the original extraction saw.
    """
    if not frames:
        return QualitySignals(frames_read=0, frames_detected=0, mean_visibility=0.0)

    span = frames[-1].index - frames[0].index + 1
    mean_visibility = sum(frame.mean_visibility for frame in frames) / len(frames)
    return QualitySignals(
        frames_read=span, frames_detected=len(frames), mean_visibility=mean_visibility
    )
