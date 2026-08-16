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

import pandas as pd

from aced.pose.landmarks import (
    DEFAULT_FPS,
    MEDIAPIPE_INDEX,
    FrameLandmarks,
    Joint,
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

    for row in table.itertuples(index=False):
        record = row._asdict()
        joints: dict[Joint, Landmark] = {}
        for joint, index in MEDIAPIPE_INDEX.items():
            joints[joint] = Landmark(
                x=float(record[f"x_{index}"]),
                y=float(record[f"y_{index}"]),
                # Some recorded CSVs carry only x/y; treat the rest as unknown rather than
                # refusing to load, since features never touch them.
                z=float(record.get(f"z_{index}", 0.0)),
                visibility=float(record.get(f"vis_{index}", 0.0)),
            )
        frame_index = int(record["frame"])
        yield FrameLandmarks(index=frame_index, timestamp=frame_index / fps, joints=joints)


def quality_from_frames(frames: list[FrameLandmarks]) -> QualitySignals:
    """Quality signals for a recorded sequence.

    `frames_read` is inferred from the index span, so a CSV with gaps reports the undetected
    proportion the original extraction saw.
    """
    if not frames:
        return QualitySignals(frames_read=0, frames_detected=0, mean_visibility=0.0)

    span = frames[-1].index - frames[0].index + 1
    mean_visibility = sum(frame.mean_visibility for frame in frames) / len(frames)
    return QualitySignals(frames_read=span, frames_detected=len(frames), mean_visibility=mean_visibility)
