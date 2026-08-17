"""Landmark and video types shared by every stage downstream of pose extraction.

The models use nine of MediaPipe's thirty-three landmarks. They are addressed by anatomical
name everywhere in this package; the numeric MediaPipe indices appear only here and in the
extractor, so a change of pose backend does not ripple outward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Joint(StrEnum):
    """The nine landmarks the trained models consume."""

    NOSE = "nose"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"


#: MediaPipe Pose landmark index for each retained joint. The research code keeps exactly
#: {0, 11, 12, 13, 14, 15, 16, 23, 24} and drops the rest.
MEDIAPIPE_INDEX: dict[Joint, int] = {
    Joint.NOSE: 0,
    Joint.LEFT_SHOULDER: 11,
    Joint.RIGHT_SHOULDER: 12,
    Joint.LEFT_ELBOW: 13,
    Joint.RIGHT_ELBOW: 14,
    Joint.LEFT_WRIST: 15,
    Joint.RIGHT_WRIST: 16,
    Joint.LEFT_HIP: 23,
    Joint.RIGHT_HIP: 24,
}

#: The default frame rate assumed when a container reports none. Matches the research code.
DEFAULT_FPS = 30.0


@dataclass(frozen=True, slots=True)
class Landmark:
    """One joint in one frame.

    Coordinates are MediaPipe's normalized image space: `x` and `y` in roughly [0, 1] with
    `y` increasing downward. Features use `x` and `y` only — the research method is 2D, and
    `z` is carried for rendering and quality checks rather than for inference.
    """

    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True, slots=True)
class FrameLandmarks:
    """The retained landmarks for a single frame in which a pose was detected.

    `index` is the position of the frame in the source video, not a position in the sequence
    of detected frames. Frames without a detection are never emitted, so consecutive records
    can differ by more than one — that gap is meaningful and the windower splits runs on it.
    """

    index: int
    timestamp: float
    joints: dict[Joint, Landmark]

    def __post_init__(self) -> None:
        missing = set(Joint) - set(self.joints)
        if missing:
            names = ", ".join(sorted(joint.value for joint in missing))
            raise ValueError(f"frame {self.index} is missing landmarks: {names}")

    def point(self, joint: Joint) -> tuple[float, float]:
        """The joint's 2D position, which is all the features use."""
        landmark = self.joints[joint]
        return landmark.x, landmark.y

    @property
    def mean_visibility(self) -> float:
        return sum(landmark.visibility for landmark in self.joints.values()) / len(self.joints)


@dataclass(frozen=True, slots=True)
class VideoMeta:
    """What the container says about the source video.

    `fps_defaulted` records that the frame rate was missing or non-positive and
    `DEFAULT_FPS` was substituted, which makes every derived timestamp approximate. Reports
    surface that rather than presenting invented times as measured ones.
    """

    fps: float
    frame_count: int
    width: int
    height: int
    fps_defaulted: bool = False

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def timestamp(self, frame_index: int) -> float:
        return frame_index / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True, slots=True)
class QualitySignals:
    """Conditions that degrade confidence in an analysis.

    Multi-person detection is deliberately absent: `mp.solutions.pose` returns a single pose
    and exposes no signal about anyone else in frame. The limitation is disclosed in the
    report instead of being flagged per video.
    """

    frames_read: int
    frames_detected: int
    mean_visibility: float

    @property
    def undetected_proportion(self) -> float:
        if self.frames_read == 0:
            return 0.0
        return 1.0 - (self.frames_detected / self.frames_read)
