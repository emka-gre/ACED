"""Quality flags: conditions that degrade confidence, stated plainly and attached to the report.

Two of the three signals come straight from pose extraction (`QualitySignals`); the third —
a video too short to fill one analysis window — additionally needs the model's window
length, which pose extraction has no notion of.
"""

from __future__ import annotations

from aced.pose.landmarks import LOW_VISIBILITY_THRESHOLD, QualitySignals
from aced.report.types import QualityFlag, QualityFlagKind

#: Above this proportion of frames with no detected pose, results are flagged rather than
#: presented as if the whole climb were seen. Chosen generously: MediaPipe drops individual
#: frames routinely (motion blur, a hand crossing the face) without the analysis being
#: compromised, so the flag is reserved for genuinely incomplete coverage.
UNDETECTED_PROPORTION_THRESHOLD = 0.2


def is_too_short(quality: QualitySignals, window_length: int) -> bool:
    """Whether the video has fewer detected frames in total than one analysis window.

    This is deliberately about the video as a whole, not about whether any *individual*
    window came out padded: a long video whose pose detection is merely fragmented into
    many short runs would pad every window it produces without the video itself being too
    short — that is a coverage problem (`INCOMPLETE_POSE_DETECTION`), not this one.
    """
    return quality.frames_detected < window_length


def build_quality_flags(quality: QualitySignals, window_length: int) -> list[QualityFlag]:
    flags: list[QualityFlag] = []

    if quality.undetected_proportion > UNDETECTED_PROPORTION_THRESHOLD:
        flags.append(
            QualityFlag(
                kind=QualityFlagKind.INCOMPLETE_POSE_DETECTION,
                message=(
                    "The climber's pose could not be detected in a significant portion of "
                    "the video, so some parts of the analysis may be less reliable."
                ),
            )
        )

    if quality.is_low_visibility(LOW_VISIBILITY_THRESHOLD):
        flags.append(
            QualityFlag(
                kind=QualityFlagKind.LOW_VISIBILITY,
                message=(
                    "The climber's joints were often faint or partially hidden, which makes "
                    "the detected positions less reliable."
                ),
            )
        )

    if is_too_short(quality, window_length):
        flags.append(
            QualityFlag(
                kind=QualityFlagKind.VIDEO_TOO_SHORT,
                message=(
                    "The video was too short to fill one full analysis window, so results "
                    "are based on a padded, partial window and are less reliable."
                ),
            )
        )

    return flags
