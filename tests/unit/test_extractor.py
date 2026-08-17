"""Pose extraction, driven by a fake video and a fake pose backend.

Nothing here decodes a file or loads MediaPipe. The extractor takes its frame stream and its
pose model as collaborators precisely so this suite can run in CI, where the reference video
is not committed and MediaPipe may not be installed. The real OpenCV and MediaPipe paths are
covered by the `slow` parity test against the recorded reference.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from aced.pose.extractor import PoseExtraction, UnreadableVideoError, extract_poses
from aced.pose.landmarks import (
    DEFAULT_FPS,
    LOW_VISIBILITY_THRESHOLD,
    MEDIAPIPE_INDEX,
    Joint,
    VideoMeta,
)

TOTAL_MEDIAPIPE_LANDMARKS = 33


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float
    visibility: float


def pose_of(value: float, *, visibility: float = 0.9) -> list[FakeLandmark]:
    """A full 33-landmark pose whose coordinates encode `value`, for identity checks."""
    return [
        FakeLandmark(x=value + i / 1000, y=value, z=0.0, visibility=visibility)
        for i in range(TOTAL_MEDIAPIPE_LANDMARKS)
    ]


class FakeStream:
    """A decoded video: one opaque frame object per entry in `frames`."""

    def __init__(self, frames: list[object], meta: VideoMeta | None = None) -> None:
        self._frames = frames
        self._meta = meta or VideoMeta(fps=30.0, frame_count=len(frames), width=1920, height=1080)
        self.closed = False

    @property
    def meta(self) -> VideoMeta:
        return self._meta

    def __iter__(self) -> Iterator[object]:
        yield from self._frames

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Returns whatever the script says for each successive frame."""

    def __init__(self, script: list[list[FakeLandmark] | None]) -> None:
        self._script = script
        self._calls = 0
        self.closed = False

    def detect(self, frame: object) -> list[FakeLandmark] | None:
        result = self._script[self._calls]
        self._calls += 1
        return result

    def close(self) -> None:
        self.closed = True


def build(script: list[list[FakeLandmark] | None], meta: VideoMeta | None = None) -> PoseExtraction:
    frames: list[object] = [f"frame-{i}" for i in range(len(script))]
    return PoseExtraction(FakeStream(frames, meta), FakeBackend(script))


def test_emits_one_record_per_detected_frame() -> None:
    extraction = build([pose_of(0.1), pose_of(0.2), pose_of(0.3)])

    records = list(extraction)

    assert [record.index for record in records] == [0, 1, 2]
    assert extraction.detected_any


def test_retains_only_the_nine_model_landmarks() -> None:
    extraction = build([pose_of(0.5)])

    (record,) = list(extraction)

    assert set(record.joints) == set(Joint)
    assert len(record.joints) == 9


def test_retained_landmarks_come_from_the_right_mediapipe_indices() -> None:
    """The subset must be a lookup by index, not the first nine landmarks."""
    extraction = build([pose_of(0.0)])

    (record,) = list(extraction)

    for joint, index in MEDIAPIPE_INDEX.items():
        # pose_of encodes the source index in x, so a mis-mapped joint is visible here.
        assert record.joints[joint].x == pytest.approx(index / 1000)


def test_undetected_frames_are_omitted_and_leave_a_gap() -> None:
    """Poses on 0-2 and 5-6, nothing on 3-4: the emitted indices must show the hole."""
    script: list[list[FakeLandmark] | None] = [
        pose_of(0.1),
        pose_of(0.2),
        pose_of(0.3),
        None,
        None,
        pose_of(0.6),
        pose_of(0.7),
    ]
    extraction = build(script)

    indices = [record.index for record in extraction]

    assert indices == [0, 1, 2, 5, 6]
    assert extraction.quality.frames_read == 7
    assert extraction.quality.frames_detected == 5


def test_no_pose_anywhere_completes_without_error() -> None:
    extraction = build([None, None, None])

    records = list(extraction)

    assert records == []
    assert extraction.completed
    assert not extraction.detected_any
    assert extraction.quality.undetected_proportion == pytest.approx(1.0)


def test_timestamps_derive_from_the_frame_rate() -> None:
    meta = VideoMeta(fps=60.0, frame_count=3, width=640, height=480)
    extraction = build([pose_of(0.1), None, pose_of(0.3)], meta)

    records = list(extraction)

    assert [record.timestamp for record in records] == pytest.approx([0.0, 2 / 60])


def test_zero_decodable_frames_is_an_error_not_an_empty_result() -> None:
    """An empty read is a broken file, and must not look like a climber-free video."""
    extraction = build([])

    with pytest.raises(UnreadableVideoError, match="no frames could be decoded"):
        list(extraction)

    assert not extraction.completed


def test_backend_returning_too_few_landmarks_is_an_error() -> None:
    short_pose = pose_of(0.1)[:10]
    extraction = build([short_pose])

    with pytest.raises(UnreadableVideoError, match="landmarks"):
        list(extraction)


def test_stream_and_backend_are_closed_after_iteration() -> None:
    stream = FakeStream(["a", "b"])
    backend = FakeBackend([pose_of(0.1), pose_of(0.2)])
    extraction = PoseExtraction(stream, backend)

    list(extraction)

    assert stream.closed
    assert backend.closed


def test_stream_and_backend_are_closed_when_iteration_is_abandoned() -> None:
    """A caller that stops early must not leak the capture or the pose model."""
    stream = FakeStream(["a", "b", "c"])
    backend = FakeBackend([pose_of(0.1), pose_of(0.2), pose_of(0.3)])
    extraction = PoseExtraction(stream, backend)

    iterator = iter(extraction)
    next(iterator)
    iterator.close()

    assert stream.closed
    assert backend.closed


class TestQualitySignals:
    def test_mean_visibility_averages_detected_frames_only(self) -> None:
        script: list[list[FakeLandmark] | None] = [
            pose_of(0.1, visibility=0.8),
            None,
            pose_of(0.3, visibility=0.6),
        ]
        extraction = build(script)

        list(extraction)

        assert extraction.quality.mean_visibility == pytest.approx(0.7)

    def test_undetected_proportion(self) -> None:
        extraction = build([pose_of(0.1), None, None, pose_of(0.4)])

        list(extraction)

        assert extraction.quality.undetected_proportion == pytest.approx(0.5)

    def test_low_visibility_is_flagged(self) -> None:
        faint = LOW_VISIBILITY_THRESHOLD - 0.1
        extraction = build([pose_of(0.1, visibility=faint)])

        list(extraction)

        assert extraction.quality.is_low_visibility()

    def test_clear_footage_is_not_flagged(self) -> None:
        extraction = build([pose_of(0.1, visibility=0.95)])

        list(extraction)

        assert not extraction.quality.is_low_visibility()

    def test_no_detections_is_not_low_visibility(self) -> None:
        """Nothing was visible to be faint — that is 'no climber', a different signal."""
        extraction = build([None, None])

        list(extraction)

        assert not extraction.quality.is_low_visibility()
        assert not extraction.detected_any


class TestVideoMetadata:
    def test_reported_frame_rate_is_used_as_measured(self) -> None:
        meta = VideoMeta(fps=30.0, frame_count=90, width=1920, height=1080)
        extraction = build([pose_of(0.1)], meta)

        assert extraction.meta.fps == 30.0
        assert not extraction.meta.fps_defaulted
        assert extraction.meta.duration_seconds == pytest.approx(3.0)

    def test_defaulted_frame_rate_is_flagged_as_approximate(self) -> None:
        meta = VideoMeta(fps=DEFAULT_FPS, frame_count=60, width=640, height=480, fps_defaulted=True)
        extraction = build([pose_of(0.1)], meta)

        assert extraction.meta.fps == DEFAULT_FPS
        assert extraction.meta.fps_defaulted

    def test_metadata_is_available_before_iteration(self) -> None:
        """The API reports duration and dimensions while analysis is still running."""
        extraction = build([pose_of(0.1), pose_of(0.2)])

        assert extraction.meta.width == 1920
        assert extraction.meta.height == 1080
        assert not extraction.completed


class TestOpenFailures:
    def test_missing_file_raises_unreadable(self, tmp_path) -> None:
        with pytest.raises(UnreadableVideoError, match="no such video file"):
            extract_poses(tmp_path / "absent.mov")

    def test_non_video_file_raises_unreadable(self, tmp_path) -> None:
        not_a_video = tmp_path / "notes.txt"
        not_a_video.write_text("this is not a video")

        with pytest.raises(UnreadableVideoError):
            extract_poses(not_a_video)

    def test_injected_collaborators_bypass_opencv(self) -> None:
        """`extract_poses` must not touch the filesystem when both are supplied."""
        stream = FakeStream(["a"])
        backend = FakeBackend([pose_of(0.1)])

        extraction = extract_poses("/nonexistent/path.mov", stream=stream, backend=backend)

        assert [record.index for record in extraction] == [0]
