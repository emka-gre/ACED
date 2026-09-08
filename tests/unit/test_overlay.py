"""Skeleton-overlay rendering, driven by fake frames and a fake sink.

Nothing here decodes or encodes a file. The frames are small numpy arrays and the sink keeps
what it is handed, so the drawing itself is asserted on real pixels while no codec has to be
available. The real OpenCV encode path is exercised by the archive-gated parity test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from aced.inference.pipeline import PredictedClass
from aced.pose.extractor import UnreadableVideoError
from aced.pose.landmarks import FrameLandmarks, Joint, Landmark, VideoMeta
from aced.render.overlay import (
    DEFAULT_STYLE,
    SKELETON_EDGES,
    OverlayStyle,
    draw_skeleton,
    render_overlay,
    to_pixel,
)

FRAME_SIZE = 120

#: A plausible upright climber, so the drawn bones have real length and the colour counts
#: below are not measuring a single overlapping blob.
POSITIONS: dict[Joint, tuple[float, float]] = {
    Joint.NOSE: (0.50, 0.15),
    Joint.LEFT_SHOULDER: (0.40, 0.30),
    Joint.RIGHT_SHOULDER: (0.60, 0.30),
    Joint.LEFT_ELBOW: (0.33, 0.45),
    Joint.RIGHT_ELBOW: (0.67, 0.45),
    Joint.LEFT_WRIST: (0.28, 0.60),
    Joint.RIGHT_WRIST: (0.72, 0.60),
    Joint.LEFT_HIP: (0.44, 0.68),
    Joint.RIGHT_HIP: (0.56, 0.68),
}


def landmarks_for(index: int, *, visibility: float = 0.9) -> FrameLandmarks:
    joints = {
        joint: Landmark(x=x, y=y, z=0.0, visibility=visibility)
        for joint, (x, y) in POSITIONS.items()
    }
    return FrameLandmarks(index=index, timestamp=index / 30.0, joints=joints)


def blank_frame() -> np.ndarray:
    return np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)


class FakeStream:
    """A decoded video of blank frames."""

    def __init__(self, count: int, meta: VideoMeta | None = None) -> None:
        self.frames = [blank_frame() for _ in range(count)]
        self._meta = meta or VideoMeta(
            fps=30.0, frame_count=count, width=FRAME_SIZE, height=FRAME_SIZE
        )
        self.closed = False

    @property
    def meta(self) -> VideoMeta:
        return self._meta

    def __iter__(self) -> Iterator[np.ndarray]:
        yield from self.frames

    def close(self) -> None:
        self.closed = True


class FakeSink:
    """Keeps every frame it is handed, so the render can be inspected pixel by pixel."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


def count_colour(frame: np.ndarray, colour: tuple[int, int, int]) -> int:
    return int((frame == np.array(colour, dtype=np.uint8)).all(axis=2).sum())


class TestPixelMapping:
    def test_normalized_coordinates_scale_to_the_frame(self) -> None:
        assert to_pixel(0.5, 0.25, 200, 100) == (100, 25)

    def test_coordinates_outside_the_frame_are_not_clamped(self) -> None:
        """MediaPipe tracks joints just outside the crop; clamping would bend the limb."""
        assert to_pixel(-0.1, 1.2, 100, 100) == (-10, 120)


class TestDrawing:
    def test_the_skeleton_is_drawn_in_the_given_colour(self) -> None:
        frame = blank_frame()
        draw_skeleton(frame, landmarks_for(0), DEFAULT_STYLE.fault)

        assert count_colour(frame, DEFAULT_STYLE.fault) > 0
        assert count_colour(frame, DEFAULT_STYLE.sound) == 0

    def test_every_retained_joint_is_marked(self) -> None:
        frame = blank_frame()
        draw_skeleton(frame, landmarks_for(0), DEFAULT_STYLE.sound)

        for joint, (x, y) in POSITIONS.items():
            column, row = to_pixel(x, y, FRAME_SIZE, FRAME_SIZE)
            assert tuple(frame[row, column]) == DEFAULT_STYLE.sound, f"{joint} not drawn"

    def test_the_head_is_joined_to_the_body(self) -> None:
        """The nose is in no MediaPipe connection between retained joints; without the neck
        convention it would float unattached."""
        frame = blank_frame()
        draw_skeleton(frame, landmarks_for(0), DEFAULT_STYLE.sound)

        nose_column, nose_row = to_pixel(*POSITIONS[Joint.NOSE], FRAME_SIZE, FRAME_SIZE)
        shoulder_row = to_pixel(*POSITIONS[Joint.LEFT_SHOULDER], FRAME_SIZE, FRAME_SIZE)[1]
        between = frame[(nose_row + shoulder_row) // 2, nose_column]
        assert tuple(between) != (0, 0, 0)

    def test_only_the_retained_joints_are_connected(self) -> None:
        """Nine landmarks are what the classifier saw; the overlay claims no more than that."""
        assert set(SKELETON_EDGES) <= {(a, b) for a in Joint for b in Joint if a is not b}
        assert Joint.NOSE not in {joint for edge in SKELETON_EDGES for joint in edge}

    def test_thickness_distinguishes_the_classes_without_colour(self) -> None:
        """The fault/sound distinction has to survive greyscale and colour blindness."""
        sound, fault = blank_frame(), blank_frame()
        draw_skeleton(sound, landmarks_for(0), DEFAULT_STYLE.sound)
        draw_skeleton(
            fault, landmarks_for(0), DEFAULT_STYLE.fault, thickness=DEFAULT_STYLE.fault_thickness
        )

        drawn_sound = int((sound.any(axis=2)).sum())
        drawn_fault = int((fault.any(axis=2)).sum())
        assert drawn_fault > drawn_sound


class TestRendering:
    def test_every_decoded_frame_is_written(self) -> None:
        stream, sink = FakeStream(5), FakeSink()
        result = render_overlay(
            "source.mov",
            [landmarks_for(i) for i in range(5)],
            {},
            "out.mp4",
            stream=stream,
            sink=sink,
        )

        assert result.frames_written == 5
        assert len(sink.frames) == 5

    def test_undetected_frames_pass_through_untouched(self) -> None:
        """A gap must not shift the frames after it; the overlay stays in sync with the
        source, and a frame with no pose simply carries no skeleton."""
        stream, sink = FakeStream(4), FakeSink()
        detected = [landmarks_for(0), landmarks_for(3)]
        result = render_overlay("source.mov", detected, {}, "out.mp4", stream=stream, sink=sink)

        assert result.frames_written == 4
        assert result.frames_with_skeleton == 2
        assert sink.frames[1].any() == np.False_
        assert sink.frames[2].any() == np.False_
        assert sink.frames[0].any() and sink.frames[3].any()

    def test_frames_are_coloured_by_their_assigned_class(self) -> None:
        stream, sink = FakeStream(2), FakeSink()
        classes = {0: PredictedClass.SOUND, 1: PredictedClass.FAULT}
        render_overlay(
            "source.mov",
            [landmarks_for(0), landmarks_for(1)],
            classes,
            "out.mp4",
            stream=stream,
            sink=sink,
        )

        assert count_colour(sink.frames[0], DEFAULT_STYLE.sound) > 0
        assert count_colour(sink.frames[0], DEFAULT_STYLE.fault) == 0
        assert count_colour(sink.frames[1], DEFAULT_STYLE.fault) > 0
        assert count_colour(sink.frames[1], DEFAULT_STYLE.sound) == 0

    def test_unclassified_frames_are_not_coloured_sound(self) -> None:
        """A frame no window covered was never judged. Drawing it sound would assert a
        verdict the classifier never reached."""
        stream, sink = FakeStream(2), FakeSink()
        classes = {0: PredictedClass.SOUND}
        result = render_overlay(
            "source.mov",
            [landmarks_for(0), landmarks_for(1)],
            classes,
            "out.mp4",
            stream=stream,
            sink=sink,
        )

        assert count_colour(sink.frames[1], DEFAULT_STYLE.unclassified) > 0
        assert count_colour(sink.frames[1], DEFAULT_STYLE.sound) == 0
        assert result.unclassified_frames == 1

    def test_the_source_frames_are_not_modified(self) -> None:
        stream, sink = FakeStream(1), FakeSink()
        render_overlay(
            "source.mov",
            [landmarks_for(0)],
            {0: PredictedClass.FAULT},
            "out.mp4",
            stream=stream,
            sink=sink,
        )

        assert stream.frames[0].any() == np.False_

    def test_the_result_counts_each_class(self) -> None:
        stream, sink = FakeStream(4), FakeSink()
        classes = {
            0: PredictedClass.SOUND,
            1: PredictedClass.FAULT,
            2: PredictedClass.FAULT,
        }
        detected = [landmarks_for(i) for i in range(3)]  # frame 3 has no pose
        result = render_overlay(
            "source.mov", detected, classes, "out.mp4", stream=stream, sink=sink
        )

        assert (result.sound_frames, result.fault_frames) == (1, 2)
        assert result.unclassified_frames == 0
        assert (result.frames_written, result.frames_with_skeleton) == (4, 3)

    def test_a_custom_style_is_honoured(self) -> None:
        style = OverlayStyle(sound=(1, 2, 3), fault=(4, 5, 6), unclassified=(7, 8, 9))
        stream, sink = FakeStream(1), FakeSink()
        render_overlay(
            "source.mov",
            [landmarks_for(0)],
            {0: PredictedClass.FAULT},
            "out.mp4",
            stream=stream,
            sink=sink,
            style=style,
        )

        assert count_colour(sink.frames[0], (4, 5, 6)) > 0

    def test_a_video_with_no_decodable_frames_is_rejected(self) -> None:
        stream, sink = FakeStream(0), FakeSink()
        with pytest.raises(UnreadableVideoError):
            render_overlay("source.mov", [], {}, "out.mp4", stream=stream, sink=sink)

    def test_both_ends_are_closed_even_when_the_render_fails(self) -> None:
        stream, sink = FakeStream(0), FakeSink()
        with pytest.raises(UnreadableVideoError):
            render_overlay("source.mov", [], {}, "out.mp4", stream=stream, sink=sink)

        assert stream.closed and sink.closed

    def test_no_file_is_written_when_the_sink_is_supplied(self, tmp_path: Path) -> None:
        stream, sink = FakeStream(1), FakeSink()
        destination = tmp_path / "out.mp4"
        result = render_overlay(
            "source.mov", [landmarks_for(0)], {}, destination, stream=stream, sink=sink
        )

        assert result.path is None
        assert not destination.exists()


class TestEncoding:
    """The one place a real encoder runs.

    The source is still a fake stream — decoding is the extractor's problem and is covered
    there. What is being checked is that the default sink actually opens, sizes itself from
    the frames, and leaves a file behind, which no amount of drawing assertions would catch.
    """

    def test_a_playable_file_is_written_by_the_default_sink(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        stream = FakeStream(10)
        destination = tmp_path / "renders" / "overlay.mp4"
        classes = dict.fromkeys(range(10), PredictedClass.FAULT)

        result = render_overlay(
            "source.mov", [landmarks_for(i) for i in range(10)], classes, destination, stream=stream
        )

        assert result.path == destination
        assert destination.exists() and destination.stat().st_size > 0

        # Read it back: a `VideoWriter` that silently wrote nothing usable still leaves a
        # non-empty file, so the frame size is what proves the render landed.
        capture = cv2.VideoCapture(str(destination))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == FRAME_SIZE
            assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == FRAME_SIZE
        finally:
            capture.release()

    def test_the_destination_directory_is_created(self, tmp_path: Path) -> None:
        stream = FakeStream(1)
        destination = tmp_path / "does" / "not" / "exist" / "overlay.mp4"

        render_overlay("source.mov", [landmarks_for(0)], {}, destination, stream=stream)

        assert destination.exists()


class TestRenderingIsOnDemand:
    """The spec's second scenario: an analysis that nobody asked an overlay for renders none.

    Asserted structurally rather than behaviourally — the guarantee is that no analysis stage
    can reach the renderer at all, which an import check states more durably than any single
    pipeline run could.
    """

    def test_no_analysis_stage_imports_the_renderer(self) -> None:
        package = Path(__file__).resolve().parents[2] / "aced"
        stages = ("pose", "features", "inference", "report")

        for stage in stages:
            for module in (package / stage).rglob("*.py"):
                assert "aced.render" not in module.read_text(), (
                    f"{module.relative_to(package.parent)} imports the renderer; overlay "
                    "rendering must stay on-demand, never part of an analysis"
                )
