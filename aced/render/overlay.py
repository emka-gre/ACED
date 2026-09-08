"""Draw the classified skeleton over the original footage.

This is an artifact, not a stage. Nothing in `aced.pose`, `aced.features`, `aced.inference`
or `aced.report` imports this module: an analysis produces a report and stops, and the
overlay is rendered later, only if someone asks for it. Rendering costs a full second decode
and re-encode of the video, which is the most expensive thing in the system and is worthless
for the majority of analyses nobody watches back.

The skeleton is drawn from the nine landmarks the models consume, not MediaPipe's thirty-
three. Drawing the full pose would show a detection this application never used — the
overlay is meant to show what the classifier saw, and it saw nine joints.

Colour carries the frame's assigned class, so the video answers "where did it think I was
dropping into straight arms" by playback alone. Frames the classifier never covered — the
tail too short to fill a window, or any frame where no pose was detected — are drawn
neutral rather than being coloured sound, because "not judged" and "judged fine" are
different statements and the overlay should not conflate them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from aced.inference.pipeline import PredictedClass
from aced.pose.extractor import (
    FrameStream,
    OpenCvFrameStream,
    UnreadableVideoError,
)
from aced.pose.landmarks import FrameLandmarks, Joint

if TYPE_CHECKING:
    from aced.pose.extractor import Frame


class OverlayRenderError(OSError):
    """The overlay could not be written.

    Distinct from `UnreadableVideoError`, which means the source could not be read. This one
    means the source was fine and the destination was not — an unavailable codec, an
    unwritable path — and the caller's remedy is different in each case.
    """


#: Bones between the retained joints. Every pair here is a real MediaPipe pose connection;
#: the head is the exception and is handled separately, because MediaPipe links the nose to
#: the eyes and the eyes are not retained, so a nose drawn from these edges alone would float
#: unattached above the body.
SKELETON_EDGES: tuple[tuple[Joint, Joint], ...] = (
    (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.LEFT_SHOULDER, Joint.LEFT_HIP),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.RIGHT_HIP),
)


@dataclass(frozen=True, slots=True)
class OverlayStyle:
    """How the skeleton is drawn. Colours are BGR, because OpenCV is.

    Sound and fault are separated on blue-versus-orange rather than the conventional green-
    versus-red: red and green are the pair a red-green colour blind viewer cannot tell apart,
    and roughly one in twelve men has that. Line weight carries the same distinction a second
    time, so the class survives a greyscale screenshot or a colour blind viewer entirely.
    """

    sound: tuple[int, int, int] = (200, 150, 40)
    fault: tuple[int, int, int] = (40, 110, 240)
    unclassified: tuple[int, int, int] = (150, 150, 150)
    thickness: int = 2
    fault_thickness: int = 4
    joint_radius: int = 4


DEFAULT_STYLE = OverlayStyle()


class VideoSink(Protocol):
    """Somewhere encoded frames go. Injectable for the same reason `FrameStream` is."""

    def write(self, frame: Frame) -> None: ...

    def close(self) -> None: ...


class OpenCvVideoSink:
    """An encoded video file on disk.

    Dimensions come from the frames themselves rather than from `VideoMeta`, because a
    container that lies about its frame rate tends to lie about its dimensions too, and a
    `VideoWriter` opened at the wrong size silently writes nothing usable.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        fps: float,
        width: int,
        height: int,
        fourcc: str = "mp4v",
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._writer = cv2.VideoWriter(
            str(self._path),
            # `VideoWriter.fourcc`, not the module-level `VideoWriter_fourcc` alias: the
            # alias is untyped in the OpenCV stubs and the static method is the documented
            # spelling in OpenCV 4.
            cv2.VideoWriter.fourcc(*fourcc),
            fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise OverlayRenderError(
                f"could not open {self._path.name} for writing with codec {fourcc!r} — the "
                "codec may not be available in this OpenCV build"
            )

    def write(self, frame: Frame) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """What the render produced, for the caller that has to report on it.

    The counts are a cheap integrity check: `frames_written` should match the source's frame
    count, and a `frames_with_skeleton` far below it means the overlay is mostly bare footage
    because pose detection failed, not because the render did.
    """

    path: Path | None
    frames_written: int
    frames_with_skeleton: int
    sound_frames: int
    fault_frames: int
    unclassified_frames: int


def to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Normalized MediaPipe coordinates to pixel coordinates in a frame of this size.

    MediaPipe's normalized space is not clamped to [0, 1] — a joint tracked just outside the
    crop reports slightly beyond it. The values are passed through unclamped and OpenCV
    clips the drawing, which keeps a limb leaving the frame pointing the right way instead of
    snapping it to the edge.
    """
    return round(x * width), round(y * height)


def draw_skeleton(
    frame: Frame,
    record: FrameLandmarks,
    colour: tuple[int, int, int],
    *,
    style: OverlayStyle = DEFAULT_STYLE,
    thickness: int | None = None,
) -> None:
    """Draw one frame's skeleton onto `frame`, in place."""
    import cv2

    height, width = frame.shape[:2]
    line_width = style.thickness if thickness is None else thickness
    points = {joint: to_pixel(*record.point(joint), width, height) for joint in Joint}

    for start, end in SKELETON_EDGES:
        cv2.line(frame, points[start], points[end], colour, line_width, lineType=cv2.LINE_AA)

    # The neck: nose to the midpoint of the shoulders. Not a MediaPipe connection — a drawing
    # convention, so the head reads as attached to the body it belongs to.
    left, right = points[Joint.LEFT_SHOULDER], points[Joint.RIGHT_SHOULDER]
    neck = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
    cv2.line(frame, points[Joint.NOSE], neck, colour, line_width, lineType=cv2.LINE_AA)

    for point in points.values():
        cv2.circle(frame, point, style.joint_radius, colour, -1, lineType=cv2.LINE_AA)


def render_overlay(
    source: Path | str,
    landmarks: Iterable[FrameLandmarks],
    classes: Mapping[int, PredictedClass],
    destination: Path | str,
    *,
    style: OverlayStyle = DEFAULT_STYLE,
    stream: FrameStream | None = None,
    sink: VideoSink | None = None,
) -> OverlayResult:
    """Re-encode `source` with the classified skeleton drawn over it.

    `landmarks` are the records pose extraction emitted and `classes` the frame-level
    assignment from `frame_classes` — both keyed on the source's real frame indices, which is
    what lets undetected frames pass through untouched rather than shifting everything after
    them out of sync.

    `stream` and `sink` are injection points for tests and for a future encoder, matching
    `extract_poses`. The default path constructs OpenCV on both ends and imports it nowhere
    else.
    """
    by_frame = {record.index: record for record in landmarks}
    frames = OpenCvFrameStream(source) if stream is None else stream
    destination = Path(destination)

    written = with_skeleton = sound = fault = unclassified = 0
    output: VideoSink | None = sink

    try:
        for index, frame in enumerate(frames):
            # The sink is opened from the first decoded frame, so its dimensions are the ones
            # actually being written rather than the ones the container claims.
            if output is None:
                height, width = frame.shape[:2]
                output = OpenCvVideoSink(
                    destination, fps=frames.meta.fps, width=width, height=height
                )

            record = by_frame.get(index)
            canvas = frame.copy()
            if record is not None:
                predicted = classes.get(index)
                if predicted is PredictedClass.FAULT:
                    fault += 1
                    colour, thickness = style.fault, style.fault_thickness
                elif predicted is PredictedClass.SOUND:
                    sound += 1
                    colour, thickness = style.sound, style.thickness
                else:
                    unclassified += 1
                    colour, thickness = style.unclassified, style.thickness
                draw_skeleton(canvas, record, colour, style=style, thickness=thickness)
                with_skeleton += 1

            output.write(canvas)
            written += 1

        if written == 0:
            raise UnreadableVideoError(
                f"no frames could be decoded from {Path(source).name} — nothing to render"
            )
    finally:
        frames.close()
        if output is not None:
            output.close()

    return OverlayResult(
        path=destination if sink is None else None,
        frames_written=written,
        frames_with_skeleton=with_skeleton,
        sound_frames=sound,
        fault_frames=fault,
        unclassified_frames=unclassified,
    )
