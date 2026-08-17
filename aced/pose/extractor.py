"""Extract per-frame landmarks from a video with MediaPipe Pose.

This is the one stage that touches the pose backend. Everything downstream consumes
`FrameLandmarks`, so the video reader and the pose model are both injectable: the unit suite
drives the extractor with fakes and never imports MediaPipe or decodes a file, which is what
lets pose extraction be tested in CI where neither is available.

Landmarks are emitted as each frame is processed rather than collected into a DataFrame, per
the streaming design — a 30-second climb is a few hundred frames, but the same generator is
what a real-time path would consume later.

The research extractor (`streamlit/pose_extraction.py` in the thesis archive) returned an
empty DataFrame when the file could not be opened, making an unreadable video look exactly
like a video with no climber in it. `UnreadableVideoError` exists to keep those apart.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aced.pose.landmarks import (
    DEFAULT_FPS,
    MEDIAPIPE_INDEX,
    FrameLandmarks,
    Joint,
    Landmark,
    QualitySignals,
    VideoMeta,
)

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt

    #: One decoded frame. Left unparameterised by dtype: OpenCV's `read` is typed as
    #: returning `Mat | ndarray[integer | floating]`, and the extractor only ever passes the
    #: array through to the backend, so pinning it to uint8 would buy a cast and no safety.
    Frame = npt.NDArray[Any]
else:  # `Frame` is only ever an annotation; numpy stays an implementation detail at runtime.
    Frame = object


class UnreadableVideoError(OSError):
    """The file could not be opened or decoded as video.

    Distinct from "no pose was detected": that is a valid analysis of a valid file, and the
    report says so. This means there was nothing to analyse in the first place.
    """


@runtime_checkable
class RawLandmark(Protocol):
    """One landmark as the pose backend reports it, before the subset is applied."""

    x: float
    y: float
    z: float
    visibility: float


class PoseBackend(Protocol):
    """A pose model that turns one frame into landmarks, or into nothing."""

    def detect(self, frame: Frame) -> Sequence[RawLandmark] | None:
        """All backend landmarks for the frame, or `None` when no pose was found."""
        ...

    def close(self) -> None: ...


class FrameStream(Protocol):
    """A decoded video: its container metadata, then its frames in order."""

    @property
    def meta(self) -> VideoMeta: ...

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class OpenCvFrameStream:
    """Frames from a file on disk, read through OpenCV.

    Opening is eager so that an unreadable file fails at construction, before a caller has
    started consuming a generator and has to decide what a mid-iteration failure means.
    """

    def __init__(self, path: Path | str) -> None:
        import cv2

        self._cv2 = cv2
        self._path = Path(path)
        if not self._path.exists():
            raise UnreadableVideoError(f"no such video file: {self._path}")

        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            self._capture.release()
            raise UnreadableVideoError(
                f"could not open {self._path.name} as video — the file may be corrupt or in an "
                "unsupported codec"
            )
        self._meta = self._read_meta()

    def _read_meta(self) -> VideoMeta:
        """Container metadata, with a documented substitute for a missing frame rate.

        A non-positive or NaN fps is common in phone recordings remuxed by a share sheet.
        Substituting the default keeps timestamps derivable, and `fps_defaulted` is what the
        report uses to present them as approximate rather than measured.
        """
        cv2 = self._cv2
        raw_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        fps_defaulted = not (raw_fps > 0.0) or raw_fps != raw_fps  # non-positive or NaN
        fps = DEFAULT_FPS if fps_defaulted else raw_fps

        # Frame counts are advisory: some containers report 0 or a negative value. The count
        # is metadata only — iteration stops when decoding stops, never at this number.
        frame_count = max(int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
        return VideoMeta(
            fps=fps,
            frame_count=frame_count,
            width=max(int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 0),
            height=max(int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 0),
            fps_defaulted=fps_defaulted,
        )

    @property
    def meta(self) -> VideoMeta:
        return self._meta

    def __iter__(self) -> Iterator[Frame]:
        while True:
            success, frame = self._capture.read()
            if not success:
                return
            yield frame

    def close(self) -> None:
        self._capture.release()


class MediaPipePoseBackend:
    """MediaPipe Pose, configured as the research extractor configured it.

    `static_image_mode=False` matches `streamlit/pose_extraction.py`; it makes MediaPipe
    track across frames instead of re-detecting on each one, which is both faster and what
    the parity reference was produced under. Every other parameter is left at its default
    for the same reason — the recorded reference is only meaningful against these settings.
    """

    def __init__(self, *, static_image_mode: bool = False) -> None:
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._pose = mp.solutions.pose.Pose(static_image_mode=static_image_mode)

    def detect(self, frame: Frame) -> Sequence[RawLandmark] | None:
        # MediaPipe expects RGB; OpenCV decodes BGR. Getting this backwards does not raise,
        # it just quietly degrades detection, so the conversion lives next to the call.
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        landmarks = getattr(results, "pose_landmarks", None)
        if landmarks is None:
            return None
        detected: Sequence[RawLandmark] = landmarks.landmark
        return detected

    def close(self) -> None:
        self._pose.close()


class PoseExtraction:
    """One extraction run: metadata up front, landmarks as they are produced.

    `meta` is available immediately, because the API needs to report duration and dimensions
    before analysis finishes. `quality` is only meaningful once iteration has completed —
    the undetected proportion is not known until the last frame has been read.
    """

    def __init__(self, stream: FrameStream, backend: PoseBackend) -> None:
        self._stream = stream
        self._backend = backend
        self._frames_read = 0
        self._frames_detected = 0
        self._visibility_total = 0.0
        self._completed = False

    @property
    def meta(self) -> VideoMeta:
        return self._stream.meta

    @property
    def completed(self) -> bool:
        """Whether iteration ran to the end of the video."""
        return self._completed

    @property
    def detected_any(self) -> bool:
        """False means a readable video in which no climber was ever found."""
        return self._frames_detected > 0

    @property
    def quality(self) -> QualitySignals:
        """Signals for the frames seen so far; final once `completed` is true."""
        mean_visibility = (
            self._visibility_total / self._frames_detected if self._frames_detected else 0.0
        )
        return QualitySignals(
            frames_read=self._frames_read,
            frames_detected=self._frames_detected,
            mean_visibility=mean_visibility,
        )

    def __iter__(self) -> Iterator[FrameLandmarks]:
        """Yield one record per frame in which a pose was detected.

        The frame counter advances on every decoded frame, detected or not, so an emitted
        record's `index` is its true position in the video. Downstream, a jump in those
        indices is what tells the windower a run of frames was lost and the window must be
        split there rather than bridged across the gap.
        """
        try:
            for frame in self._stream:
                index = self._frames_read
                self._frames_read += 1

                detected = self._backend.detect(frame)
                if detected is None:
                    continue

                record = self._to_frame_landmarks(index, detected)
                self._frames_detected += 1
                self._visibility_total += record.mean_visibility
                yield record

            if self._frames_read == 0:
                raise UnreadableVideoError(
                    "the video opened but no frames could be decoded — the file is likely "
                    "truncated or in an unsupported codec"
                )
            self._completed = True
        finally:
            self.close()

    def _to_frame_landmarks(self, index: int, detected: Sequence[RawLandmark]) -> FrameLandmarks:
        """Keep the nine model landmarks and drop MediaPipe's other twenty-four."""
        required = max(MEDIAPIPE_INDEX.values())
        if len(detected) <= required:
            raise UnreadableVideoError(
                f"frame {index}: pose backend returned {len(detected)} landmarks, "
                f"but index {required} is required"
            )

        joints: dict[Joint, Landmark] = {}
        for joint, landmark_index in MEDIAPIPE_INDEX.items():
            raw = detected[landmark_index]
            joints[joint] = Landmark(
                x=float(raw.x),
                y=float(raw.y),
                z=float(raw.z),
                visibility=float(raw.visibility),
            )
        return FrameLandmarks(
            index=index,
            timestamp=self.meta.timestamp(index),
            joints=joints,
        )

    def close(self) -> None:
        self._stream.close()
        self._backend.close()


def extract_poses(
    path: Path | str,
    *,
    stream: FrameStream | None = None,
    backend: PoseBackend | None = None,
) -> PoseExtraction:
    """Open a video and prepare it for extraction.

    Raises `UnreadableVideoError` here, at open time, rather than on first iteration — the
    caller learns the file is unusable before it has committed to consuming a generator.

    `stream` and `backend` exist for tests and for swapping the pose model later; the default
    path constructs OpenCV and MediaPipe, and imports neither until it does.
    """
    frames = OpenCvFrameStream(path) if stream is None else stream
    try:
        pose = MediaPipePoseBackend() if backend is None else backend
    except BaseException:
        frames.close()
        raise
    return PoseExtraction(frames, pose)
