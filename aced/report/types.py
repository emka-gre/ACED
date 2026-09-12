"""The report's public shape.

Everything here is a frozen dataclass of JSON-native types (`str`, `float`, `int`, `bool`,
tuples, and nested dataclasses of the same) so `to_dict()` never has to reach for anything
beyond built-in `dataclasses.asdict`-style recursion, and a report round-trips through JSON
without loss.

Nothing here carries a scaled feature vector, a model weight, or a backend-specific field —
`Fault.attribution` records engineered feature values used for the explanation (a name, its
value, and the sound-window baseline it is compared against), which is not the same thing as
exposing the model's input. See `language.py` for the separate guarantee that *rendered
explanation text* additionally names no feature identifiers or statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QualityFlagKind(StrEnum):
    INCOMPLETE_POSE_DETECTION = "incomplete_pose_detection"
    LOW_VISIBILITY = "low_visibility"
    VIDEO_TOO_SHORT = "video_too_short"


@dataclass(frozen=True, slots=True)
class QualityFlag:
    """A condition that degrades confidence in the analysis, in plain language."""

    kind: QualityFlagKind
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "message": self.message}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QualityFlag:
        return cls(kind=QualityFlagKind(payload["kind"]), message=str(payload["message"]))


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's part in a fault's attribution.

    `deviation` is the standardized difference from the sound-window baseline — `(value -
    baseline) / baseline_spread`, zero when the baseline has no spread to divide by rather
    than an infinite or undefined result. This is engineering detail, not the copy shown to
    a climber; see `language.py` for that.
    """

    feature: str
    value: float
    baseline: float
    deviation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": self.value,
            "baseline": self.baseline,
            "deviation": self.deviation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureContribution:
        return cls(
            feature=str(payload["feature"]),
            value=float(payload["value"]),
            baseline=float(payload["baseline"]),
            deviation=float(payload["deviation"]),
        )


@dataclass(frozen=True, slots=True)
class Attribution:
    """Why a fault was flagged, relative to this climber's own sound windows in this video.

    `reliable` is false when the video had no sound-classified window to compare against —
    the research code silently fell back to the all-window mean in that case; this instead
    says so and carries no dominant feature to be confident about.
    """

    reliable: bool
    dominant: FeatureContribution | None
    top_contributors: tuple[FeatureContribution, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reliable": self.reliable,
            "dominant": self.dominant.to_dict() if self.dominant is not None else None,
            "top_contributors": [c.to_dict() for c in self.top_contributors],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Attribution:
        dominant = payload.get("dominant")
        return cls(
            reliable=bool(payload["reliable"]),
            dominant=FeatureContribution.from_dict(dominant) if dominant is not None else None,
            top_contributors=tuple(
                FeatureContribution.from_dict(c) for c in payload.get("top_contributors", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class Segment:
    """One classified window, as the report presents it — no model-internal fields."""

    index: int
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    predicted: str
    confidence: float | None
    low_reliability: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "predicted": self.predicted,
            "confidence": self.confidence,
            "low_reliability": self.low_reliability,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Segment:
        confidence = payload.get("confidence")
        return cls(
            index=int(payload["index"]),
            frame_start=int(payload["frame_start"]),
            frame_end=int(payload["frame_end"]),
            time_start=float(payload["time_start"]),
            time_end=float(payload["time_end"]),
            predicted=str(payload["predicted"]),
            confidence=float(confidence) if confidence is not None else None,
            low_reliability=bool(payload["low_reliability"]),
        )


@dataclass(frozen=True, slots=True)
class Fault:
    """One distinct span of faulty technique, merged from its constituent windows."""

    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    confidence: float | None
    attribution: Attribution
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "confidence": self.confidence,
            "attribution": self.attribution.to_dict(),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Fault:
        confidence = payload.get("confidence")
        return cls(
            frame_start=int(payload["frame_start"]),
            frame_end=int(payload["frame_end"]),
            time_start=float(payload["time_start"]),
            time_end=float(payload["time_end"]),
            confidence=float(confidence) if confidence is not None else None,
            attribution=Attribution.from_dict(payload["attribution"]),
            explanation=str(payload["explanation"]),
        )


@dataclass(frozen=True, slots=True)
class Summary:
    """The climb at a glance.

    `duration_seconds` deliberately repeats `VideoInfo.duration_seconds` — the
    analysis-report spec asks the summary itself to state duration alongside window and
    fault counts, so a caller can render the summary without also holding `VideoInfo`. Both
    are set from the same `VideoMeta` in `build_report` and must not diverge.
    """

    duration_seconds: float
    window_count: int
    faulty_window_count: int
    faulty_proportion: float
    fault_count: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "window_count": self.window_count,
            "faulty_window_count": self.faulty_window_count,
            "faulty_proportion": self.faulty_proportion,
            "fault_count": self.fault_count,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Summary:
        return cls(
            duration_seconds=float(payload["duration_seconds"]),
            window_count=int(payload["window_count"]),
            faulty_window_count=int(payload["faulty_window_count"]),
            faulty_proportion=float(payload["faulty_proportion"]),
            fault_count=int(payload["fault_count"]),
            message=str(payload["message"]),
        )


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Container facts a report needs — not the frames themselves."""

    duration_seconds: float
    frame_count: int
    fps: float
    fps_defaulted: bool
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "fps_defaulted": self.fps_defaulted,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VideoInfo:
        return cls(
            duration_seconds=float(payload["duration_seconds"]),
            frame_count=int(payload["frame_count"]),
            fps=float(payload["fps"]),
            fps_defaulted=bool(payload["fps_defaulted"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Which bundle produced this report, and how it was validated."""

    bundle_id: str
    version: str
    trained_on: str | None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "version": self.version,
            "trained_on": self.trained_on,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelInfo:
        trained_on = payload.get("trained_on")
        return cls(
            bundle_id=str(payload["bundle_id"]),
            version=str(payload["version"]),
            trained_on=str(trained_on) if trained_on is not None else None,
            metrics={str(k): v for k, v in dict(payload.get("metrics", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Everything a client needs to show a climber their analyzed climb.

    This is the only thing the CLI, the API, and the frontend ever see downstream of a
    completed analysis — never a feature vector, a window array, or a model object.
    """

    video: VideoInfo
    summary: Summary
    segments: tuple[Segment, ...]
    faults: tuple[Fault, ...]
    quality_flags: tuple[QualityFlag, ...]
    model: ModelInfo
    scope: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "summary": self.summary.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "faults": [fault.to_dict() for fault in self.faults],
            "quality_flags": [flag.to_dict() for flag in self.quality_flags],
            "model": self.model.to_dict(),
            "scope": self.scope,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalysisReport:
        """The inverse of `to_dict`, for a report read back from storage.

        Strict on shape: a missing key raises rather than defaulting, because a report that
        deserialises with holes in it is one the API would then serve as complete.
        """
        return cls(
            video=VideoInfo.from_dict(payload["video"]),
            summary=Summary.from_dict(payload["summary"]),
            segments=tuple(Segment.from_dict(s) for s in payload["segments"]),
            faults=tuple(Fault.from_dict(f) for f in payload["faults"]),
            quality_flags=tuple(QualityFlag.from_dict(q) for q in payload["quality_flags"]),
            model=ModelInfo.from_dict(payload["model"]),
            scope=str(payload["scope"]),
            limitations=tuple(str(item) for item in payload["limitations"]),
        )

    @classmethod
    def from_json(cls, text: str) -> AnalysisReport:
        import json

        payload: dict[str, Any] = json.loads(text)
        return cls.from_dict(payload)

    def to_json(self, **kwargs: Any) -> str:
        import json

        return json.dumps(self.to_dict(), **kwargs)
