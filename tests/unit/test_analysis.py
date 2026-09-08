"""The pipeline orchestrator: stages wired together, driven off the bundle's manifest.

The individual stages are covered in their own suites. What is asserted here is the wiring —
that the windower takes its geometry from the manifest rather than from its own defaults,
that the landmarks and frame classes an overlay needs come back alongside the report, and
that a video with no climber in it produces a report rather than an exception.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from aced.analysis import analyze
from aced.features.compute import UnknownFeatureError
from aced.inference.bundle import (
    BundleManifest,
    Confidence,
    ModelBundle,
    ModelKind,
    Scaling,
)
from aced.inference.pipeline import PredictedClass
from aced.pose.landmarks import FrameLandmarks, QualitySignals, VideoMeta
from tests.conftest import SELECTED_FEATURES

SOUND, FAULT = 1, 0


def manifest(**overrides: Any) -> BundleManifest:
    base: dict[str, Any] = {
        "id": "test-bundle",
        "version": "1.0.0",
        "model_kind": ModelKind.FLAT,
        "model_file": "model.joblib",
        "features": SELECTED_FEATURES,
        "window_length": 60,
        "stride": 20,
        "landmarks": (),
        "threshold": 0.5,
        "scaling": Scaling.INTERNAL,
        "confidence": Confidence.NONE,
    }
    base.update(overrides)
    return BundleManifest(**base)


class AlternatingModel:
    """Labels windows sound, fault, sound, fault… so both classes reach the report."""

    def __init__(self, labels: list[int] | None = None) -> None:
        self.labels = labels
        self.rows_seen = 0

    def predict(self, data: Any) -> np.ndarray:
        rows = len(np.asarray(data))
        self.rows_seen = rows
        if self.labels is not None:
            return np.asarray(self.labels[:rows])
        return np.asarray([SOUND if index % 2 == 0 else FAULT for index in range(rows)])


def bundle_with(model: Any = None, **overrides: Any) -> ModelBundle:
    from pathlib import Path

    return ModelBundle(
        manifest=manifest(**overrides),
        model=model if model is not None else AlternatingModel(),
        scaler=None,
        path=Path("/nonexistent/bundle"),
    )


class FakeExtraction:
    """Stands in for `PoseExtraction`: metadata, frames, and final quality signals."""

    def __init__(self, frames: list[FrameLandmarks], meta: VideoMeta | None = None) -> None:
        self.frames = frames
        self.meta = meta or VideoMeta(fps=30.0, frame_count=len(frames), width=1080, height=1920)
        self.iterated = False

    def __iter__(self) -> Iterator[FrameLandmarks]:
        self.iterated = True
        yield from self.frames

    @property
    def quality(self) -> QualitySignals:
        return QualitySignals(
            frames_read=self.meta.frame_count,
            frames_detected=len(self.frames),
            mean_visibility=0.9,
        )


@pytest.fixture
def extraction(reference_frames: list[FrameLandmarks]) -> FakeExtraction:
    return FakeExtraction(reference_frames)


class TestOrchestration:
    def test_a_report_is_produced_for_a_real_climb(self, extraction: FakeExtraction) -> None:
        analysis = analyze("unused.mov", bundle_with(), extraction=extraction)

        assert analysis.report.segments
        assert analysis.report.summary.window_count == len(analysis.report.segments)
        assert analysis.report.model.bundle_id == "test-bundle"

    def test_the_supplied_extraction_is_used_rather_than_the_path(
        self, extraction: FakeExtraction
    ) -> None:
        """The path is never opened when frames are already in hand — which is also why
        a nonexistent one is safe to pass here."""
        analyze("/nonexistent/never-opened.mov", bundle_with(), extraction=extraction)

        assert extraction.iterated

    def test_window_geometry_comes_from_the_manifest(self, extraction: FakeExtraction) -> None:
        """Not from `Windower`'s own defaults: a bundle trained on different windows must
        be segmented the way it was trained, and nothing here may assume 60/20."""
        default = analyze("unused.mov", bundle_with(), extraction=extraction)
        narrow = analyze(
            "unused.mov",
            bundle_with(window_length=30, stride=10),
            extraction=FakeExtraction(list(extraction.frames)),
        )

        assert narrow.report.summary.window_count > default.report.summary.window_count
        assert all(
            segment.frame_end - segment.frame_start == 29 for segment in narrow.report.segments
        )

    def test_every_window_becomes_a_segment_in_order(self, extraction: FakeExtraction) -> None:
        analysis = analyze("unused.mov", bundle_with(), extraction=extraction)

        starts = [segment.frame_start for segment in analysis.report.segments]
        assert starts == sorted(starts)
        assert len(analysis.windows) == len(analysis.report.segments)


class TestOverlayInputs:
    """What `analyze` hands back beyond the report, so an overlay needs no second decode."""

    def test_the_detected_landmarks_come_back_in_order(
        self, extraction: FakeExtraction, reference_frames: list[FrameLandmarks]
    ) -> None:
        analysis = analyze("unused.mov", bundle_with(), extraction=extraction)

        assert len(analysis.landmarks) == len(reference_frames)
        assert [frame.index for frame in analysis.landmarks] == [
            frame.index for frame in reference_frames
        ]

    def test_frame_classes_cover_exactly_the_windowed_frames(
        self, extraction: FakeExtraction
    ) -> None:
        analysis = analyze("unused.mov", bundle_with(), extraction=extraction)

        covered = {frame for window in analysis.windows for frame in window.frame_indices}
        assert set(analysis.frame_classes) == covered

    def test_both_classes_reach_the_frame_assignment(self, extraction: FakeExtraction) -> None:
        analysis = analyze("unused.mov", bundle_with(), extraction=extraction)

        assert set(analysis.frame_classes.values()) == {
            PredictedClass.SOUND,
            PredictedClass.FAULT,
        }

    def test_the_report_carries_no_pipeline_internals(self, extraction: FakeExtraction) -> None:
        """Landmarks and frame classes travel beside the report, never inside it."""
        analysis = analyze("unused.mov", bundle_with(), extraction=extraction)
        payload = analysis.report.to_dict()

        assert "landmarks" not in payload
        assert "frame_classes" not in payload


class TestDegenerateVideos:
    def test_a_video_with_no_detected_pose_still_reports(self) -> None:
        """A readable video with no climber in it is a valid analysis with an empty result,
        not a failure. The report says so through its quality flags."""
        empty = FakeExtraction([], meta=VideoMeta(fps=30.0, frame_count=300, width=64, height=64))
        analysis = analyze("unused.mov", bundle_with(), extraction=empty)

        assert analysis.report.segments == ()
        assert analysis.report.summary.fault_count == 0
        assert analysis.report.quality_flags
        assert analysis.frame_classes == {}

    def test_a_run_too_short_for_a_window_is_still_analyzed(
        self, reference_frames: list[FrameLandmarks]
    ) -> None:
        """Ten frames is a sixth of a window; the windower pads it rather than dropping it."""
        analysis = analyze(
            "unused.mov", bundle_with(), extraction=FakeExtraction(reference_frames[:10])
        )

        assert len(analysis.report.segments) == 1
        assert analysis.report.segments[0].low_reliability


class TestBundleAgreement:
    def test_a_bundle_naming_an_unknown_feature_fails_loudly(
        self, extraction: FakeExtraction
    ) -> None:
        with pytest.raises(UnknownFeatureError, match="not computed"):
            analyze(
                "unused.mov",
                bundle_with(features=("left_elbow_angle", "invented_feature")),
                extraction=extraction,
            )
