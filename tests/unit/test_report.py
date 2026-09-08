"""Report generation: types, fault merging, attribution, quality flags, and language hygiene.

These build synthetic windows and results directly rather than running a real bundle — the
parity suite already pins real-model behaviour end to end; what matters here is what the
report layer does with whatever classifications it is handed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from aced.features.compute import FEATURE_NAMES
from aced.features.windowing import Window
from aced.inference.bundle import BundleManifest, Confidence, ModelKind, Scaling
from aced.inference.pipeline import PredictedClass, WindowResult
from aced.pose.landmarks import QualitySignals, VideoMeta
from aced.report.attribution import attribute_fault, compute_baseline
from aced.report.build import build_report
from aced.report.language import FEATURE_TEMPLATES, format_timestamp, render_explanation
from aced.report.merge import aggregate_confidence, merge_fault_spans
from aced.report.quality import build_quality_flags, is_too_short
from aced.report.types import (
    Attribution,
    FeatureContribution,
    QualityFlagKind,
)

FEATURES = ("f0", "f1")


def window(
    values: list[list[float]], frame_start: int = 0, real_length: int | None = None
) -> Window:
    length = real_length if real_length is not None else len(values)
    array = np.zeros((len(values), len(FEATURES)))
    array[:length] = np.asarray(values[:length], dtype=float)
    return Window(
        values=array,
        frame_start=frame_start,
        frame_end=frame_start + length - 1,
        real_length=length,
        feature_names=FEATURES,
    )


def result(
    index: int,
    predicted: PredictedClass,
    frame_start: int,
    real_length: int,
    confidence: float | None = None,
    padded: bool = False,
) -> WindowResult:
    return WindowResult(
        index=index,
        predicted=predicted,
        frame_start=frame_start,
        frame_end=frame_start + real_length - 1,
        real_length=real_length,
        padded=padded,
        confidence=confidence,
    )


VIDEO = VideoMeta(fps=30.0, frame_count=900, width=1080, height=1920)


class TestFaultMerging:
    def test_overlapping_faulty_windows_merge_into_one_fault(self) -> None:
        # frames 100-160, 120-180, 140-200 — all overlapping, all faulty.
        results = [
            result(0, PredictedClass.FAULT, 100, 60),
            result(1, PredictedClass.FAULT, 120, 60),
            result(2, PredictedClass.FAULT, 140, 60),
        ]
        spans = merge_fault_spans(results)
        assert len(spans) == 1
        assert spans[0].frame_start == 100
        assert spans[0].frame_end == 199  # inclusive: window(140, len=60) ends at frame 199

    def test_separated_faults_kept_distinct(self) -> None:
        results = [
            result(0, PredictedClass.FAULT, 60, 60),
            result(1, PredictedClass.SOUND, 200, 60),
            result(2, PredictedClass.FAULT, 400, 60),
        ]
        spans = merge_fault_spans(results)
        assert [(span.frame_start, span.frame_end) for span in spans] == [(60, 119), (400, 459)]

    def test_no_faulty_windows_gives_no_spans(self) -> None:
        results = [result(0, PredictedClass.SOUND, 0, 60)]
        assert merge_fault_spans(results) == []

    def test_out_of_order_input_still_merges_correctly(self) -> None:
        results = [
            result(1, PredictedClass.FAULT, 120, 60),
            result(0, PredictedClass.FAULT, 100, 60),
        ]
        spans = merge_fault_spans(results)
        assert len(spans) == 1
        assert spans[0].frame_start == 100

    def test_confidence_is_averaged_across_the_span(self) -> None:
        results = [
            result(0, PredictedClass.FAULT, 0, 60, confidence=0.2),
            result(1, PredictedClass.FAULT, 20, 60, confidence=0.4),
        ]
        spans = merge_fault_spans(results)
        assert aggregate_confidence(spans[0]) == pytest.approx(0.3)

    def test_confidence_is_none_when_the_backend_gives_none(self) -> None:
        results = [result(0, PredictedClass.FAULT, 0, 60, confidence=None)]
        spans = merge_fault_spans(results)
        assert aggregate_confidence(spans[0]) is None

    def test_a_sound_window_between_still_overlapping_faults_still_splits_them(self) -> None:
        # At length 60/stride 20, a fault window two positions away from another still
        # overlaps it by 20 frames even with a sound window in between — the sound verdict
        # must still split them into two faults rather than being swallowed by the overlap.
        results = [
            result(0, PredictedClass.FAULT, 100, 60),  # 100-159
            result(1, PredictedClass.SOUND, 120, 60),  # 120-179
            result(2, PredictedClass.FAULT, 140, 60),  # 140-199, overlaps window 0
        ]
        spans = merge_fault_spans(results)
        assert [(span.frame_start, span.frame_end) for span in spans] == [(100, 159), (140, 199)]


class TestSummaryArithmetic:
    def test_matches_the_spec_scenario(self) -> None:
        # 20 windows, 6 faulty, merging into 2 distinct faults.
        results = (
            [result(i, PredictedClass.FAULT, i * 20, 60) for i in range(3)]
            + [result(i, PredictedClass.SOUND, 1000 + i * 20, 60) for i in range(3, 14)]
            + [result(i, PredictedClass.FAULT, 5000 + (i - 14) * 20, 60) for i in range(14, 17)]
            + [result(i, PredictedClass.SOUND, 9000 + i * 20, 60) for i in range(17, 20)]
        )
        windows = [window([[0.0, 0.0]] * 60) for _ in results]
        manifest = BundleManifest(
            id="m",
            version="1",
            model_kind=ModelKind.FLAT,
            model_file="model.joblib",
            features=FEATURES,
            window_length=60,
            stride=20,
            landmarks=(),
            threshold=0.5,
            scaling=Scaling.INTERNAL,
            confidence=Confidence.NONE,
        )
        quality = QualitySignals(frames_read=900, frames_detected=900, mean_visibility=0.9)

        report = build_report(
            video=VIDEO, quality=quality, windows=windows, results=results, manifest=manifest
        )

        assert report.summary.window_count == 20
        assert report.summary.faulty_window_count == 6
        assert report.summary.faulty_proportion == pytest.approx(0.3)
        assert report.summary.fault_count == 2

    def test_no_faults_states_so(self) -> None:
        results = [result(0, PredictedClass.SOUND, 0, 60)]
        windows = [window([[0.0, 0.0]] * 60)]
        manifest = BundleManifest(
            id="m",
            version="1",
            model_kind=ModelKind.FLAT,
            model_file="model.joblib",
            features=FEATURES,
            window_length=60,
            stride=20,
            landmarks=(),
            threshold=0.5,
            scaling=Scaling.INTERNAL,
            confidence=Confidence.NONE,
        )
        quality = QualitySignals(frames_read=60, frames_detected=60, mean_visibility=0.9)
        report = build_report(
            video=VIDEO, quality=quality, windows=windows, results=results, manifest=manifest
        )
        assert report.faults == ()
        assert report.summary.message == "No arm-technique faults were detected."


class TestAttribution:
    def test_dominant_feature_has_the_largest_absolute_deviation(self) -> None:
        # Sound baseline: f0 hovers around 0 with spread; f1 is constant at 10.
        sound = [
            window([[-1.0, 10.0]] * 60),
            window([[1.0, 10.0]] * 60),
        ]
        sound_results = [
            result(0, PredictedClass.SOUND, 0, 60),
            result(1, PredictedClass.SOUND, 60, 60),
        ]
        fault_window = window([[5.0, 10.0]] * 60, frame_start=120)
        all_windows = [*sound, fault_window]
        all_results = [*sound_results, result(2, PredictedClass.FAULT, 120, 60)]

        baseline = compute_baseline(all_windows, all_results)
        assert baseline is not None
        attribution = attribute_fault(all_windows, [2], baseline)

        assert attribution.reliable
        assert attribution.dominant is not None
        assert attribution.dominant.feature == "f0"  # f1 never moved: zero deviation

    def test_feature_with_zero_spread_gets_zero_deviation_not_a_crash(self) -> None:
        sound = [window([[1.0, 1.0]] * 60)]
        sound_results = [result(0, PredictedClass.SOUND, 0, 60)]
        fault_window = window([[1.0, 99.0]] * 60, frame_start=60)
        all_windows = [*sound, fault_window]
        all_results = [*sound_results, result(1, PredictedClass.FAULT, 60, 60)]

        baseline = compute_baseline(all_windows, all_results)
        assert baseline is not None
        attribution = attribute_fault(all_windows, [1], baseline)

        # f0 has zero spread in the baseline and is unchanged: deviation 0, not inf/NaN.
        f0 = next(c for c in attribution.top_contributors if c.feature == "f0")
        assert f0.deviation == 0.0

    def test_no_sound_window_gives_no_baseline(self) -> None:
        results = [result(0, PredictedClass.FAULT, 0, 60)]
        windows = [window([[1.0, 1.0]] * 60)]
        assert compute_baseline(windows, results) is None

        attribution = attribute_fault(windows, [0], None)
        assert attribution.reliable is False
        assert attribution.dominant is None
        assert attribution.top_contributors == ()

    def test_top_contributors_are_ranked_and_capped_at_three(self) -> None:
        # A local 4-feature setup, to exercise the cap at three contributors.
        four = ("a", "b", "c", "d")
        sound_window = Window(
            values=np.zeros((1, 4)), frame_start=0, frame_end=0, real_length=1, feature_names=four
        )
        sound2 = Window(
            values=np.array([[0.0, 0.0, 0.0, 0.0]]),
            frame_start=1,
            frame_end=1,
            real_length=1,
            feature_names=four,
        )
        fault_window = Window(
            values=np.array([[4.0, 3.0, 2.0, 1.0]]),
            frame_start=2,
            frame_end=2,
            real_length=1,
            feature_names=four,
        )
        # Give the baseline nonzero spread so deviations are finite and distinguishable.
        wobble = Window(
            values=np.array([[1.0, 1.0, 1.0, 1.0]]),
            frame_start=3,
            frame_end=3,
            real_length=1,
            feature_names=four,
        )
        windows = [sound_window, sound2, wobble, fault_window]
        results = [
            result(0, PredictedClass.SOUND, 0, 1),
            result(1, PredictedClass.SOUND, 1, 1),
            result(2, PredictedClass.SOUND, 3, 1),
            result(3, PredictedClass.FAULT, 2, 1),
        ]
        baseline = compute_baseline(windows, results)
        assert baseline is not None
        attribution = attribute_fault(windows, [3], baseline)

        assert len(attribution.top_contributors) == 3
        assert [c.feature for c in attribution.top_contributors] == ["a", "b", "c"]


class TestQualityFlags:
    def test_incomplete_pose_detection_is_flagged(self) -> None:
        quality = QualitySignals(frames_read=100, frames_detected=50, mean_visibility=0.9)
        flags = build_quality_flags(quality, window_length=60)
        assert any(flag.kind is QualityFlagKind.INCOMPLETE_POSE_DETECTION for flag in flags)

    def test_well_covered_video_is_not_flagged(self) -> None:
        quality = QualitySignals(frames_read=100, frames_detected=95, mean_visibility=0.9)
        flags = build_quality_flags(quality, window_length=60)
        assert not any(flag.kind is QualityFlagKind.INCOMPLETE_POSE_DETECTION for flag in flags)

    def test_low_visibility_is_flagged(self) -> None:
        quality = QualitySignals(frames_read=100, frames_detected=100, mean_visibility=0.2)
        flags = build_quality_flags(quality, window_length=60)
        assert any(flag.kind is QualityFlagKind.LOW_VISIBILITY for flag in flags)

    def test_video_too_short_is_flagged(self) -> None:
        quality = QualitySignals(frames_read=30, frames_detected=30, mean_visibility=0.9)
        assert is_too_short(quality, window_length=60)
        flags = build_quality_flags(quality, window_length=60)
        assert any(flag.kind is QualityFlagKind.VIDEO_TOO_SHORT for flag in flags)

    def test_a_full_length_window_is_not_too_short(self) -> None:
        quality = QualitySignals(frames_read=60, frames_detected=60, mean_visibility=0.9)
        assert not is_too_short(quality, window_length=60)

    def test_a_fragmented_but_long_video_is_not_reported_as_too_short(self) -> None:
        # Every window this video produces would be padded (many short runs), but the video
        # overall has plenty of detected frames — this is a coverage problem, not a length
        # one, and must not carry the "too short" message, which would mislead the climber
        # about why the results are unreliable.
        quality = QualitySignals(frames_read=1000, frames_detected=900, mean_visibility=0.9)
        assert not is_too_short(quality, window_length=60)
        flags = build_quality_flags(quality, window_length=60)
        assert not any(flag.kind is QualityFlagKind.VIDEO_TOO_SHORT for flag in flags)


class TestExplanationHygiene:
    FORBIDDEN_TERMS = (
        "z-score",
        "zscore",
        "deviation",
        "std",
        "mean",
        "baseline",
        "feature",
        "confidence",
        "threshold",
        "vector",
        "model",
    )

    @pytest.mark.parametrize("feature_name", FEATURE_NAMES)
    @pytest.mark.parametrize("deviation", [-2.0, 2.0])
    def test_no_internal_terminology_leaks(self, feature_name: str, deviation: float) -> None:
        attribution = Attribution(
            reliable=True,
            dominant=FeatureContribution(
                feature=feature_name, value=1.0, baseline=0.0, deviation=deviation
            ),
        )
        text = render_explanation(0.0, 5.0, attribution)
        lowered = text.lower()

        for name in FEATURE_NAMES:
            assert name not in lowered
        for term in self.FORBIDDEN_TERMS:
            assert term not in lowered

    def test_every_selected_feature_has_a_template(self) -> None:
        # The shipped bundles only select eight of the eighteen, but every computed
        # feature should resolve to language, not a generic fallback.
        for name in FEATURE_NAMES:
            assert name in FEATURE_TEMPLATES

    def test_unreliable_attribution_says_so_without_naming_a_feature(self) -> None:
        attribution = Attribution(reliable=False, dominant=None)
        text = render_explanation(0.0, 5.0, attribution)
        assert "sound technique elsewhere" in text
        for name in FEATURE_NAMES:
            assert name not in text.lower()

    def test_zero_deviation_dominant_does_not_claim_a_direction(self) -> None:
        # All features matched the baseline exactly (deviation 0.0): the explanation must
        # not pick a directional phrase for a change that didn't measurably happen.
        attribution = Attribution(
            reliable=True,
            dominant=FeatureContribution(
                feature="left_elbow_angle", value=170.0, baseline=170.0, deviation=0.0
            ),
        )
        text = render_explanation(0.0, 5.0, attribution)
        assert "bent" not in text
        assert "straighter" not in text
        assert "nothing stood out" in text

    def test_names_the_time_range(self) -> None:
        attribution = Attribution(
            reliable=True,
            dominant=FeatureContribution(
                feature="left_elbow_angle", value=90.0, baseline=170.0, deviation=-3.0
            ),
        )
        text = render_explanation(12.0, 16.0, attribution)
        assert format_timestamp(12.0) in text
        assert format_timestamp(16.0) in text
        assert "left arm" in text


class TestSerializationRoundTrip:
    def test_report_serializes_to_json_without_loss(self) -> None:
        manifest = BundleManifest(
            id="m",
            version="1",
            model_kind=ModelKind.FLAT,
            model_file="model.joblib",
            features=FEATURES,
            window_length=60,
            stride=20,
            landmarks=(),
            threshold=0.5,
            scaling=Scaling.INTERNAL,
            confidence=Confidence.NONE,
            trained_on="2026-04-07",
            metrics={"accuracy": 0.9},
        )
        results = [
            result(0, PredictedClass.SOUND, 0, 60),
            result(1, PredictedClass.FAULT, 20, 60, confidence=None),
        ]
        windows = [window([[0.0, 0.0]] * 60), window([[9.0, 9.0]] * 60, frame_start=20)]
        quality = QualitySignals(frames_read=80, frames_detected=80, mean_visibility=0.9)

        report = build_report(
            video=VIDEO, quality=quality, windows=windows, results=results, manifest=manifest
        )

        round_tripped = json.loads(report.to_json())
        assert round_tripped == report.to_dict()
        assert round_tripped["model"]["bundle_id"] == "m"
        assert round_tripped["scope"]
        assert round_tripped["limitations"]

    def test_no_model_internals_are_exposed(self) -> None:
        manifest = BundleManifest(
            id="m",
            version="1",
            model_kind=ModelKind.FLAT,
            model_file="model.joblib",
            features=FEATURES,
            window_length=60,
            stride=20,
            landmarks=(),
            threshold=0.5,
            scaling=Scaling.INTERNAL,
            confidence=Confidence.NONE,
        )
        results = [result(0, PredictedClass.SOUND, 0, 60)]
        windows = [window([[0.0, 0.0]] * 60)]
        quality = QualitySignals(frames_read=60, frames_detected=60, mean_visibility=0.9)
        report = build_report(
            video=VIDEO, quality=quality, windows=windows, results=results, manifest=manifest
        )
        payload = report.to_dict()
        blob = json.dumps(payload)
        assert "scaler" not in blob
        assert "threshold" not in blob
        assert "model_kind" not in blob
