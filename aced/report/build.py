"""Assembling a complete `AnalysisReport` from a finished analysis.

This is the seam between inference and presentation: everything upstream (pose extraction,
features, windowing, classification) is oblivious to reports, and everything downstream (the
CLI, the API, the frontend) is oblivious to windows, feature vectors, and model objects. The
CLI (section 8) and the API worker are expected to be the only callers — both run the same
pipeline up to a `(windows, results)` pair and then call this once.
"""

from __future__ import annotations

from collections.abc import Sequence

from aced.features.windowing import Window
from aced.inference.bundle import BundleManifest
from aced.inference.pipeline import PredictedClass, WindowResult
from aced.pose.landmarks import QualitySignals, VideoMeta
from aced.report.attribution import attribute_fault, compute_baseline
from aced.report.language import attach_explanation
from aced.report.merge import aggregate_confidence, merge_fault_spans
from aced.report.quality import build_quality_flags
from aced.report.scope import LIMITATIONS, SCOPE_STATEMENT
from aced.report.types import AnalysisReport, ModelInfo, Segment, Summary, VideoInfo


def _summary_message(fault_count: int, faulty_window_count: int, window_count: int) -> str:
    if fault_count == 0:
        return "No arm-technique faults were detected."
    faults_word = "fault" if fault_count == 1 else "faults"
    return (
        f"Detected {fault_count} arm-technique {faults_word} across {faulty_window_count} "
        f"of {window_count} analyzed windows."
    )


def build_report(
    *,
    video: VideoMeta,
    quality: QualitySignals,
    windows: Sequence[Window],
    results: Sequence[WindowResult],
    manifest: BundleManifest,
) -> AnalysisReport:
    """Build the report. `windows` and `results` must correspond position-for-position —
    i.e. `results` is exactly what `InferencePipeline.classify(windows)` returned for this
    `windows`, in the same order.
    """
    video_info = VideoInfo(
        duration_seconds=video.duration_seconds,
        frame_count=video.frame_count,
        fps=video.fps,
        fps_defaulted=video.fps_defaulted,
        width=video.width,
        height=video.height,
    )

    segments = tuple(
        Segment(
            index=result.index,
            frame_start=result.frame_start,
            frame_end=result.frame_end,
            time_start=video.timestamp(result.frame_start),
            time_end=video.timestamp(result.frame_end),
            predicted=result.predicted.value,
            confidence=result.confidence,
            low_reliability=result.padded,
        )
        for result in results
    )

    window_count = len(results)
    faulty_window_count = sum(1 for result in results if result.predicted is PredictedClass.FAULT)
    faulty_proportion = faulty_window_count / window_count if window_count else 0.0

    spans = merge_fault_spans(results)
    baseline = compute_baseline(windows, results)
    faults = tuple(
        attach_explanation(
            frame_start=span.frame_start,
            frame_end=span.frame_end,
            time_start=video.timestamp(span.frame_start),
            time_end=video.timestamp(span.frame_end),
            confidence=aggregate_confidence(span),
            attribution=attribute_fault(windows, span.window_indices, baseline),
        )
        for span in spans
    )

    summary = Summary(
        duration_seconds=video.duration_seconds,
        window_count=window_count,
        faulty_window_count=faulty_window_count,
        faulty_proportion=faulty_proportion,
        fault_count=len(faults),
        message=_summary_message(len(faults), faulty_window_count, window_count),
    )

    return AnalysisReport(
        video=video_info,
        summary=summary,
        segments=segments,
        faults=faults,
        quality_flags=tuple(build_quality_flags(quality, manifest.window_length)),
        model=ModelInfo(
            bundle_id=manifest.id,
            version=manifest.version,
            trained_on=manifest.trained_on,
            metrics=dict(manifest.metrics),
        ),
        scope=SCOPE_STATEMENT,
        limitations=LIMITATIONS,
    )
