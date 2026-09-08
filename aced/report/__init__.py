"""Report generation: turning classified windows into a climber-facing analysis report.

`build_report` is the entry point; everything else here is exported for callers that need a
piece of it directly (the API surfacing quality flags before a full report exists, tests).
"""

from __future__ import annotations

from aced.report.attribution import Baseline, attribute_fault, compute_baseline
from aced.report.build import build_report
from aced.report.language import FEATURE_TEMPLATES, format_timestamp, render_explanation
from aced.report.merge import FaultSpan, aggregate_confidence, merge_fault_spans
from aced.report.quality import build_quality_flags, is_too_short
from aced.report.scope import LIMITATIONS, SCOPE_STATEMENT
from aced.report.types import (
    AnalysisReport,
    Attribution,
    Fault,
    FeatureContribution,
    ModelInfo,
    QualityFlag,
    QualityFlagKind,
    Segment,
    Summary,
    VideoInfo,
)

__all__ = [
    "FEATURE_TEMPLATES",
    "LIMITATIONS",
    "SCOPE_STATEMENT",
    "AnalysisReport",
    "Attribution",
    "Baseline",
    "Fault",
    "FaultSpan",
    "FeatureContribution",
    "ModelInfo",
    "QualityFlag",
    "QualityFlagKind",
    "Segment",
    "Summary",
    "VideoInfo",
    "aggregate_confidence",
    "attribute_fault",
    "build_quality_flags",
    "build_report",
    "compute_baseline",
    "format_timestamp",
    "is_too_short",
    "merge_fault_spans",
    "render_explanation",
]
