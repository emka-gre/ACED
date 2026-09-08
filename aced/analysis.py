"""One run of the whole pipeline, from a video file to a finished report.

`build_report` is the seam between inference and presentation; this is the seam above it,
and it exists because two callers need exactly the same sequence of stages: the CLI, and the
API worker that section 9 adds. Written once here, neither can drift from the other, which is
what the API's "the report matches the CLI's output for the same video" test ultimately
rests on.

Everything is driven off the bundle's manifest — the feature list, the window length, the
stride — so an analysis is defined by the bundle it ran under and nothing is hardcoded to
the shipped models' particular numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aced.features.compute import FeatureExtractor
from aced.features.windowing import Window, Windower
from aced.inference.bundle import ModelBundle
from aced.inference.pipeline import InferencePipeline, PredictedClass, frame_classes
from aced.pose.extractor import PoseExtraction, extract_poses
from aced.pose.landmarks import FrameLandmarks
from aced.report.build import build_report
from aced.report.types import AnalysisReport


@dataclass(frozen=True, slots=True)
class Analysis:
    """A finished analysis: the report, and what an overlay would need to draw it.

    The landmarks and frame classes are carried alongside the report rather than folded into
    it — they are pipeline internals no climber should ever be shown, and the report type
    deliberately holds no model-internal fields. A caller rendering an overlay in the same
    process reuses them and does not decode the video a second time.
    """

    report: AnalysisReport
    landmarks: tuple[FrameLandmarks, ...]
    frame_classes: Mapping[int, PredictedClass]
    windows: tuple[Window, ...]


def analyze(
    source: Path | str,
    bundle: ModelBundle,
    *,
    extraction: PoseExtraction | None = None,
) -> Analysis:
    """Run pose extraction, features, windowing, inference and report building.

    Raises `UnreadableVideoError` if the file cannot be decoded and `BundleError` if the
    bundle and the extractor disagree about features. A video that decodes fine but contains
    no detectable climber is *not* an error: it produces a valid report whose quality flags
    say no pose was found, because that is a real answer about a real video.

    `extraction` is the injection point that `extract_poses` provides for the pose stage, so
    a caller with frames already in hand — a test, or a worker that extracted once and is
    re-running under a second bundle — does not have to decode again.
    """
    extraction = extract_poses(source) if extraction is None else extraction
    manifest = bundle.manifest

    extractor = FeatureExtractor(manifest.features)
    windower = Windower(
        manifest.features,
        length=manifest.window_length,
        stride=manifest.stride,
    )

    landmarks: list[FrameLandmarks] = []
    windows: list[Window] = []
    for frame in extraction:
        landmarks.append(frame)
        windows.extend(windower.push(frame.index, extractor.from_frame(frame)))
    windows.extend(windower.close())

    results = InferencePipeline(bundle).classify(windows)

    report = build_report(
        video=extraction.meta,
        quality=extraction.quality,
        windows=windows,
        results=results,
        manifest=manifest,
    )

    return Analysis(
        report=report,
        landmarks=tuple(landmarks),
        frame_classes=frame_classes(results),
        windows=tuple(windows),
    )
