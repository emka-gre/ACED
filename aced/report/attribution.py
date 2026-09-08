"""Explaining a fault by how far its features sit from this climber's own sound technique.

The baseline is the mean and spread of each feature over every real (non-padded) frame of
every sound-classified window in the same video — never the whole thesis training set, and
never, when there is no sound baseline, a silent fallback to the all-window mean. That
fallback is what the research code did; this instead reports the absence and marks the
attribution unreliable (see the `analysis-report` spec's "Insufficient sound baseline").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aced.features.windowing import Window
from aced.inference.pipeline import PredictedClass, WindowResult
from aced.report.types import Attribution, FeatureContribution

#: How many contributing features a fault's explanation ranks, beyond the dominant one.
TOP_CONTRIBUTOR_COUNT = 3


@dataclass(frozen=True, slots=True)
class Baseline:
    """Per-feature mean and spread over this video's sound-classified real frames."""

    feature_names: tuple[str, ...]
    mean: npt.NDArray[np.float64]
    spread: npt.NDArray[np.float64]


def _real_rows(windows: Sequence[Window]) -> npt.NDArray[np.float64]:
    if not windows:
        return np.empty((0, 0))
    rows = [window.values[: window.real_length] for window in windows]
    return np.concatenate(rows, axis=0)


def compute_baseline(windows: Sequence[Window], results: Sequence[WindowResult]) -> Baseline | None:
    """The sound-window baseline for this video, or `None` if it has no sound window at all."""
    sound_windows = [
        window
        for window, result in zip(windows, results, strict=True)
        if result.predicted is PredictedClass.SOUND
    ]
    rows = _real_rows(sound_windows)
    if rows.shape[0] == 0:
        return None

    feature_names = windows[0].feature_names
    return Baseline(
        feature_names=feature_names,
        mean=rows.mean(axis=0),
        spread=rows.std(axis=0),
    )


def _fault_features(windows: Sequence[Window], indices: Sequence[int]) -> npt.NDArray[np.float64]:
    selected = [windows[index] for index in indices]
    rows = _real_rows(selected)
    mean: npt.NDArray[np.float64] = rows.mean(axis=0)
    return mean


def attribute_fault(
    windows: Sequence[Window], window_indices: Sequence[int], baseline: Baseline | None
) -> Attribution:
    """Rank this fault's features by standardized deviation from the sound baseline.

    A feature with no spread in the baseline (every sound window agreed on it exactly)
    yields a deviation of zero rather than a division by zero — there is nothing to deviate
    *from* in a meaningful sense, so it is not reported as the dominant contributor by
    construction.
    """
    if baseline is None:
        return Attribution(reliable=False, dominant=None, top_contributors=())

    values = _fault_features(windows, window_indices)
    # `np.where` evaluates both branches eagerly, so the division by (possibly zero) spread
    # would otherwise warn even on the elements it discards; `errstate` silences that, and
    # `where` still picks 0.0 wherever `spread` is zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        deviations = np.where(
            baseline.spread > 0.0, (values - baseline.mean) / baseline.spread, 0.0
        )

    ranked = sorted(
        range(len(baseline.feature_names)),
        key=lambda i: abs(deviations[i]),
        reverse=True,
    )
    contributions = tuple(
        FeatureContribution(
            feature=baseline.feature_names[i],
            value=float(values[i]),
            baseline=float(baseline.mean[i]),
            deviation=float(deviations[i]),
        )
        for i in ranked[:TOP_CONTRIBUTOR_COUNT]
    )
    dominant = contributions[0] if contributions else None
    return Attribution(reliable=True, dominant=dominant, top_contributors=contributions)
