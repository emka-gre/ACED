"""Numerical parity with the research pipeline, stage by stage.

The reference files in `fixtures/reference/` were produced by running the thesis code
unmodified (`scripts/generate_reference.py`). If these tests pass, the port computes the
same numbers the shipped models were fitted against; if they fail, the port has drifted and
every prediction downstream is suspect.

Tolerances are tight on purpose. This is the same arithmetic in a different arrangement, so
anything beyond floating-point noise is a real difference, not a rounding artefact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aced.features.compute import FeatureExtractor
from aced.features.windowing import Windower, flat_column_names
from aced.pose.csv_source import read_landmark_csv
from aced.pose.landmarks import FrameLandmarks
from tests.conftest import LANDMARKS, REFERENCE, SELECTED_FEATURES

pytestmark = pytest.mark.parity

#: Same arithmetic, different order of operations — allow only floating-point noise.
TOLERANCE = 1e-9


def _skip_if_missing(path: Path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} not recorded yet — run scripts/generate_reference.py")
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


def build_windows(frames: list[FrameLandmarks]) -> list[np.ndarray]:
    """Run the packaged pipeline over landmark records and return flattened windows."""
    extractor = FeatureExtractor(SELECTED_FEATURES)
    windower = Windower(SELECTED_FEATURES)

    flattened: list[np.ndarray] = []
    for frame in frames:
        for window in windower.push(frame.index, extractor.from_frame(frame)):
            flattened.append(window.flatten())
    for window in windower.close():
        flattened.append(window.flatten())
    return flattened


class TestFeatureParity:
    def test_per_frame_features_match(self, reference_frames: list[FrameLandmarks]) -> None:
        reference = _skip_if_missing(REFERENCE / "features_reference.csv")
        extractor = FeatureExtractor(SELECTED_FEATURES)

        assert len(reference) == len(reference_frames)
        # The research code emits frame and label alongside the eight selected features.
        assert list(reference.columns) == ["frame", "label", *SELECTED_FEATURES]

        computed = np.array([extractor.from_frame(frame) for frame in reference_frames])
        expected = reference[list(SELECTED_FEATURES)].to_numpy()

        np.testing.assert_allclose(computed, expected, rtol=TOLERANCE, atol=TOLERANCE)

    def test_frame_indices_are_preserved(self, reference_frames: list[FrameLandmarks]) -> None:
        reference = _skip_if_missing(REFERENCE / "features_reference.csv")
        assert [frame.index for frame in reference_frames] == reference["frame"].tolist()


class TestWindowParity:
    def test_window_count_and_layout(self, reference_frames: list[FrameLandmarks]) -> None:
        reference = _skip_if_missing(REFERENCE / "segments_reference.csv")
        windows = build_windows(reference_frames)

        assert len(windows) == len(reference), "different number of windows"
        assert list(reference.columns) == list(flat_column_names(SELECTED_FEATURES, 60))
        assert windows[0].shape == (480,)

    def test_window_values_match(self, reference_frames: list[FrameLandmarks]) -> None:
        reference = _skip_if_missing(REFERENCE / "segments_reference.csv")
        windows = np.array(build_windows(reference_frames))
        np.testing.assert_allclose(windows, reference.to_numpy(), rtol=TOLERANCE, atol=TOLERANCE)


class TestEdgeCaseFixtures:
    """The awkward inputs, which the research code exercises least."""

    def test_gap_fixture_splits_into_two_runs(self) -> None:
        frames = read_landmark_csv(LANDMARKS / "edge_gap.csv")
        extractor = FeatureExtractor(SELECTED_FEATURES)
        windower = Windower(SELECTED_FEATURES)

        windows = []
        for frame in frames:
            windows.extend(windower.push(frame.index, extractor.from_frame(frame)))
        windows.extend(windower.close())

        missing = set(range(41, 55))
        assert all(not (set(window.frame_indices) & missing) for window in windows)
        assert any(window.padded for window in windows), "the 41-frame run should be padded"

    def test_short_run_fixture_yields_one_padded_window(self) -> None:
        frames = read_landmark_csv(LANDMARKS / "edge_short_run.csv")
        extractor = FeatureExtractor(SELECTED_FEATURES)
        windower = Windower(SELECTED_FEATURES)

        windows = []
        for frame in frames:
            windows.extend(windower.push(frame.index, extractor.from_frame(frame)))
        windows.extend(windower.close())

        assert len(windows) == 1
        assert windows[0].real_length == 35
        assert windows[0].values.shape == (60, 8)

    def test_stride_unaligned_fixture_flushes_its_tail(self) -> None:
        frames = read_landmark_csv(LANDMARKS / "edge_stride_unaligned.csv")
        extractor = FeatureExtractor(SELECTED_FEATURES)
        windower = Windower(SELECTED_FEATURES)

        windows = []
        for frame in frames:
            windows.extend(windower.push(frame.index, extractor.from_frame(frame)))
        windows.extend(windower.close())

        assert [window.frame_start for window in windows] == [0, 20, 40, 60, 80, 85]
        assert windows[-1].frame_end == 144, "the final frame must be covered"
