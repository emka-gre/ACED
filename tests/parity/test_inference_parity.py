"""End-to-end inference parity: the packaged pipeline against the research pipeline.

This is the gate the change proposal names. If the classifications here match the recorded
reference, the port reproduces the validated method — including the removal of the
inference-time scaler refit, which these tests prove costs nothing numerically.

Both bundles are covered because they take different paths: the LSTM is externally scaled
and emits probabilities, the SVM scales internally and emits bare labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aced.features.compute import FeatureExtractor
from aced.features.windowing import Window, Windower
from aced.inference.bundle import Confidence, ModelBundle, Scaling, load_bundle
from aced.inference.pipeline import InferencePipeline, PredictedClass, frame_classes
from aced.pose.landmarks import FrameLandmarks
from tests.conftest import REFERENCE

pytestmark = pytest.mark.parity

MODELS = Path(__file__).resolve().parents[2] / "models"
LSTM_BUNDLE = MODELS / "lstm-arm-technique" / "v1"
SVM_BUNDLE = MODELS / "svm-arm-technique" / "v1"


def load_or_skip(path: Path) -> ModelBundle:
    if not path.exists():
        pytest.skip(f"{path.name} bundle not built — run scripts/build_bundles.py")
    return load_bundle(path)


def reference_or_skip(name: str) -> pd.DataFrame:
    path = REFERENCE / name
    if not path.exists():
        pytest.skip(f"{name} not recorded — run scripts/generate_reference.py")
    return pd.read_csv(path)


def build_windows(frames: list[FrameLandmarks], features: tuple[str, ...]) -> list[Window]:
    extractor = FeatureExtractor(features)
    windower = Windower(features)
    windows: list[Window] = []
    for frame in frames:
        windows.extend(windower.push(frame.index, extractor.from_frame(frame)))
    windows.extend(windower.close())
    return windows


@pytest.fixture(scope="module")
def lstm() -> ModelBundle:
    return load_or_skip(LSTM_BUNDLE)


@pytest.fixture(scope="module")
def svm() -> ModelBundle:
    return load_or_skip(SVM_BUNDLE)


class TestBundleShape:
    def test_lstm_declares_external_scaling_and_probabilities(self, lstm: ModelBundle) -> None:
        assert lstm.manifest.scaling is Scaling.EXTERNAL
        assert lstm.manifest.confidence is Confidence.PROBABILITY
        assert lstm.scaler is not None
        assert lstm.scaler.n_features_in_ == 480

    def test_svm_declares_internal_scaling_and_no_probability(self, svm: ModelBundle) -> None:
        # The saved artifact is Pipeline(StandardScaler, SVC): applying a bundle scaler on
        # top would scale every value twice.
        assert svm.manifest.scaling is Scaling.INTERNAL
        assert svm.manifest.confidence is Confidence.NONE
        assert svm.scaler is None

    def test_both_bundles_declare_the_same_feature_schema(
        self, lstm: ModelBundle, svm: ModelBundle, selected_features: tuple[str, ...]
    ) -> None:
        assert lstm.manifest.features == selected_features
        assert svm.manifest.features == selected_features
        assert lstm.manifest.window_length == svm.manifest.window_length == 60
        assert lstm.manifest.stride == svm.manifest.stride == 20


class TestLstmParity:
    def test_window_classifications_match(
        self, lstm: ModelBundle, reference_frames: list[FrameLandmarks]
    ) -> None:
        reference = reference_or_skip("predictions_lstm_reference.csv")
        windows = build_windows(reference_frames, lstm.manifest.features)
        results = InferencePipeline(lstm).classify(windows)

        assert len(results) == len(reference)
        expected = [
            PredictedClass.SOUND if value == 1 else PredictedClass.FAULT
            for value in reference["prediction"]
        ]
        assert [result.predicted for result in results] == expected

    def test_probabilities_match(
        self, lstm: ModelBundle, reference_frames: list[FrameLandmarks]
    ) -> None:
        reference = reference_or_skip("predictions_lstm_reference.csv")
        windows = build_windows(reference_frames, lstm.manifest.features)
        results = InferencePipeline(lstm).classify(windows)

        produced = np.array([result.confidence for result in results], dtype=float)
        # Float32 inference, so allow float32 noise but nothing structural.
        np.testing.assert_allclose(
            produced, reference["probability"].to_numpy(), rtol=1e-5, atol=1e-6
        )

    def test_nothing_is_fitted_at_inference(
        self,
        lstm: ModelBundle,
        reference_frames: list[FrameLandmarks],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The research pipeline refit its scaler on every run. This one must not.

        Parity above already shows the loaded scaler and the refit agree numerically; this
        shows the refit no longer happens at all.
        """

        def explode(*args: object, **kwargs: object) -> None:
            raise AssertionError("inference fitted a scaler")

        monkeypatch.setattr(type(lstm.scaler), "fit", explode)
        monkeypatch.setattr(type(lstm.scaler), "fit_transform", explode)

        windows = build_windows(reference_frames, lstm.manifest.features)
        assert len(InferencePipeline(lstm).classify(windows)) == 29


class TestSvmParity:
    def test_window_classifications_match(
        self, svm: ModelBundle, reference_frames: list[FrameLandmarks]
    ) -> None:
        reference = reference_or_skip("predictions_svm_reference.csv")
        windows = build_windows(reference_frames, svm.manifest.features)
        results = InferencePipeline(svm).classify(windows)

        assert len(results) == len(reference)
        expected = [
            PredictedClass.SOUND if value == 1 else PredictedClass.FAULT
            for value in reference["prediction"]
        ]
        assert [result.predicted for result in results] == expected

    def test_no_confidence_is_invented(
        self, svm: ModelBundle, reference_frames: list[FrameLandmarks]
    ) -> None:
        windows = build_windows(reference_frames, svm.manifest.features)
        results = InferencePipeline(svm).classify(windows)
        assert all(result.confidence is None for result in results)


class TestBackendsAgreeOnShape:
    def test_result_interface_is_identical(
        self, lstm: ModelBundle, svm: ModelBundle, reference_frames: list[FrameLandmarks]
    ) -> None:
        windows = build_windows(reference_frames, lstm.manifest.features)
        lstm_results = InferencePipeline(lstm).classify(windows)
        svm_results = InferencePipeline(svm).classify(windows)

        assert len(lstm_results) == len(svm_results)
        for left, right in zip(lstm_results, svm_results, strict=True):
            assert (left.frame_start, left.frame_end) == (right.frame_start, right.frame_end)
            assert left.real_length == right.real_length
            assert isinstance(left.predicted, PredictedClass)
            assert isinstance(right.predicted, PredictedClass)


class TestFrameAttribution:
    def test_every_detected_frame_gets_a_class(
        self, svm: ModelBundle, reference_frames: list[FrameLandmarks]
    ) -> None:
        windows = build_windows(reference_frames, svm.manifest.features)
        results = InferencePipeline(svm).classify(windows)
        classes = frame_classes(results)

        covered = {frame for result in results for frame in result.frame_indices}
        assert set(classes) == covered
        assert covered <= {frame.index for frame in reference_frames}
