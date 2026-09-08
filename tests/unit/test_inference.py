"""Bundle loading, scaling modes, thresholds, and frame attribution.

These use stand-in models rather than the shipped artifacts: the real ones are covered by the
parity suite, and the behaviour under test here — what happens when a bundle is wrong — is
exactly what a correct artifact cannot exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from aced.features.windowing import Window
from aced.inference.bundle import (
    BUNDLE_ENV_VAR,
    DEFAULT_BUNDLE_ID,
    BundleError,
    BundleManifest,
    Confidence,
    ModelBundle,
    ModelKind,
    Scaling,
    default_bundle_path,
    load_bundle,
    models_root,
)
from aced.inference.pipeline import InferencePipeline, PredictedClass, WindowResult, frame_classes

FEATURES = ("f0", "f1")
LENGTH = 4


def manifest(**overrides: Any) -> BundleManifest:
    base: dict[str, Any] = {
        "id": "test-bundle",
        "version": "1.0.0",
        "model_kind": ModelKind.FLAT,
        "model_file": "model.joblib",
        "features": FEATURES,
        "window_length": LENGTH,
        "stride": 2,
        "landmarks": (),
        "threshold": 0.5,
        "scaling": Scaling.INTERNAL,
        "confidence": Confidence.NONE,
    }
    base.update(overrides)
    return BundleManifest(**base)


def window(
    values: list[list[float]], frame_start: int = 0, real_length: int | None = None
) -> Window:
    array = np.zeros((LENGTH, len(FEATURES)))
    array[: len(values)] = np.asarray(values, dtype=float)
    length = real_length if real_length is not None else len(values)
    return Window(
        values=array,
        frame_start=frame_start,
        frame_end=frame_start + length - 1,
        real_length=length,
        feature_names=FEATURES,
    )


class LabelModel:
    """A flat-input model that returns fixed labels and records what it was given."""

    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.seen: Any = None

    def predict(self, data: Any) -> np.ndarray:
        self.seen = np.asarray(data)
        return np.asarray(self.labels[: len(self.seen)])


class ProbabilityModel:
    """A sequence-input model that returns fixed probabilities."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities
        self.seen: Any = None

    def predict(self, data: Any, verbose: int = 0) -> np.ndarray:
        self.seen = np.asarray(data)
        return np.asarray(self.probabilities[: len(self.seen)]).reshape(-1, 1)


class DoublingScaler:
    """Stands in for a fitted scaler; doubles every value so its effect is visible."""

    n_features_in_ = LENGTH * len(FEATURES)

    def __init__(self) -> None:
        self.transform_calls = 0

    def transform(self, data: Any) -> np.ndarray:
        self.transform_calls += 1
        return np.asarray(data) * 2.0

    def fit(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("inference must never fit")


class TestManifest:
    def test_missing_keys_are_named(self) -> None:
        with pytest.raises(BundleError, match="window_length"):
            BundleManifest.from_dict(
                {"id": "x", "version": "1", "model_kind": "flat", "model_file": "m"}
            )

    def test_round_trips_through_dict(self) -> None:
        original = manifest(trained_on="2026-04-07", metrics={"accuracy": 0.8})
        assert BundleManifest.from_dict(original.to_dict()) == original

    def test_flat_width_is_length_times_features(self) -> None:
        assert manifest().flat_width == LENGTH * len(FEATURES)

    def test_provenance_carries_what_a_report_needs(self) -> None:
        provenance = manifest(trained_on="2026-04-07").provenance
        assert provenance["bundle"] == "test-bundle"
        assert provenance["version"] == "1.0.0"
        assert provenance["trained_on"] == "2026-04-07"


def write_bundle(
    directory: Path, overrides: dict[str, Any], with_scaler: bool, with_model: bool = True
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = manifest().to_dict()
    payload.update(overrides)
    (directory / "bundle.json").write_text(json.dumps(payload))
    if with_model:
        joblib.dump(LabelModel([1]), directory / "model.joblib")
    if with_scaler:
        joblib.dump(DoublingScaler(), directory / "scaler.joblib")
    return directory


class TestBundleLoading:
    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError, match="not found"):
            load_bundle(tmp_path / "nope")

    def test_missing_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(BundleError, match="manifest"):
            load_bundle(tmp_path / "empty")

    def test_missing_model_file(self, tmp_path: Path) -> None:
        path = write_bundle(tmp_path / "b", {}, with_scaler=False, with_model=False)
        with pytest.raises(BundleError, match="model file"):
            load_bundle(path)

    def test_external_scaling_without_a_scaler_is_rejected(self, tmp_path: Path) -> None:
        path = write_bundle(tmp_path / "b", {"scaling": "external"}, with_scaler=False)
        with pytest.raises(BundleError, match=r"ships no scaler\.joblib"):
            load_bundle(path)

    def test_internal_scaling_with_a_scaler_is_rejected(self, tmp_path: Path) -> None:
        # Ambiguity here is how the SVM would get scaled twice.
        path = write_bundle(tmp_path / "b", {"scaling": "internal"}, with_scaler=True)
        with pytest.raises(BundleError, match="ambiguous"):
            load_bundle(path)

    def test_scaler_width_must_match_the_manifest(self, tmp_path: Path) -> None:
        path = write_bundle(
            tmp_path / "b", {"scaling": "external", "window_length": 8}, with_scaler=True
        )
        with pytest.raises(BundleError, match="scaler expects"):
            load_bundle(path)

    def test_valid_bundle_loads(self, tmp_path: Path) -> None:
        path = write_bundle(tmp_path / "b", {"scaling": "external"}, with_scaler=True)
        bundle = load_bundle(path)
        assert bundle.manifest.id == "test-bundle"
        assert bundle.scaler is not None


class TestFeatureSchemaValidation:
    def test_matching_schema_passes(self) -> None:
        bundle = ModelBundle(manifest(), LabelModel([1]), None, Path("."))
        bundle.validate_features(FEATURES)

    def test_reordered_features_are_rejected(self) -> None:
        # Same names, wrong order: every column would be scaled by another column's
        # statistics. The research code reindexed silently; this refuses.
        bundle = ModelBundle(manifest(), LabelModel([1]), None, Path("."))
        with pytest.raises(BundleError, match="order differs"):
            bundle.validate_features(("f1", "f0"))

    def test_missing_feature_is_named(self) -> None:
        bundle = ModelBundle(manifest(), LabelModel([1]), None, Path("."))
        with pytest.raises(BundleError, match="missing f1"):
            bundle.validate_features(("f0",))

    def test_unexpected_feature_is_named(self) -> None:
        bundle = ModelBundle(manifest(), LabelModel([1]), None, Path("."))
        with pytest.raises(BundleError, match="unexpected f2"):
            bundle.validate_features(("f0", "f1", "f2"))


class TestScalingModes:
    def test_external_scaling_transforms_before_predicting(self) -> None:
        scaler = DoublingScaler()
        model = LabelModel([1])
        bundle = ModelBundle(manifest(scaling=Scaling.EXTERNAL), model, scaler, Path("."))

        InferencePipeline(bundle).classify([window([[1.0, 2.0]] * LENGTH)])

        assert scaler.transform_calls == 1
        assert model.seen[0][0] == 2.0, "the model should have seen scaled values"

    def test_internal_scaling_passes_raw_values(self) -> None:
        # The shipped SVM scales its own input; handing it pre-scaled features would
        # double-scale and produce confident nonsense.
        model = LabelModel([1])
        bundle = ModelBundle(manifest(scaling=Scaling.INTERNAL), model, None, Path("."))

        InferencePipeline(bundle).classify([window([[1.0, 2.0]] * LENGTH)])

        assert model.seen[0][0] == 1.0, "the model should have seen raw values"

    def test_nothing_is_ever_fitted(self) -> None:
        scaler = DoublingScaler()  # its fit() raises
        bundle = ModelBundle(manifest(scaling=Scaling.EXTERNAL), LabelModel([1]), scaler, Path("."))
        InferencePipeline(bundle).classify([window([[1.0, 2.0]] * LENGTH)])


class TestClassification:
    def test_empty_input_gives_empty_output(self) -> None:
        bundle = ModelBundle(manifest(), LabelModel([]), None, Path("."))
        assert InferencePipeline(bundle).classify([]) == []

    def test_one_result_per_window_in_order(self) -> None:
        bundle = ModelBundle(manifest(), LabelModel([1, 0, 1]), None, Path("."))
        windows = [window([[1.0, 1.0]] * LENGTH, frame_start=start) for start in (0, 2, 4)]

        results = InferencePipeline(bundle).classify(windows)

        assert [result.index for result in results] == [0, 1, 2]
        assert [result.frame_start for result in results] == [0, 2, 4]
        assert [result.predicted for result in results] == [
            PredictedClass.SOUND,
            PredictedClass.FAULT,
            PredictedClass.SOUND,
        ]

    def test_label_backend_reports_no_confidence(self) -> None:
        bundle = ModelBundle(manifest(), LabelModel([1]), None, Path("."))
        results = InferencePipeline(bundle).classify([window([[1.0, 1.0]] * LENGTH)])
        assert results[0].confidence is None

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.49, PredictedClass.FAULT),
            (0.50, PredictedClass.SOUND),  # at the threshold, inclusive
            (0.51, PredictedClass.SOUND),
        ],
    )
    def test_threshold_is_inclusive_and_comes_from_the_bundle(
        self, probability: float, expected: PredictedClass
    ) -> None:
        bundle = ModelBundle(
            manifest(model_kind=ModelKind.SEQUENCE, confidence=Confidence.PROBABILITY),
            ProbabilityModel([probability]),
            None,
            Path("."),
        )
        results = InferencePipeline(bundle).classify([window([[1.0, 1.0]] * LENGTH)])
        assert results[0].predicted is expected
        assert results[0].confidence == pytest.approx(probability)

    def test_a_raised_threshold_changes_the_verdict(self) -> None:
        strict = ModelBundle(
            manifest(
                model_kind=ModelKind.SEQUENCE, confidence=Confidence.PROBABILITY, threshold=0.9
            ),
            ProbabilityModel([0.6]),
            None,
            Path("."),
        )
        results = InferencePipeline(strict).classify([window([[1.0, 1.0]] * LENGTH)])
        assert results[0].predicted is PredictedClass.FAULT

    def test_sequence_backend_receives_time_and_feature_axes(self) -> None:
        model = ProbabilityModel([0.9])
        bundle = ModelBundle(
            manifest(model_kind=ModelKind.SEQUENCE, confidence=Confidence.PROBABILITY),
            model,
            None,
            Path("."),
        )
        InferencePipeline(bundle).classify([window([[1.0, 2.0]] * LENGTH)])
        assert model.seen.shape == (1, LENGTH, len(FEATURES))

    def test_flat_backend_receives_one_row_per_window(self) -> None:
        model = LabelModel([1])
        bundle = ModelBundle(manifest(), model, None, Path("."))
        InferencePipeline(bundle).classify([window([[1.0, 2.0]] * LENGTH)])
        assert model.seen.shape == (1, LENGTH * len(FEATURES))

    def test_window_of_the_wrong_shape_is_rejected(self) -> None:
        bundle = ModelBundle(manifest(window_length=99), LabelModel([1]), None, Path("."))
        with pytest.raises(ValueError, match="does not match bundle"):
            InferencePipeline(bundle).classify([window([[1.0, 1.0]] * LENGTH)])


def result(predicted: PredictedClass, frame_start: int, real_length: int) -> WindowResult:
    return WindowResult(
        index=0,
        predicted=predicted,
        frame_start=frame_start,
        frame_end=frame_start + real_length - 1,
        real_length=real_length,
        padded=False,
    )


class TestFrameAttribution:
    def test_majority_wins(self) -> None:
        results = [
            result(PredictedClass.SOUND, 0, 3),
            result(PredictedClass.SOUND, 0, 3),
            result(PredictedClass.FAULT, 0, 3),
        ]
        assert frame_classes(results)[1] is PredictedClass.SOUND

    def test_ties_go_to_sound(self) -> None:
        # Matches the research code: an even split is not evidence of a fault.
        results = [result(PredictedClass.SOUND, 0, 2), result(PredictedClass.FAULT, 0, 2)]
        assert frame_classes(results)[0] is PredictedClass.SOUND

    def test_fault_majority_wins(self) -> None:
        results = [
            result(PredictedClass.FAULT, 5, 2),
            result(PredictedClass.FAULT, 5, 2),
            result(PredictedClass.SOUND, 5, 2),
        ]
        assert frame_classes(results)[5] is PredictedClass.FAULT

    def test_padded_positions_cast_no_votes(self) -> None:
        # A window padded to 4 but covering 2 real frames must not vote for frames 2 and 3.
        results = [result(PredictedClass.FAULT, 0, 2)]
        assert set(frame_classes(results)) == {0, 1}

    def test_frames_no_window_covers_are_absent(self) -> None:
        results = [result(PredictedClass.SOUND, 10, 2)]
        assert set(frame_classes(results)) == {10, 11}


class TestDefaultBundleResolution:
    """Which bundle runs when the caller names none.

    This is the one place a wrong answer is silent: every other bundle failure raises, but
    quietly resolving to a different backend would produce a complete, plausible report from
    a model the caller did not choose.
    """

    def test_the_environment_variable_wins_outright(self, monkeypatch: Any) -> None:
        monkeypatch.setenv(BUNDLE_ENV_VAR, "/mounted/bundle")

        assert default_bundle_path() == Path("/mounted/bundle")

    def test_the_shipped_bundle_is_found_without_configuration(self, monkeypatch: Any) -> None:
        monkeypatch.delenv(BUNDLE_ENV_VAR, raising=False)
        root = models_root() / DEFAULT_BUNDLE_ID
        if not root.is_dir():
            pytest.skip(f"{DEFAULT_BUNDLE_ID} bundle not built — run scripts/build_bundles.py")

        resolved = default_bundle_path()

        assert resolved.parent == root
        assert (resolved / "bundle.json").is_file()

    def test_versions_are_ordered_numerically(self, monkeypatch: Any, tmp_path: Path) -> None:
        """`v10` is newer than `v2`; a lexicographic sort says the opposite."""
        monkeypatch.delenv(BUNDLE_ENV_VAR, raising=False)
        monkeypatch.setattr("aced.inference.bundle.models_root", lambda: tmp_path)
        for name in ("v1", "v2", "v10"):
            version = tmp_path / "some-bundle" / name
            version.mkdir(parents=True)
            (version / "bundle.json").write_text("{}")

        assert default_bundle_path("some-bundle").name == "v10"

    def test_a_directory_without_a_manifest_is_not_a_version(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(BUNDLE_ENV_VAR, raising=False)
        monkeypatch.setattr("aced.inference.bundle.models_root", lambda: tmp_path)
        (tmp_path / "some-bundle" / "v1").mkdir(parents=True)
        (tmp_path / "some-bundle" / "v2").mkdir(parents=True)
        (tmp_path / "some-bundle" / "v1" / "bundle.json").write_text("{}")

        assert default_bundle_path("some-bundle").name == "v1"

    def test_no_bundle_at_all_says_what_to_do(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.delenv(BUNDLE_ENV_VAR, raising=False)
        monkeypatch.setattr("aced.inference.bundle.models_root", lambda: tmp_path)

        with pytest.raises(BundleError, match=BUNDLE_ENV_VAR):
            default_bundle_path("absent-bundle")
