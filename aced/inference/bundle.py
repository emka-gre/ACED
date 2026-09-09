"""Model bundles: weights, scaler, and manifest travelling together.

A bundle is a directory:

    models/<name>/<version>/
        bundle.json      the manifest below
        model.keras      or model.joblib, depending on `model_kind`
        scaler.joblib    only when `scaling` is "external"

The research code kept these apart, and that is precisely what made its central defect
possible: the scaler was reconstructed at inference from a training CSV that nothing
guaranteed was unchanged. Shipping the fitted scaler inside the artifact makes that
unrepresentable.

Two declarations in the manifest carry real weight:

`scaling`   "external" means this pipeline scales the features before the model sees them.
            "internal" means the model artifact scales its own input and must be handed raw
            features — the shipped SVM is a `Pipeline(StandardScaler, SVC)` and would be
            silently double-scaled otherwise, producing plausible and wrong predictions.

`confidence` "probability" means the model emits a calibrated score in [0, 1]. "none" means
            it does not, which is the case for the shipped `SVC(probability=False)`. The
            report says nothing about confidence rather than dressing a decision-function
            margin up as one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

MANIFEST_NAME = "bundle.json"
SCALER_NAME = "scaler.joblib"

#: Environment variable naming a bundle directory outright, checked before anything is
#: searched for. This is how a container points at a bundle mounted somewhere else.
BUNDLE_ENV_VAR = "ACED_BUNDLE"

#: The backend used when a caller names none.
#:
#: Settled by task 3.6 on the held-out validation split (178 windows, `scripts/
#: evaluate_bundles.py`). The SVM scores marginally higher — F1 0.902 against 0.885 — but the
#: two backends agree on 171 of 178 windows and the difference rests on five disagreements
#: against two, which an exact McNemar test cannot distinguish from chance (p = 0.45). With
#: no measurable accuracy advantage to trade against, the tiebreak is what else each offers:
#: the LSTM is the only backend declaring `confidence: probability`, so under the SVM every
#: fault in every report would carry no confidence at all.
DEFAULT_BUNDLE_ID = "lstm-arm-technique"


class ModelKind(StrEnum):
    """How the model wants its input shaped."""

    SEQUENCE = "sequence"  # (batch, length, features) — the LSTM
    FLAT = "flat"  # (batch, length · features) — the SVM


class Scaling(StrEnum):
    EXTERNAL = "external"
    INTERNAL = "internal"


class Confidence(StrEnum):
    PROBABILITY = "probability"
    NONE = "none"


class BundleError(RuntimeError):
    """A bundle is missing, incomplete, or contradicts itself."""


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """Everything needed to run a model, and everything needed to describe it afterwards."""

    id: str
    version: str
    model_kind: ModelKind
    model_file: str
    features: tuple[str, ...]
    window_length: int
    stride: int
    landmarks: tuple[str, ...]
    threshold: float
    scaling: Scaling
    confidence: Confidence
    sound_class: int = 1
    trained_on: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    scaler_provenance: str | None = None
    notes: str | None = None

    @property
    def flat_width(self) -> int:
        return self.window_length * len(self.features)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BundleManifest:
        required = (
            "id",
            "version",
            "model_kind",
            "model_file",
            "features",
            "window_length",
            "stride",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise BundleError(f"manifest is missing required keys: {', '.join(missing)}")

        return cls(
            id=str(payload["id"]),
            version=str(payload["version"]),
            model_kind=ModelKind(payload["model_kind"]),
            model_file=str(payload["model_file"]),
            features=tuple(payload["features"]),
            window_length=int(payload["window_length"]),
            stride=int(payload["stride"]),
            landmarks=tuple(payload.get("landmarks", ())),
            threshold=float(payload.get("threshold", 0.5)),
            scaling=Scaling(payload.get("scaling", Scaling.EXTERNAL.value)),
            confidence=Confidence(payload.get("confidence", Confidence.NONE.value)),
            sound_class=int(payload.get("sound_class", 1)),
            trained_on=payload.get("trained_on"),
            metrics=dict(payload.get("metrics", {})),
            scaler_provenance=payload.get("scaler_provenance"),
            notes=payload.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "model_kind": self.model_kind.value,
            "model_file": self.model_file,
            "features": list(self.features),
            "window_length": self.window_length,
            "stride": self.stride,
            "landmarks": list(self.landmarks),
            "threshold": self.threshold,
            "scaling": self.scaling.value,
            "confidence": self.confidence.value,
            "sound_class": self.sound_class,
            "trained_on": self.trained_on,
            "metrics": self.metrics,
            "scaler_provenance": self.scaler_provenance,
            "notes": self.notes,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        """What a report or a health check should say about the model behind a result."""
        return {
            "bundle": self.id,
            "version": self.version,
            "model_kind": self.model_kind.value,
            "trained_on": self.trained_on,
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A loaded bundle: manifest, model object, and the scaler when there is one."""

    manifest: BundleManifest
    model: Any
    scaler: Any | None
    path: Path

    def validate_features(self, produced: tuple[str, ...]) -> None:
        """Refuse to run unless the extractor produces exactly what the model was fitted on.

        Names *and* order, with no reindexing, truncation, or zero-fill. The research code's
        `reindex(columns=..., fill_value=0.0)` turned a column misalignment into a silently
        wrong prediction; this turns it into a load-time error.
        """
        if produced == self.manifest.features:
            return

        expected = self.manifest.features
        if set(produced) == set(expected):
            raise BundleError(
                f"feature order differs from bundle {self.manifest.id}: "
                f"extractor produces {list(produced)}, bundle declares {list(expected)}"
            )
        extra = [name for name in produced if name not in expected]
        absent = [name for name in expected if name not in produced]
        details = []
        if absent:
            details.append(f"missing {', '.join(absent)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise BundleError(
            f"feature schema does not match bundle {self.manifest.id}: {'; '.join(details)}"
        )


def _validate_scaler(manifest: BundleManifest, scaler: Any) -> None:
    """The scaler must have been fitted on exactly this bundle's flattened layout.

    Checking the column *names*, not just the count, is what catches a scaler fitted on the
    right number of features in the wrong order — which would scale every column by another
    column's statistics and produce confident nonsense.
    """
    width = getattr(scaler, "n_features_in_", None)
    if width is not None and int(width) != manifest.flat_width:
        raise BundleError(
            f"bundle {manifest.id}: scaler expects {width} features, "
            f"manifest declares {manifest.flat_width}"
        )

    fitted_names = getattr(scaler, "feature_names_in_", None)
    if fitted_names is None:
        return

    expected = tuple(
        f"{name}_t{t}" for t in range(manifest.window_length) for name in manifest.features
    )
    if tuple(fitted_names) != expected:
        mismatch = next(
            (
                fitted
                for fitted, want in zip(fitted_names, expected, strict=False)
                if fitted != want
            ),
            "<length differs>",
        )
        raise BundleError(
            f"bundle {manifest.id}: scaler was fitted on different columns than the manifest "
            f"declares (first mismatch at {mismatch})"
        )


def load_bundle(path: Path | str) -> ModelBundle:
    """Load a bundle directory, failing loudly on anything incomplete or contradictory."""
    import joblib

    root = Path(path)
    if not root.is_dir():
        raise BundleError(f"bundle directory not found: {root}")

    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise BundleError(f"bundle {root.name} is missing its manifest ({MANIFEST_NAME})")

    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise BundleError(f"bundle {root.name} has an unreadable manifest: {exc}") from exc

    manifest = BundleManifest.from_dict(payload)

    model_path = root / manifest.model_file
    if not model_path.exists():
        raise BundleError(f"bundle {manifest.id} is missing its model file ({manifest.model_file})")

    scaler_path = root / SCALER_NAME
    scaler_present = scaler_path.exists()

    # The declaration and the contents must agree; ambiguity here is how features get
    # scaled twice or not at all.
    if manifest.scaling is Scaling.EXTERNAL and not scaler_present:
        raise BundleError(
            f"bundle {manifest.id} declares external scaling but ships no {SCALER_NAME}"
        )
    if manifest.scaling is Scaling.INTERNAL and scaler_present:
        raise BundleError(
            f"bundle {manifest.id} declares internal scaling but also ships {SCALER_NAME}; "
            "it is ambiguous which scaler applies"
        )

    if manifest.model_kind is ModelKind.SEQUENCE:
        import keras

        model: Any = keras.models.load_model(model_path, compile=False)
    else:
        model = joblib.load(model_path)

    scaler = joblib.load(scaler_path) if scaler_present else None
    if scaler is not None:
        _validate_scaler(manifest, scaler)

    return ModelBundle(manifest=manifest, model=model, scaler=scaler, path=root)


def models_root() -> Path:
    """The repository's `models/` directory.

    Resolved from this file rather than from the working directory, so `aced analyze` finds
    the shipped bundles from anywhere. An installed wheel carries no `models/` — bundles are
    data, not code, and are not packaged — so in that case this points at a directory that
    does not exist and `default_bundle_path` says so plainly.
    """
    return Path(__file__).resolve().parents[2] / "models"


def _version_key(path: Path) -> tuple[int, str]:
    """Order `v2` before `v10`, which a plain lexicographic sort gets backwards."""
    digits = path.name.removeprefix("v")
    return (int(digits), "") if digits.isdigit() else (-1, path.name)


def default_bundle_path(bundle_id: str = DEFAULT_BUNDLE_ID) -> Path:
    """Where to load a bundle from when the caller named none.

    `ACED_BUNDLE` wins outright if set. Otherwise the highest version of `bundle_id` under
    `models/` is used. Failing to find one is an error with an instruction in it, not a
    fallback: guessing at a different backend would silently change every prediction.
    """
    override = os.environ.get(BUNDLE_ENV_VAR)
    if override:
        return Path(override)

    root = models_root() / bundle_id
    versions = (
        sorted(
            (child for child in root.iterdir() if (child / MANIFEST_NAME).is_file()),
            key=_version_key,
        )
        if root.is_dir()
        else []
    )
    if not versions:
        raise BundleError(
            f"no {bundle_id} bundle found under {root} — build one with "
            f"scripts/build_bundles.py, pass an explicit bundle path, or set "
            f"{BUNDLE_ENV_VAR}"
        )
    return versions[-1]
