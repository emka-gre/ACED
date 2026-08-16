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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

MANIFEST_NAME = "bundle.json"
SCALER_NAME = "scaler.joblib"


class ModelKind(str, Enum):
    """How the model wants its input shaped."""

    SEQUENCE = "sequence"  # (batch, length, features) — the LSTM
    FLAT = "flat"  # (batch, length · features) — the SVM


class Scaling(str, Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"


class Confidence(str, Enum):
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
        required = ("id", "version", "model_kind", "model_file", "features", "window_length", "stride")
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
        raise BundleError(f"feature schema does not match bundle {self.manifest.id}: {'; '.join(details)}")


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
    if scaler is not None and getattr(scaler, "n_features_in_", manifest.flat_width) != manifest.flat_width:
        raise BundleError(
            f"bundle {manifest.id}: scaler expects {scaler.n_features_in_} features, "
            f"manifest declares {manifest.flat_width}"
        )

    return ModelBundle(manifest=manifest, model=model, scaler=scaler, path=root)
