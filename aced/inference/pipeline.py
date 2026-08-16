"""Classifying windows with a loaded bundle.

The label convention inherited from the research is `1 = good technique, 0 = fault`, which
is inverted relative to intuition and was a standing source of bugs in code that passed bare
integers around. It is mapped to a named class here, at the model adapter, and nothing
downstream sees the integers again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from aced.features.windowing import Window
from aced.inference.bundle import Confidence, ModelBundle, ModelKind, Scaling


class PredictedClass(str, Enum):
    SOUND = "sound"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class WindowResult:
    """One classified window, located in the source video."""

    index: int
    predicted: PredictedClass
    frame_start: int
    frame_end: int
    real_length: int
    padded: bool
    #: Present only when the backend produces a calibrated probability. A backend without
    #: one leaves this None rather than supplying a rescaled margin that reads like a
    #: probability and is not one.
    confidence: float | None = None

    @property
    def frame_indices(self) -> range:
        return range(self.frame_start, self.frame_start + self.real_length)


def adapt_sequence(windows: npt.NDArray[np.float64], length: int, n_features: int) -> npt.NDArray[np.float32]:
    """Time and feature axes preserved: `(batch, length, features)`."""
    return np.asarray(windows, dtype=np.float32).reshape((-1, length, n_features))


def adapt_flat(windows: npt.NDArray[np.float64], length: int, n_features: int) -> npt.NDArray[np.float64]:
    """Flattened, time-major: `(batch, length · features)`."""
    return np.asarray(windows, dtype=np.float64).reshape((-1, length * n_features))


class InferencePipeline:
    """Scales (when the bundle says to), adapts, predicts, and names the result."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.manifest = bundle.manifest

    def classify(self, windows: Sequence[Window]) -> list[WindowResult]:
        """One result per window, in input order. An empty input gives an empty result."""
        if not windows:
            return []

        self.bundle.validate_features(windows[0].feature_names)
        length = self.manifest.window_length
        n_features = len(self.manifest.features)

        for window in windows:
            if window.values.shape != (length, n_features):
                raise ValueError(
                    f"window shape {window.values.shape} does not match bundle "
                    f"{self.manifest.id} ({length}, {n_features})"
                )

        flat = np.array([window.flatten() for window in windows], dtype=np.float64)
        prepared = self._scale(flat)

        if self.manifest.model_kind is ModelKind.SEQUENCE:
            model_input: Any = adapt_sequence(prepared, length, n_features)
        else:
            model_input = adapt_flat(prepared, length, n_features)

        classes, confidences = self._predict(model_input)

        return [
            WindowResult(
                index=position,
                predicted=classes[position],
                frame_start=window.frame_start,
                frame_end=window.frame_end,
                real_length=window.real_length,
                padded=window.padded,
                confidence=confidences[position],
            )
            for position, window in enumerate(windows)
        ]

    def _scale(self, flat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply the persisted scaler, or nothing at all if the model scales its own input.

        Only `transform` is ever called. Nothing here fits: that is the whole point of the
        bundle format, and a test asserts it.
        """
        if self.manifest.scaling is Scaling.INTERNAL:
            return flat
        assert self.bundle.scaler is not None  # guaranteed by load_bundle
        scaled: npt.NDArray[np.float64] = self.bundle.scaler.transform(flat)
        return scaled

    def _predict(self, model_input: Any) -> tuple[list[PredictedClass], list[float | None]]:
        model = self.bundle.model
        sound = self.manifest.sound_class

        if self.manifest.confidence is Confidence.PROBABILITY:
            raw = np.asarray(model.predict(model_input, verbose=0)).reshape(-1)
            # The score is the probability of the positive class, which is the sound class.
            is_sound = raw >= self.manifest.threshold
            classes = [PredictedClass.SOUND if flag else PredictedClass.FAULT for flag in is_sound]
            confidences: list[float | None] = [float(value) for value in raw]
            return classes, confidences

        labels = np.asarray(model.predict(model_input)).reshape(-1)
        classes = [
            PredictedClass.SOUND if int(label) == sound else PredictedClass.FAULT for label in labels
        ]
        return classes, [None] * len(classes)


def frame_classes(results: Sequence[WindowResult]) -> dict[int, PredictedClass]:
    """Assign a class to each frame by majority vote across the windows covering it.

    Ties go to sound, matching the research code. Padded positions belong to no frame and
    are excluded by construction: `frame_indices` covers only a window's real frames.
    """
    votes: dict[int, list[int]] = {}
    for result in results:
        for frame in result.frame_indices:
            tally = votes.setdefault(frame, [0, 0])
            tally[0 if result.predicted is PredictedClass.SOUND else 1] += 1

    return {
        frame: PredictedClass.FAULT if fault > sound else PredictedClass.SOUND
        for frame, (sound, fault) in votes.items()
    }
