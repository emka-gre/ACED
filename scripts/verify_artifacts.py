"""Check that the thesis artifacts load in this environment before anything is built on them.

Task 1.4a. The models were produced under Keras 3.12.1 and scikit-learn of the thesis era; if
they do not load here, every downstream parity claim is meaningless, and it is much cheaper to
find that out now than halfway through the bundle migration.

Usage:
    python scripts/verify_artifacts.py [--archive PATH]

Exits non-zero if any required artifact is missing or fails to load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DEFAULT_ARCHIVE = (
    Path.home() / "Desktop" / "aced-thesis-archive" / "extracted" / "bachelor-project-main"
)

# What the artifacts must look like. Established by inspecting the archive; see
# THESIS_ARCHIVE.md. A mismatch here means the archive is not the one we verified against.
EXPECTED_FEATURE_COUNT = 480
EXPECTED_WINDOW = 60
EXPECTED_SELECTED_FEATURES = 8
EXPECTED_SCALER_SAMPLES = 1588


def _ok(message: str) -> None:
    print(f"  ok    {message}")


def _fail(message: str) -> None:
    print(f"  FAIL  {message}")


def check_scaler(path: Path) -> bool:
    """The LSTM's scaler: 480 flat features, in the order the windower emits them."""
    import joblib

    scaler: Any = joblib.load(path)
    passed = True

    if getattr(scaler, "n_features_in_", None) != EXPECTED_FEATURE_COUNT:
        _fail(
            f"scaler expects {getattr(scaler, 'n_features_in_', None)} features, "
            f"want {EXPECTED_FEATURE_COUNT}"
        )
        passed = False
    else:
        _ok(f"scaler fitted on {EXPECTED_FEATURE_COUNT} features")

    names = list(getattr(scaler, "feature_names_in_", []))
    first, last = names[:1], names[-1:]
    if first != ["left_elbow_angle_t0"] or last != ["left_elbow_height_rel_shoulder_t59"]:
        _fail(f"scaler feature order unexpected: {names[:1]} … {names[-1:]}")
        passed = False
    else:
        _ok("scaler feature names are the expected t-major flat columns")

    if int(getattr(scaler, "n_samples_seen_", 0)) != EXPECTED_SCALER_SAMPLES:
        _fail(
            f"scaler saw {getattr(scaler, 'n_samples_seen_', None)} samples, "
            f"want {EXPECTED_SCALER_SAMPLES}"
        )
        passed = False
    else:
        _ok(f"scaler fitted on {EXPECTED_SCALER_SAMPLES} training windows")

    return passed


def check_scaler_provenance(scaler_path: Path, train_csv: Path) -> bool:
    """Confirm the persisted scaler is the one a refit from the training split produces.

    This is the claim the whole 'remove the inference-time refit' change rests on: if the two
    agree exactly, swapping the refit for a load changes no prediction.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    if not train_csv.exists():
        print(f"  skip  training split not extracted ({train_csv.name}); provenance not re-checked")
        return True

    scaler: Any = joblib.load(scaler_path)
    frame = pd.read_csv(train_csv)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]

    if list(frame.columns) != list(scaler.feature_names_in_):
        _fail("training split columns do not match the scaler's feature order")
        return False

    refit = StandardScaler().fit(frame)
    mean_delta = float(np.abs(refit.mean_ - scaler.mean_).max())
    scale_delta = float(np.abs(refit.scale_ - scaler.scale_).max())

    if mean_delta or scale_delta:
        _fail(f"refit diverges from persisted scaler (mean {mean_delta}, scale {scale_delta})")
        return False

    _ok("persisted scaler is bitwise identical to a refit from the training split")
    return True


def check_svm(path: Path) -> bool:
    """The SVM scales its own input — the bundle must declare internal scaling."""
    import joblib

    model: Any = joblib.load(path)
    passed = True

    steps = dict(getattr(model, "named_steps", {}))
    if "standardscaler" not in steps:
        _fail("SVM artifact has no internal StandardScaler; scaling assumption is wrong")
        passed = False
    else:
        _ok("SVM artifact scales internally (Pipeline with StandardScaler)")

    if getattr(model, "n_features_in_", None) != EXPECTED_FEATURE_COUNT:
        _fail(
            f"SVM expects {getattr(model, 'n_features_in_', None)} features, "
            f"want {EXPECTED_FEATURE_COUNT}"
        )
        passed = False
    else:
        _ok(f"SVM expects {EXPECTED_FEATURE_COUNT} flat features")

    svc = steps.get("svc")
    if svc is not None and getattr(svc, "probability", False):
        print("  note  SVC was fitted with probability=True — a calibrated confidence is available")
    else:
        _ok("SVC has no calibrated probability (confidence kind is 'none', as the bundle declares)")

    return passed


def check_lstm(path: Path) -> bool:
    """The LSTM: a Keras 3 archive taking (60, 8) and emitting one sigmoid."""
    import keras

    model: Any = keras.models.load_model(path, compile=False)
    passed = True

    shape = tuple(model.input_shape)
    if shape != (None, EXPECTED_WINDOW, EXPECTED_SELECTED_FEATURES):
        _fail(
            f"LSTM input shape {shape}, want "
            f"(None, {EXPECTED_WINDOW}, {EXPECTED_SELECTED_FEATURES})"
        )
        passed = False
    else:
        _ok(f"LSTM takes {shape}")

    if tuple(model.output_shape) != (None, 1):
        _fail(f"LSTM output shape {model.output_shape}, want (None, 1)")
        passed = False
    else:
        _ok("LSTM emits a single value per window")

    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=DEFAULT_ARCHIVE, help="extracted thesis archive root"
    )
    args = parser.parse_args()

    root: Path = args.archive
    if not root.exists():
        print(f"Archive not found at {root}. See THESIS_ARCHIVE.md.", file=sys.stderr)
        return 2

    scaler_path = root / "training" / "LTSM" / "scaler.pkl"
    svm_path = root / "training" / "MODELS" / "SVM_model"
    lstm_path = root / "training" / "MODELS" / "LSTM_model.keras"
    train_csv = root / "ML_pipeline" / "df_X_m_train_3split_2.csv"

    results: list[bool] = []
    for label, path, check in (
        ("LSTM scaler", scaler_path, check_scaler),
        ("SVM model", svm_path, check_svm),
        ("LSTM model", lstm_path, check_lstm),
    ):
        print(f"{label}: {path.name}")
        if not path.exists():
            _fail(f"missing: {path}")
            results.append(False)
            continue
        try:
            results.append(check(path))
        except Exception as exc:
            _fail(f"{type(exc).__name__}: {exc}")
            results.append(False)

    print("Scaler provenance")
    if scaler_path.exists():
        try:
            results.append(check_scaler_provenance(scaler_path, train_csv))
        except Exception as exc:
            _fail(f"{type(exc).__name__}: {exc}")
            results.append(False)

    if all(results):
        print("\nAll artifacts load and match expectations.")
        return 0
    print("\nSome artifacts failed. Do not build bundles until this is resolved.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
