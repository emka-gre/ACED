"""Assemble model bundles from the thesis artifacts.

Task 3.2. This is a one-time migration, run again only if an artifact changes. It reads from
the archive and writes `models/<id>/<version>/` — the manifest, the model, and, for the LSTM,
the scaler fitted during training.

The scaler is copied, never refitted. Its provenance is verified here rather than asserted:
the persisted parameters are compared against a refit from the training split, and the
result is recorded in the manifest.

Run:
    python scripts/build_bundles.py [--archive PATH] [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT_ARCHIVE = Path.home() / "Desktop" / "aced-thesis-archive" / "extracted" / "bachelor-project-main"

WINDOW_LENGTH = 60
STRIDE = 20

#: The eight features the research kept after selection, in the order they were kept.
SELECTED_FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "right_wrist_shoulder_dist",
    "left_head_elbow_dist",
    "right_head_wrist_dist",
    "right_hip_elbow_dist",
    "left_wrist_shoulder_dist",
    "left_elbow_height_rel_shoulder",
]

RETAINED_LANDMARKS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]

SCOPE_NOTE = (
    "Classifies arm technique only, from 2D pose. Trained on a small set of climbers in one "
    "gym, labelled by one person. One climber per video."
)


def check_scaler_provenance(scaler: Any, train_csv: Path) -> str:
    """Return a provenance string, refusing to build if the scaler is not what we think."""
    if not train_csv.exists():
        return (
            f"copied from training/LTSM/scaler.pkl; not re-verified in this build "
            f"({train_csv.name} not extracted)"
        )

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    frame = pd.read_csv(train_csv)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    if list(frame.columns) != list(scaler.feature_names_in_):
        raise SystemExit("training split columns do not match the scaler; refusing to build")

    refit = StandardScaler().fit(frame)
    mean_delta = float(np.abs(refit.mean_ - scaler.mean_).max())
    scale_delta = float(np.abs(refit.scale_ - scaler.scale_).max())
    if mean_delta or scale_delta:
        raise SystemExit(
            f"persisted scaler diverges from a refit of {train_csv.name} "
            f"(mean {mean_delta}, scale {scale_delta}); resolve before building"
        )

    return (
        f"fitted during LSTM training on {train_csv.name} ({len(frame)} windows); verified in this "
        "build to be bitwise identical to a refit from that file"
    )


def write_bundle(directory: Path, manifest: dict[str, Any], force: bool) -> None:
    if directory.exists() and not force:
        raise SystemExit(f"{directory.relative_to(ROOT)} already exists; pass --force to rebuild")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest -> {(directory / 'bundle.json').relative_to(ROOT)}")


def build_lstm(archive: Path, force: bool) -> None:
    import joblib

    print("LSTM bundle")
    model_source = archive / "training" / "MODELS" / "LSTM_model.keras"
    scaler_source = archive / "training" / "LTSM" / "scaler.pkl"
    for path in (model_source, scaler_source):
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")

    scaler = joblib.load(scaler_source)
    provenance = check_scaler_provenance(scaler, archive / "ML_pipeline" / "df_X_m_train_3split_2.csv")
    print(f"  scaler: {provenance}")

    expected_width = WINDOW_LENGTH * len(SELECTED_FEATURES)
    if scaler.n_features_in_ != expected_width:
        raise SystemExit(f"scaler expects {scaler.n_features_in_} features, want {expected_width}")

    directory = MODELS / "lstm-arm-technique" / "v1"
    manifest = {
        "id": "lstm-arm-technique",
        "version": "1.0.0",
        "model_kind": "sequence",
        "model_file": "model.keras",
        "features": SELECTED_FEATURES,
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "landmarks": RETAINED_LANDMARKS,
        "threshold": 0.5,
        "scaling": "external",
        "confidence": "probability",
        "sound_class": 1,
        "trained_on": "2026-04-07",
        "metrics": {},
        "scaler_provenance": provenance,
        "notes": SCOPE_NOTE,
    }

    write_bundle(directory, manifest, force)
    shutil.copy2(model_source, directory / "model.keras")
    joblib.dump(scaler, directory / "scaler.joblib")
    print(f"  model  -> {(directory / 'model.keras').relative_to(ROOT)}")
    print(f"  scaler -> {(directory / 'scaler.joblib').relative_to(ROOT)}")


def build_svm(archive: Path, force: bool) -> None:
    import joblib

    print("SVM bundle")
    model_source = archive / "training" / "MODELS" / "SVM_model"
    if not model_source.exists():
        raise SystemExit(f"missing artifact: {model_source}")

    model = joblib.load(model_source)
    steps = dict(getattr(model, "named_steps", {}))
    if "standardscaler" not in steps:
        raise SystemExit(
            "the SVM artifact does not carry its own scaler; the internal-scaling "
            "declaration would be wrong. Investigate before building."
        )
    svc = steps.get("svc")
    has_probability = bool(getattr(svc, "probability", False))
    print(f"  scales internally: yes; calibrated probability: {'yes' if has_probability else 'no'}")

    directory = MODELS / "svm-arm-technique" / "v1"
    manifest = {
        "id": "svm-arm-technique",
        "version": "1.0.0",
        "model_kind": "flat",
        "model_file": "model.joblib",
        "features": SELECTED_FEATURES,
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "landmarks": RETAINED_LANDMARKS,
        "threshold": 0.5,
        # The saved artifact is Pipeline(StandardScaler, SVC): it scales its own input, so
        # the pipeline must hand it raw features or every value is scaled twice.
        "scaling": "internal",
        "confidence": "probability" if has_probability else "none",
        "sound_class": 1,
        "trained_on": None,
        "metrics": {},
        "scaler_provenance": "internal to the saved Pipeline; no separate scaler ships",
        "notes": SCOPE_NOTE,
    }

    write_bundle(directory, manifest, force)
    joblib.dump(model, directory / "model.joblib")
    print(f"  model  -> {(directory / 'model.joblib').relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--force", action="store_true", help="overwrite existing bundles")
    parser.add_argument("--only", choices=["lstm", "svm"], help="build just one bundle")
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"Archive not found at {args.archive}. See THESIS_ARCHIVE.md.")

    if args.only in (None, "lstm"):
        build_lstm(args.archive, args.force)
    if args.only in (None, "svm"):
        build_svm(args.archive, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
