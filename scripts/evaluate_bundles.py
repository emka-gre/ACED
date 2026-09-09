"""Evaluate the shipped bundles on the held-out validation split.

Task 3.6. Parity proves the port reproduces the research pipeline; it says nothing about
whether the research pipeline is any good. This is the measurement that answers the second
question, and it is what decides which backend ships by default.

Evaluation runs through `InferencePipeline`, not through a reimplementation of it. What is
measured is therefore exactly what the CLI and the API will do to a window — the bundle's own
scaling mode, its adapter, its threshold — rather than a separate copy of that logic that
could be right while the shipped one is wrong.

The fault class is the positive class throughout. For a coaching tool the question is "of the
moves it flagged, how many were really faults, and of the real faults, how many did it
catch" — accuracy alone would look respectable on a model that simply never flagged anything.

Usage:

    python scripts/evaluate_bundles.py                    # metrics to stdout
    python scripts/evaluate_bundles.py --write            # ...and into each bundle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aced.features.windowing import Window, flat_column_names
from aced.inference.bundle import (
    BundleError,
    Confidence,
    ModelBundle,
    load_bundle,
)
from aced.inference.pipeline import InferencePipeline, PredictedClass

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

#: Where the splits are looked for, in order. The thesis archive keeps them under
#: `ML_pipeline/`; a working copy pulled out of it can sit anywhere and be named with --data.
DEFAULT_DATA_DIRS = (
    ROOT / ".thesis-data",
    Path.home()
    / "Desktop"
    / "aced-thesis-archive"
    / "extracted"
    / "bachelor-project-main"
    / "ML_pipeline",
)

#: Filename stems tried for the validation split, features and labels. The archive's own
#: copies carry a `_2` suffix on some files; both spellings are the same data.
FEATURE_NAMES = ("df_X_m_val_3split", "df_X_m_val_3split_2")
LABEL_NAMES = ("df_y_m_val_3split", "df_y_m_val_3split_2")

INDEX_COLUMN = "Unnamed: 0"


@dataclass(frozen=True)
class Metrics:
    """Counts first, rates derived from them — so a rate can always be traced to its counts."""

    n: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    roc_auc: float | None = None

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def to_manifest(self) -> dict[str, Any]:
        """What goes into `bundle.json`, rounded to a precision the sample size supports.

        178 windows do not justify more than three decimal places, and a metrics block that
        reads `0.8426966292134831` invites the reader to believe otherwise.
        """
        payload: dict[str, Any] = {
            "accuracy": round(self.accuracy, 3),
            "fault_precision": round(self.precision, 3),
            "fault_recall": round(self.recall, 3),
            "fault_f1": round(self.f1, 3),
            "validation_windows": self.n,
        }
        if self.roc_auc is not None:
            payload["roc_auc"] = round(self.roc_auc, 3)
        return payload


def find_split(data_dirs: tuple[Path, ...]) -> tuple[Path, Path]:
    """Locate the validation features and labels, or say exactly what was looked for."""
    for directory in data_dirs:
        if not directory.is_dir():
            continue
        features = next(
            (directory / f"{n}.csv" for n in FEATURE_NAMES if (directory / f"{n}.csv").is_file()),
            None,
        )
        labels = next(
            (directory / f"{n}.csv" for n in LABEL_NAMES if (directory / f"{n}.csv").is_file()),
            None,
        )
        if features is not None and labels is not None:
            return features, labels

    searched = "\n  ".join(str(d) for d in data_dirs)
    raise SystemExit(
        "validation split not found. Looked for "
        f"{FEATURE_NAMES[0]}.csv and {LABEL_NAMES[0]}.csv under:\n  {searched}\n"
        "Pass --data with the directory holding them. See THESIS_ARCHIVE.md."
    )


def load_split(features_path: Path, labels_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    if INDEX_COLUMN in features.columns:
        features = features.drop(columns=[INDEX_COLUMN])
    if INDEX_COLUMN in labels.columns:
        labels = labels.drop(columns=[INDEX_COLUMN])

    if len(features) != len(labels):
        raise SystemExit(
            f"{features_path.name} has {len(features)} rows but "
            f"{labels_path.name} has {len(labels)} — these are not the same split"
        )
    return features, labels.to_numpy().reshape(-1).astype(int)


def windows_from(frame: pd.DataFrame, bundle: ModelBundle) -> list[Window]:
    """Turn flat validation rows into windows the shipped pipeline can classify.

    The column order is asserted, not assumed. A silently reordered column would scale every
    feature by another feature's statistics and produce confident nonsense — which is the
    exact failure `validate_features` exists to prevent, so it is checked here too.
    """
    manifest = bundle.manifest
    expected = flat_column_names(manifest.features, manifest.window_length)
    actual = tuple(frame.columns)
    if actual != expected:
        if set(actual) == set(expected):
            raise SystemExit(
                f"the validation split's columns are in a different order from bundle "
                f"{manifest.id}'s flat layout; refusing to reindex"
            )
        raise SystemExit(
            f"the validation split does not match bundle {manifest.id}: "
            f"{len(actual)} columns against {len(expected)} expected"
        )

    values = frame.to_numpy(dtype=np.float64)
    length, n_features = manifest.window_length, len(manifest.features)
    return [
        Window(
            values=row.reshape(length, n_features),
            frame_start=index * length,
            frame_end=index * length + length - 1,
            real_length=length,
            feature_names=manifest.features,
        )
        for index, row in enumerate(values)
    ]


def evaluate(
    bundle: ModelBundle, frame: pd.DataFrame, labels: np.ndarray
) -> tuple[Metrics, np.ndarray]:
    """Run the bundle over the split and count outcomes, fault as the positive class.

    The per-window correctness vector comes back too: comparing two backends means asking
    where they disagree, which their summary metrics cannot answer.
    """
    results = InferencePipeline(bundle).classify(windows_from(frame, bundle))

    predicted_fault = np.array([r.predicted is PredictedClass.FAULT for r in results])
    actual_fault = labels != bundle.manifest.sound_class

    roc_auc = None
    if bundle.manifest.confidence is Confidence.PROBABILITY:
        confidences = [r.confidence for r in results]
        if all(c is not None for c in confidences):
            # The model's probability is of the *sound* class, so the fault score is its
            # complement. Computed here rather than imported so the script has no dependency
            # beyond what the package already needs.
            roc_auc = _roc_auc(actual_fault, 1.0 - np.array(confidences, dtype=float))

    metrics = Metrics(
        n=len(results),
        true_positive=int(np.sum(predicted_fault & actual_fault)),
        false_positive=int(np.sum(predicted_fault & ~actual_fault)),
        true_negative=int(np.sum(~predicted_fault & ~actual_fault)),
        false_negative=int(np.sum(~predicted_fault & actual_fault)),
        roc_auc=roc_auc,
    )
    return metrics, predicted_fault == actual_fault


def _roc_auc(actual: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, by the rank-sum identity.

    Equivalent to the probability that a randomly chosen fault scores above a randomly chosen
    sound window, which is the reading that means something here.
    """
    positives, negatives = int(actual.sum()), int((~actual).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Tied scores share the average of the ranks they span, or every tie would be scored as
    # though the model had separated them.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        tied = inverse == index
        ranks[tied] = ranks[tied].mean()

    rank_sum = ranks[actual].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def mcnemar(first: np.ndarray, second: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on two classifiers' per-window correctness.

    Windows both get right, or both get wrong, carry no information about which is better —
    only the disagreements do. Under the null hypothesis that neither is better, each
    disagreement is a coin flip, so the two-sided exact binomial on those pairs is the test.

    Comparing two accuracies without this is how a three-window difference on a 178-window
    set gets written up as one model beating another.
    """
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    if discordant == 0:
        return first_only, second_only, 1.0

    tail = sum(comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    return first_only, second_only, min(1.0, 2 * tail / 2**discordant)


def compare(results: dict[str, tuple[Metrics, np.ndarray]]) -> None:
    """Say whether the difference between two backends is one the data can support."""
    if len(results) != 2:
        return

    (
        (first_name, (first_metrics, first_correct)),
        (second_name, (second_metrics, second_correct)),
    ) = results.items()
    first_only, second_only, p = mcnemar(first_correct, second_correct)
    agreement = int(np.sum(first_correct == second_correct))

    print("\ncomparison")
    print(f"  agree on           {agreement}/{len(first_correct)} windows")
    print(f"  {first_name} right, other wrong: {first_only}")
    print(f"  {second_name} right, other wrong: {second_only}")
    print(f"  exact McNemar      p = {p:.3f}")
    if p >= 0.05:
        print(
            "  → the difference is not distinguishable from chance on this split; "
            "choose on what else the backends offer, not on these counts"
        )
    else:
        better = first_name if first_metrics.f1 > second_metrics.f1 else second_name
        print(f"  → {better} is measurably better on this split")


def bundle_directories() -> list[Path]:
    return sorted(
        version
        for family in MODELS.iterdir()
        if family.is_dir()
        for version in family.iterdir()
        if (version / "bundle.json").is_file()
    )


def write_metrics(path: Path, metrics: Metrics, split_name: str) -> None:
    manifest_path = path / "bundle.json"
    payload = json.loads(manifest_path.read_text())
    payload["metrics"] = {**metrics.to_manifest(), "validation_split": split_name}
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def report(name: str, metrics: Metrics) -> None:
    print(f"\n{name}")
    print(f"  windows        {metrics.n}")
    print(f"  accuracy       {metrics.accuracy:.3f}")
    print(f"  fault precision{metrics.precision:>8.3f}")
    print(f"  fault recall   {metrics.recall:.3f}")
    print(f"  fault F1       {metrics.f1:.3f}")
    if metrics.roc_auc is not None:
        print(f"  ROC AUC        {metrics.roc_auc:.3f}")
    print(
        f"  confusion      TP {metrics.true_positive}  FP {metrics.false_positive}  "
        f"TN {metrics.true_negative}  FN {metrics.false_negative}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="directory holding the validation split CSVs")
    parser.add_argument(
        "--write", action="store_true", help="record the metrics in each bundle.json"
    )
    args = parser.parse_args()

    data_dirs = (args.data,) if args.data is not None else DEFAULT_DATA_DIRS
    features_path, labels_path = find_split(data_dirs)
    print(f"validation split: {features_path.name} + {labels_path.name}")

    frame, labels = load_split(features_path, labels_path)
    faults = int((labels != 1).sum())
    print(f"  {len(frame)} windows — {faults} fault, {len(frame) - faults} sound")

    evaluated: dict[str, tuple[Metrics, np.ndarray]] = {}
    for directory in bundle_directories():
        try:
            bundle = load_bundle(directory)
        except BundleError as error:
            print(f"\n{directory.name}: {error}", file=sys.stderr)
            continue

        metrics, correct = evaluate(bundle, frame, labels)
        evaluated[bundle.manifest.id] = (metrics, correct)
        report(f"{bundle.manifest.id} {bundle.manifest.version}", metrics)
        if args.write:
            write_metrics(directory, metrics, features_path.stem)
            print("  recorded in bundle.json")

    compare(evaluated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
