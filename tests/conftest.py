"""Shared fixture paths and loaders.

Everything here reads from `fixtures/`, never from the thesis archive, so the suite runs on
a machine that has only this repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aced.pose.csv_source import read_landmark_csv
from aced.pose.landmarks import FrameLandmarks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LANDMARKS = FIXTURES / "landmarks"
REFERENCE = FIXTURES / "reference"

#: The eight features the shipped bundles select, in the order the research code kept them.
SELECTED_FEATURES: tuple[str, ...] = (
    "left_elbow_angle",
    "right_elbow_angle",
    "right_wrist_shoulder_dist",
    "left_head_elbow_dist",
    "right_head_wrist_dist",
    "right_hip_elbow_dist",
    "left_wrist_shoulder_dist",
    "left_elbow_height_rel_shoulder",
)


@pytest.fixture(scope="session")
def selected_features() -> tuple[str, ...]:
    return SELECTED_FEATURES


@pytest.fixture(scope="session")
def reference_frames() -> list[FrameLandmarks]:
    """619 contiguous frames of a real climb — the primary parity fixture."""
    return read_landmark_csv(LANDMARKS / "reference_62_B.csv")


@pytest.fixture(scope="session")
def hand_computed() -> dict[str, Any]:
    """Feature values worked out independently of both the research code and this package."""
    with (REFERENCE / "hand_computed_features.json").open() as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def load_edge_case(name: str) -> list[FrameLandmarks]:
    return read_landmark_csv(LANDMARKS / f"edge_{name}.csv")


def reference_file(name: str) -> Path:
    return REFERENCE / name
