"""Record expected feature values computed independently of the research code.

Task 2.4. Parity fixtures prove the port agrees with the thesis pipeline; they cannot prove
the thesis pipeline was right. These fixtures pin the feature definitions themselves:

  synthetic   frames built from coordinates chosen so every feature has an exact value that
              can be read off by hand — a right angle is 90, a straight arm is 180, a
              unit-square diagonal is sqrt(2)
  degenerate  coincident landmarks, where the angle is undefined and the contract says zero
  reference   two real frames from the reference landmark CSV, computed here with plain
              scalar arithmetic rather than the ported implementation

Nothing in this file imports from `aced` — that is the point. Feature semantics, restated
from the research code:

    angle(a, b, c)  angle at vertex b between ba and bc, in degrees, 0–180
    offset(a, b)    y_b - y_a, signed; image y grows downward, so positive means b is lower
    distance(a, b)  euclidean distance in the 2D (x, y) plane

Run:
    python scripts/make_feature_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LANDMARKS = ROOT / "fixtures" / "landmarks" / "reference_62_B.csv"
OUT = ROOT / "fixtures" / "reference" / "hand_computed_features.json"

Point = tuple[float, float]

# MediaPipe landmark indices for the nine joints the models use.
LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


def angle(pose: dict[str, Point], a: str, b: str, c: str) -> float:
    (ax, ay), (bx, by), (cx, cy) = pose[a], pose[b], pose[c]
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    ba_mag = math.sqrt(bax * bax + bay * bay)
    bc_mag = math.sqrt(bcx * bcx + bcy * bcy)
    if ba_mag == 0.0 or bc_mag == 0.0:
        return 0.0  # undefined; contract says zero
    cosine = (bax * bcx + bay * bcy) / (ba_mag * bc_mag)
    if not -1.0 <= cosine <= 1.0:
        # Only reachable through floating-point overshoot at exact collinearity.
        cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def offset(pose: dict[str, Point], a: str, b: str) -> float:
    return pose[b][1] - pose[a][1]


def distance(pose: dict[str, Point], a: str, b: str) -> float:
    (ax, ay), (bx, by) = pose[a], pose[b]
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def compute(pose: dict[str, Point]) -> dict[str, float]:
    """All eighteen features, in the order the research code emits them."""
    return {
        "left_elbow_angle": angle(pose, "left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow_angle": angle(pose, "right_shoulder", "right_elbow", "right_wrist"),
        "left_shoulder_angle": angle(pose, "left_hip", "left_shoulder", "left_elbow"),
        "right_shoulder_angle": angle(pose, "right_hip", "right_shoulder", "right_elbow"),
        "left_wrist_height_rel_shoulder": offset(pose, "left_shoulder", "left_wrist"),
        "right_wrist_height_rel_shoulder": offset(pose, "right_shoulder", "right_wrist"),
        "left_elbow_height_rel_shoulder": offset(pose, "left_shoulder", "left_elbow"),
        "right_elbow_height_rel_shoulder": offset(pose, "right_shoulder", "right_elbow"),
        "left_wrist_shoulder_dist": distance(pose, "left_wrist", "left_shoulder"),
        "right_wrist_shoulder_dist": distance(pose, "right_wrist", "right_shoulder"),
        "left_head_elbow_dist": distance(pose, "nose", "left_elbow"),
        "right_head_elbow_dist": distance(pose, "nose", "right_elbow"),
        "left_head_wrist_dist": distance(pose, "nose", "left_wrist"),
        "right_head_wrist_dist": distance(pose, "nose", "right_wrist"),
        "left_hip_elbow_dist": distance(pose, "left_hip", "left_elbow"),
        "right_hip_elbow_dist": distance(pose, "right_hip", "right_elbow"),
        "left_hip_wrist_dist": distance(pose, "left_hip", "left_wrist"),
        "right_hip_wrist_dist": distance(pose, "right_hip", "right_wrist"),
    }


# A pose whose every feature is exact. Coordinates are in the same normalized space
# MediaPipe emits, with y increasing downward.
#
#   left arm  : shoulder (0.0, 0.0) → elbow (0.0, 0.1) → wrist (0.1, 0.1)
#               ba = (0, -0.1), bc = (0.1, 0) → perpendicular → 90 degrees
#   right arm : shoulder (0.2, 0.0) → elbow (0.2, 0.1) → wrist (0.2, 0.2)
#               ba and bc are antiparallel → 180 degrees, a fully extended arm
#   nose      : (0.1, -0.1), one tenth above the shoulder line
#   hips      : directly below each shoulder, so each shoulder angle is the angle between
#               straight down and straight down — 0 degrees on the left, 0 on the right
RIGHT_ANGLE_POSE: dict[str, Point] = {
    "nose": (0.1, -0.1),
    "left_shoulder": (0.0, 0.0),
    "right_shoulder": (0.2, 0.0),
    "left_elbow": (0.0, 0.1),
    "right_elbow": (0.2, 0.1),
    "left_wrist": (0.1, 0.1),
    "right_wrist": (0.2, 0.2),
    "left_hip": (0.0, 0.3),
    "right_hip": (0.2, 0.3),
}

# Every joint on one point: distances collapse to zero and both angle arguments are
# zero-length, so the angles are undefined and must be emitted as zero, not NaN.
COINCIDENT_POSE: dict[str, Point] = {name: (0.5, 0.5) for name in LANDMARK_INDEX}

SYNTHETIC = [
    {
        "name": "right_angle_and_straight_arm",
        "pose": RIGHT_ANGLE_POSE,
        "rationale": (
            "Left elbow is a right angle (90) and the right arm is fully extended (180). "
            "Left wrist sits 0.1 below and 0.1 across from the shoulder, so the "
            "wrist-shoulder distance is sqrt(0.02) ~= 0.1414. Hips are directly below "
            "shoulders and elbows are too, so both shoulder angles are 0."
        ),
        "spot_checks": {
            "left_elbow_angle": 90.0,
            "right_elbow_angle": 180.0,
            "left_shoulder_angle": 0.0,
            "right_shoulder_angle": 0.0,
            "left_wrist_height_rel_shoulder": 0.1,
            "right_elbow_height_rel_shoulder": 0.1,
            "left_wrist_shoulder_dist": math.sqrt(0.02),
            "right_wrist_shoulder_dist": 0.2,
        },
    },
    {
        "name": "coincident_landmarks",
        "pose": COINCIDENT_POSE,
        "rationale": (
            "All landmarks share a position. Distances and offsets are exactly zero and the "
            "angles are undefined; the contract requires zero rather than NaN."
        ),
        "spot_checks": {name: 0.0 for name in compute(COINCIDENT_POSE)},
    },
]

# Frames picked from the reference clip: the first frame, and one from the middle where the
# climber is mid-move rather than static on the start holds.
REFERENCE_FRAMES = [1, 300]


def pose_from_row(row: "pd.Series[Any]") -> dict[str, Point]:
    return {
        name: (float(row[f"x_{index}"]), float(row[f"y_{index}"]))
        for name, index in LANDMARK_INDEX.items()
    }


def main() -> int:
    payload: dict[str, Any] = {
        "description": (
            "Expected feature values computed independently of both the research code and the "
            "aced package. Generated by scripts/make_feature_fixtures.py."
        ),
        "feature_order": list(compute(RIGHT_ANGLE_POSE)),
        "synthetic": [],
        "reference_frames": [],
    }

    for case in SYNTHETIC:
        pose: dict[str, Point] = case["pose"]  # type: ignore[assignment]
        computed = compute(pose)
        for feature, expected in case["spot_checks"].items():  # type: ignore[union-attr]
            assert math.isclose(computed[feature], expected, abs_tol=1e-12), (
                f"{case['name']}: {feature} computed {computed[feature]}, hand value {expected}"
            )
        payload["synthetic"].append(
            {
                "name": case["name"],
                "rationale": case["rationale"],
                "landmarks": {name: list(point) for name, point in pose.items()},
                "expected": computed,
            }
        )

    frame_data = pd.read_csv(LANDMARKS)
    for frame_number in REFERENCE_FRAMES:
        matches = frame_data[frame_data["frame"] == frame_number]
        if matches.empty:
            raise SystemExit(f"frame {frame_number} not present in {LANDMARKS.name}")
        row = matches.iloc[0]
        payload["reference_frames"].append(
            {
                "source": LANDMARKS.name,
                "frame": int(frame_number),
                "landmarks": {name: list(point) for name, point in pose_from_row(row).items()},
                "expected": compute(pose_from_row(row)),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(payload['synthetic'])} synthetic cases, {len(payload['reference_frames'])} reference frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
