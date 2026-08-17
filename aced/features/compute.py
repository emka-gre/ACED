"""The eighteen engineered features, ported from the research code.

These definitions are load-bearing: the shipped models were fitted against these exact
values under these exact names, so the arithmetic and the spelling are both frozen. The
research code's misspelled method names (`compute_poition`, `compute_ditance`) are corrected
here, but every emitted feature name is byte-identical to what the models expect — including
`left_wrist_height_rel_shoulder` and the rest of the original vocabulary.

Everything is computed from 2D `(x, y)` only. The method is 2D and inherits the camera-angle
sensitivity that comes with it; `z` is deliberately unused.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from aced.pose.landmarks import FrameLandmarks, Joint

#: Feature names in the order the research code emits them. The order matters: it is the
#: order the flattened `<feature>_t<index>` columns were built in, and therefore the order
#: the scaler and the flat-input models were fitted against.
FEATURE_NAMES: tuple[str, ...] = (
    # angles
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    # signed vertical offsets relative to the corresponding shoulder
    "left_wrist_height_rel_shoulder",
    "right_wrist_height_rel_shoulder",
    "left_elbow_height_rel_shoulder",
    "right_elbow_height_rel_shoulder",
    # inter-joint distances
    "left_wrist_shoulder_dist",
    "right_wrist_shoulder_dist",
    "left_head_elbow_dist",
    "right_head_elbow_dist",
    "left_head_wrist_dist",
    "right_head_wrist_dist",
    "left_hip_elbow_dist",
    "right_hip_elbow_dist",
    "left_hip_wrist_dist",
    "right_hip_wrist_dist",
)

Point = tuple[float, float]


def joint_angle(a: Point, vertex: Point, c: Point) -> float:
    """Angle at `vertex` between the rays to `a` and `c`, in degrees.

    Returns zero when either ray has zero length, which makes the angle undefined. The
    research code produced NaN there and replaced it with zero one step later; doing it here
    keeps every feature finite at the point of computation.
    """
    ax, ay = a[0] - vertex[0], a[1] - vertex[1]
    cx, cy = c[0] - vertex[0], c[1] - vertex[1]
    a_mag = math.sqrt(ax * ax + ay * ay)
    c_mag = math.sqrt(cx * cx + cy * cy)
    if a_mag == 0.0 or c_mag == 0.0:
        return 0.0

    cosine = (ax * cx + ay * cy) / (a_mag * c_mag)
    # Exact collinearity can overshoot the arccos domain by an ulp or two.
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def vertical_offset(reference: Point, target: Point) -> float:
    """Signed vertical difference `target - reference`.

    Image `y` grows downward, so a positive value means the target sits *below* the
    reference. The research code's `compute_poition(dataset, a, b)` returns `y_b - y_a`; the
    argument order is preserved by callers below so the sign convention the models were
    trained on is unchanged.
    """
    return target[1] - reference[1]


def euclidean_distance(a: Point, b: Point) -> float:
    """Distance between two joints in the 2D image plane."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _finite(value: float) -> float:
    """Non-finite results become zero, matching the research pipeline's `fillna(0.0)`."""
    return value if math.isfinite(value) else 0.0


def compute_features(frame: FrameLandmarks) -> dict[str, float]:
    """All eighteen features for one frame, keyed by name in canonical order."""
    nose = frame.point(Joint.NOSE)
    l_shoulder = frame.point(Joint.LEFT_SHOULDER)
    r_shoulder = frame.point(Joint.RIGHT_SHOULDER)
    l_elbow = frame.point(Joint.LEFT_ELBOW)
    r_elbow = frame.point(Joint.RIGHT_ELBOW)
    l_wrist = frame.point(Joint.LEFT_WRIST)
    r_wrist = frame.point(Joint.RIGHT_WRIST)
    l_hip = frame.point(Joint.LEFT_HIP)
    r_hip = frame.point(Joint.RIGHT_HIP)

    values = {
        "left_elbow_angle": joint_angle(l_shoulder, l_elbow, l_wrist),
        "right_elbow_angle": joint_angle(r_shoulder, r_elbow, r_wrist),
        "left_shoulder_angle": joint_angle(l_hip, l_shoulder, l_elbow),
        "right_shoulder_angle": joint_angle(r_hip, r_shoulder, r_elbow),
        "left_wrist_height_rel_shoulder": vertical_offset(l_shoulder, l_wrist),
        "right_wrist_height_rel_shoulder": vertical_offset(r_shoulder, r_wrist),
        "left_elbow_height_rel_shoulder": vertical_offset(l_shoulder, l_elbow),
        "right_elbow_height_rel_shoulder": vertical_offset(r_shoulder, r_elbow),
        "left_wrist_shoulder_dist": euclidean_distance(l_wrist, l_shoulder),
        "right_wrist_shoulder_dist": euclidean_distance(r_wrist, r_shoulder),
        "left_head_elbow_dist": euclidean_distance(nose, l_elbow),
        "right_head_elbow_dist": euclidean_distance(nose, r_elbow),
        "left_head_wrist_dist": euclidean_distance(nose, l_wrist),
        "right_head_wrist_dist": euclidean_distance(nose, r_wrist),
        "left_hip_elbow_dist": euclidean_distance(l_hip, l_elbow),
        "right_hip_elbow_dist": euclidean_distance(r_hip, r_elbow),
        "left_hip_wrist_dist": euclidean_distance(l_hip, l_wrist),
        "right_hip_wrist_dist": euclidean_distance(r_hip, r_wrist),
    }
    return {name: _finite(value) for name, value in values.items()}


class UnknownFeatureError(KeyError):
    """A model bundle asked for a feature this extractor does not compute."""


class FeatureExtractor:
    """Turns one frame's landmarks into the ordered feature vector a bundle asked for.

    Constructed with the bundle's declared feature list, so the vector it produces is in the
    bundle's order by construction rather than by a downstream reindex. A bundle naming a
    feature that does not exist fails here, at construction, rather than silently receiving
    a zero-filled column the way the research code's `reindex(..., fill_value=0.0)` did.
    """

    def __init__(self, selected: Sequence[str] = FEATURE_NAMES) -> None:
        unknown = [name for name in selected if name not in FEATURE_NAMES]
        if unknown:
            raise UnknownFeatureError(
                f"features not computed by this extractor: {', '.join(unknown)}. "
                f"Available: {', '.join(FEATURE_NAMES)}"
            )
        self.selected: tuple[str, ...] = tuple(selected)

    def from_frame(self, frame: FrameLandmarks) -> list[float]:
        """The selected features for one frame, in the bundle's declared order."""
        computed = compute_features(frame)
        return [computed[name] for name in self.selected]

    def named(self, frame: FrameLandmarks) -> dict[str, float]:
        """The selected features keyed by name, for reporting and debugging."""
        computed = compute_features(frame)
        return {name: computed[name] for name in self.selected}
