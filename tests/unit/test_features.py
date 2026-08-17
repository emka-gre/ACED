"""Feature computation: definitions, degenerate geometry, and bundle-driven selection."""

from __future__ import annotations

import math
from typing import Any

import pytest

from aced.features.compute import (
    FEATURE_NAMES,
    FeatureExtractor,
    UnknownFeatureError,
    compute_features,
    joint_angle,
    vertical_offset,
)
from aced.pose.landmarks import FrameLandmarks, Joint, Landmark


def pose(points: dict[str, tuple[float, float]], visibility: float = 1.0) -> FrameLandmarks:
    joints = {
        Joint(name): Landmark(x=x, y=y, z=0.0, visibility=visibility)
        for name, (x, y) in points.items()
    }
    return FrameLandmarks(index=0, timestamp=0.0, joints=joints)


def test_emits_all_eighteen_features_in_canonical_order() -> None:
    frame = pose({joint.value: (0.5, 0.5) for joint in Joint})
    assert tuple(compute_features(frame)) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 18


def test_right_angle_and_straight_arm() -> None:
    # A right angle at the elbow, and an arm extended straight down.
    assert joint_angle((0.0, 0.0), (0.0, 0.1), (0.1, 0.1)) == pytest.approx(90.0)
    assert joint_angle((0.2, 0.0), (0.2, 0.1), (0.2, 0.2)) == pytest.approx(180.0)


def test_angles_stay_within_zero_to_one_eighty(reference_frames: list[FrameLandmarks]) -> None:
    for frame in reference_frames[:100]:
        values = compute_features(frame)
        for name in ("left_elbow_angle", "right_elbow_angle", "left_shoulder_angle"):
            assert 0.0 <= values[name] <= 180.0


def test_vertical_offset_sign_convention() -> None:
    # Image y grows downward, so a wrist below the shoulder is positive. The models were
    # trained on this convention; flipping it would invert a third of the feature space.
    assert vertical_offset((0.0, 0.2), (0.0, 0.5)) == pytest.approx(0.3)
    assert vertical_offset((0.0, 0.5), (0.0, 0.2)) == pytest.approx(-0.3)


def test_coincident_landmarks_give_zero_not_nan() -> None:
    frame = pose({joint.value: (0.5, 0.5) for joint in Joint})
    values = compute_features(frame)
    assert all(value == 0.0 for value in values.values())
    assert all(math.isfinite(value) for value in values.values())


def test_zero_length_ray_gives_zero_angle() -> None:
    # Wrist exactly on the elbow: the angle at the elbow is undefined, not NaN.
    assert joint_angle((0.0, 0.0), (0.1, 0.1), (0.1, 0.1)) == 0.0


def test_collinear_points_do_not_overshoot_arccos_domain() -> None:
    # Floating point can push cos(theta) a hair past -1 for exactly collinear points.
    angle = joint_angle((0.1, 0.30000000000000004), (0.1, 0.2), (0.1, 0.1))
    assert math.isfinite(angle)
    assert angle == pytest.approx(180.0)


def test_all_features_finite_across_the_reference_clip(
    reference_frames: list[FrameLandmarks],
) -> None:
    for frame in reference_frames:
        assert all(math.isfinite(value) for value in compute_features(frame).values())


class TestSelection:
    def test_selects_in_the_declared_order(self, selected_features: tuple[str, ...]) -> None:
        extractor = FeatureExtractor(selected_features)
        frame = pose({joint.value: (0.5, 0.5) for joint in Joint})
        assert tuple(extractor.named(frame)) == selected_features
        assert len(extractor.from_frame(frame)) == 8

    def test_declared_order_is_respected_even_when_unusual(self) -> None:
        reversed_selection = tuple(reversed(FEATURE_NAMES))
        extractor = FeatureExtractor(reversed_selection)
        frame = pose({joint.value: (0.5, 0.5) for joint in Joint})
        assert tuple(extractor.named(frame)) == reversed_selection

    def test_unknown_feature_fails_at_construction(self) -> None:
        # A bundle asking for something we do not compute must fail loudly rather than
        # receiving a zero-filled column, which is how the research code hid the problem.
        with pytest.raises(UnknownFeatureError, match="left_knee_angle"):
            FeatureExtractor(["left_elbow_angle", "left_knee_angle"])


class TestHandComputedValues:
    """Pins the definitions against values derived without either implementation."""

    def test_synthetic_cases(self, hand_computed: dict[str, Any]) -> None:
        for case in hand_computed["synthetic"]:
            frame = pose({name: (point[0], point[1]) for name, point in case["landmarks"].items()})
            computed = compute_features(frame)
            for name, expected in case["expected"].items():
                assert computed[name] == pytest.approx(expected, abs=1e-12), (
                    f"{case['name']}: {name}"
                )

    def test_real_reference_frames(self, hand_computed: dict[str, Any]) -> None:
        for case in hand_computed["reference_frames"]:
            frame = pose({name: (point[0], point[1]) for name, point in case["landmarks"].items()})
            computed = compute_features(frame)
            for name, expected in case["expected"].items():
                assert computed[name] == pytest.approx(expected, rel=1e-12), (
                    f"frame {case['frame']}: {name}"
                )
