"""Turning a fault's attribution into words a climber can act on.

Every feature the shipped bundles can select maps to a body part and a pair of phrases —
one for when the fault's value sits below the sound baseline, one for above — so the
rendered sentence never has to fall back to naming the feature itself. `render_explanation`
is the only place permitted to know both the templates and the model's feature vocabulary;
nothing it returns may contain a feature identifier or a statistical term, which
`tests/unit/test_report.py` checks against every entry in `FEATURE_NAMES` plus a list of
words like "z-score" and "deviation".
"""

from __future__ import annotations

from dataclasses import dataclass

from aced.report.types import Attribution, Fault


@dataclass(frozen=True, slots=True)
class FeatureTemplate:
    body_part: str
    below_baseline: str
    above_baseline: str


#: Keyed by every name `aced.features.compute.FEATURE_NAMES` can produce, not just the eight
#: the shipped bundles select — a future bundle choosing a different subset needs no new
#: language module.
FEATURE_TEMPLATES: dict[str, FeatureTemplate] = {
    "left_elbow_angle": FeatureTemplate(
        "left arm", "held more bent than usual", "held straighter than usual"
    ),
    "right_elbow_angle": FeatureTemplate(
        "right arm", "held more bent than usual", "held straighter than usual"
    ),
    "left_shoulder_angle": FeatureTemplate(
        "left arm", "drawn in closer to the body than usual", "reached out farther than usual"
    ),
    "right_shoulder_angle": FeatureTemplate(
        "right arm", "drawn in closer to the body than usual", "reached out farther than usual"
    ),
    "left_wrist_height_rel_shoulder": FeatureTemplate(
        "left hand", "raised higher above the shoulder than usual", "dropped lower than usual"
    ),
    "right_wrist_height_rel_shoulder": FeatureTemplate(
        "right hand", "raised higher above the shoulder than usual", "dropped lower than usual"
    ),
    "left_elbow_height_rel_shoulder": FeatureTemplate(
        "left elbow", "raised higher above the shoulder than usual", "dropped lower than usual"
    ),
    "right_elbow_height_rel_shoulder": FeatureTemplate(
        "right elbow", "raised higher above the shoulder than usual", "dropped lower than usual"
    ),
    "left_wrist_shoulder_dist": FeatureTemplate(
        "left arm", "held closer to the body than usual", "reaching farther away than usual"
    ),
    "right_wrist_shoulder_dist": FeatureTemplate(
        "right arm", "held closer to the body than usual", "reaching farther away than usual"
    ),
    "left_head_elbow_dist": FeatureTemplate(
        "left elbow", "held closer to the head than usual", "held farther from the head than usual"
    ),
    "right_head_elbow_dist": FeatureTemplate(
        "right elbow", "held closer to the head than usual", "held farther from the head than usual"
    ),
    "left_head_wrist_dist": FeatureTemplate(
        "left hand", "held closer to the head than usual", "held farther from the head than usual"
    ),
    "right_head_wrist_dist": FeatureTemplate(
        "right hand", "held closer to the head than usual", "held farther from the head than usual"
    ),
    "left_hip_elbow_dist": FeatureTemplate(
        "left elbow", "tucked closer to the hip than usual", "held farther from the hip than usual"
    ),
    "right_hip_elbow_dist": FeatureTemplate(
        "right elbow", "tucked closer to the hip than usual", "held farther from the hip than usual"
    ),
    "left_hip_wrist_dist": FeatureTemplate(
        "left hand", "held closer to the hip than usual", "held farther from the hip than usual"
    ),
    "right_hip_wrist_dist": FeatureTemplate(
        "right hand", "held closer to the hip than usual", "held farther from the hip than usual"
    ),
}


def format_timestamp(seconds: float) -> str:
    """`M:SS`, rounded to the nearest second — precise enough to find the moment in a video."""
    total = round(max(seconds, 0.0))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _time_range(start: float, end: float) -> str:
    return f"{format_timestamp(start)} and {format_timestamp(end)}"


def render_explanation(
    fault_time_start: float, fault_time_end: float, attribution: Attribution
) -> str:
    """Plain-language copy for one fault. Never names a feature or a statistic."""
    when = _time_range(fault_time_start, fault_time_end)

    if not attribution.reliable or attribution.dominant is None:
        return (
            f"Unusual arm technique was detected between {when}, but there wasn't enough "
            "sound technique elsewhere in this climb to compare it against."
        )

    dominant = attribution.dominant
    if dominant.deviation == 0.0:
        # Every feature matched the sound baseline exactly — a fault was still flagged (the
        # model saw something in the full window the aggregated, per-fault feature mean
        # doesn't capture), but naming a "dominant" feature and picking a direction for it
        # would claim a specific change that the numbers don't show.
        return (
            f"Unusual arm technique was flagged between {when}, but it stayed close to the "
            "climber's usual technique — nothing stood out as the cause."
        )

    template = FEATURE_TEMPLATES.get(dominant.feature)
    if template is None:
        # A bundle feature with no language mapping yet — say something rather than crash,
        # but keep it free of the identifier itself.
        return f"Unusual arm technique was detected between {when}."

    phrase = template.below_baseline if dominant.deviation < 0 else template.above_baseline
    return f"Your {template.body_part} was {phrase} between {when}."


def attach_explanation(
    *,
    frame_start: int,
    frame_end: int,
    time_start: float,
    time_end: float,
    confidence: float | None,
    attribution: Attribution,
) -> Fault:
    """Build the `Fault` a report carries, explanation included."""
    return Fault(
        frame_start=frame_start,
        frame_end=frame_end,
        time_start=time_start,
        time_end=time_end,
        confidence=confidence,
        attribution=attribution,
        explanation=render_explanation(time_start, time_end, attribution),
    )
