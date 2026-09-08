"""What every report says about what the model does and does not cover.

Fixed copy, not derived from any single analysis — every report carries the same statement
of scope and the same limitations regardless of what it found, per the analysis-report
spec's "Model limitations disclosed" requirement.
"""

from __future__ import annotations

SCOPE_STATEMENT = (
    "This analysis covers arm technique only — principally arms held bent where a straight "
    "arm would save forearm strength. It does not judge footwork, hip position, or route "
    "reading."
)

LIMITATIONS: tuple[str, ...] = (
    "Only one climber is analyzed; footage containing several people in frame is not supported.",
    "The analysis works from 2D pose extracted from a single camera angle, which is "
    "sensitive to that angle — an arm bent toward the wall can project as straight.",
    "The model was trained on a small set of climbers in one gym, labelled by one person, "
    "so results are best read as observations rather than authoritative coaching.",
)
