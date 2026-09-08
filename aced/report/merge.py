"""Merging consecutive or overlapping faulty windows into distinct faults.

Windows overlap by construction — length 60, stride 20 — so a run of faulty windows almost
always shares frames with its neighbours. Two faulty windows merge when they are consecutive
in frame order *and* their frame ranges touch or overlap; a sound-classified window between
them breaks the chain. Both halves of that rule matter: frame adjacency alone would keep
temporally distant faults distinct but would also swallow a corrected stretch, because with
this much overlap a fault on either side of a sound window still touches frames across it.

A consequence of breaking on the sound window rather than on frame distance: two spans either
side of one sound window can overlap in frames, since each span covers the full extent of its
own windows. That is faithful to the classifications — the report's `segments` show the same
sound window — and is preferred over merging a stretch the model called sound into a fault.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from aced.inference.pipeline import PredictedClass, WindowResult


@dataclass(frozen=True, slots=True)
class FaultSpan:
    """One merged group of faulty windows, before attribution or explanation are attached."""

    frame_start: int
    frame_end: int
    confidences: tuple[float, ...]
    window_indices: tuple[int, ...]


def merge_fault_spans(results: Sequence[WindowResult]) -> list[FaultSpan]:
    """Group faulty windows into spans, ordered by frame position.

    `results` need not arrive sorted; they are ordered by `frame_start` here so windows from
    out-of-order input still merge correctly.

    A sound-classified window breaks the chain unconditionally, even when the fault window
    before it and the fault window after it still overlap in frame range — which they can:
    at length 60 and stride 20, two fault windows two positions apart (one sound window
    between them) still share 20 frames. Grouping only faulty windows by frame adjacency
    would silently fold the model's sound verdict into a single reported fault; walking the
    full sequence and resetting on sound is what actually keeps them distinct.
    """
    ordered = sorted(results, key=lambda result: result.frame_start)

    spans: list[FaultSpan] = []
    open_span: FaultSpan | None = None
    for result in ordered:
        if result.predicted is not PredictedClass.FAULT:
            if open_span is not None:
                spans.append(open_span)
            open_span = None
            continue

        if open_span is not None and result.frame_start <= open_span.frame_end + 1:
            open_span = FaultSpan(
                frame_start=open_span.frame_start,
                frame_end=max(open_span.frame_end, result.frame_end),
                confidences=(
                    *open_span.confidences,
                    *((result.confidence,) if result.confidence is not None else ()),
                ),
                window_indices=(*open_span.window_indices, result.index),
            )
        else:
            if open_span is not None:
                spans.append(open_span)
            open_span = FaultSpan(
                frame_start=result.frame_start,
                frame_end=result.frame_end,
                confidences=(result.confidence,) if result.confidence is not None else (),
                window_indices=(result.index,),
            )

    if open_span is not None:
        spans.append(open_span)

    return spans


def aggregate_confidence(span: FaultSpan) -> float | None:
    """Mean confidence of the span's constituent windows, or `None` if the backend gives none."""
    if not span.confidences:
        return None
    return sum(span.confidences) / len(span.confidences)
