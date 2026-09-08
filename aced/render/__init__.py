"""Rendered artifacts of an analysis.

Everything here is on-demand. An analysis never renders; a caller that wants an overlay asks
for one after the fact, with the landmarks and frame classes the analysis already produced.
"""

from __future__ import annotations

from aced.render.overlay import (
    DEFAULT_STYLE,
    SKELETON_EDGES,
    OpenCvVideoSink,
    OverlayRenderError,
    OverlayResult,
    OverlayStyle,
    VideoSink,
    draw_skeleton,
    render_overlay,
    to_pixel,
)

__all__ = [
    "DEFAULT_STYLE",
    "SKELETON_EDGES",
    "OpenCvVideoSink",
    "OverlayRenderError",
    "OverlayResult",
    "OverlayStyle",
    "VideoSink",
    "draw_skeleton",
    "render_overlay",
    "to_pixel",
]
