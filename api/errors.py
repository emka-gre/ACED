"""What a client is told when an analysis fails, and what only the log sees.

The split is deliberate. A stack trace from Keras or OpenCV can name filesystem paths, model
internals, and library versions; none of that helps the climber who uploaded a video, and
some of it is information a public service should not volunteer. The client gets a sentence
naming what it can do about it; the log gets everything.
"""

from __future__ import annotations

from aced.inference.bundle import BundleError
from aced.pose.extractor import UnreadableVideoError

UNREADABLE_VIDEO = (
    "The video could not be read. It may be corrupt, truncated, or in a codec this "
    "service does not support."
)
MODEL_UNAVAILABLE = (
    "The analysis model is unavailable. This is a server-side problem, not a problem "
    "with your video."
)
UNEXPECTED = "The analysis failed unexpectedly and the failure has been logged."


def client_safe_reason(error: BaseException) -> str:
    """One sentence a client can act on, for any failure the pipeline can produce.

    The default is deliberately vague. A failure nobody anticipated is exactly the case where
    the exception text is most likely to carry something internal, so unrecognised errors are
    summarised rather than paraphrased.
    """
    if isinstance(error, UnreadableVideoError):
        return UNREADABLE_VIDEO
    if isinstance(error, BundleError):
        return MODEL_UNAVAILABLE
    return UNEXPECTED
