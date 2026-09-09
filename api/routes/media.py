"""Serving the original video and the rendered overlay, seekably.

Byte ranges are the whole point of this module. A video element asked to seek issues a range
request, and a server that answers 200 with the whole file every time makes seeking in a
200 MB climb unusable.
"""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse, StreamingResponse

from api.context import ServiceContext, get_context
from api.models import ArtifactState
from api.routes.analyses import require_record
from api.services.storage import ObjectNotFoundError, iter_range

router = APIRouter(prefix="/analyses", tags=["media"])

RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


class RangeNotSatisfiableError(ValueError):
    """The range asked for lies outside the object."""


def parse_range(header: str, size: int) -> tuple[int, int]:
    """One `bytes=` range, resolved to inclusive absolute offsets.

    Only a single range is supported. Multipart range responses exist, no video element asks
    for one, and answering a multi-range request with the whole file is a correct fallback —
    so an unparseable header is treated as no header rather than as an error.
    """
    match = RANGE_PATTERN.match(header.strip())
    if match is None:
        raise ValueError(header)

    start_text, end_text = match.groups()
    if not start_text and not end_text:
        raise ValueError(header)

    if not start_text:
        # `bytes=-500`: the final 500 bytes.
        length = int(end_text)
        if length <= 0:
            raise RangeNotSatisfiableError(header)
        start = max(size - length, 0)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1

    if start >= size or start > end:
        raise RangeNotSatisfiableError(header)
    return start, min(end, size - 1)


def content_type_for(key: str) -> str:
    for suffix, media_type in CONTENT_TYPES.items():
        if key.endswith(suffix):
            return media_type
    return "application/octet-stream"


def serve_object(context: ServiceContext, key: str, range_header: str | None) -> StreamingResponse:
    try:
        size = context.storage.size(key)
    except ObjectNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "the media is no longer stored") from None

    media_type = content_type_for(key)
    if range_header:
        try:
            start, end = parse_range(range_header, size)
        except RangeNotSatisfiableError:
            raise HTTPException(
                status.HTTP_416_RANGE_NOT_SATISFIABLE,
                "the requested range is outside the file",
                headers={"Content-Range": f"bytes */{size}"},
            ) from None
        except ValueError:
            # Unparseable or multi-range: answering with the whole file is a correct
            # response to a range request, and no video element asks for a multi-range.
            pass
        else:
            return StreamingResponse(
                iter_range(context.storage.open(key), start, end),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                media_type=media_type,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Content-Length": str(end - start + 1),
                    "Accept-Ranges": "bytes",
                },
            )

    return StreamingResponse(
        iter_range(context.storage.open(key), 0, size - 1),
        media_type=media_type,
        headers={"Content-Length": str(size), "Accept-Ranges": "bytes"},
    )


@router.get("/{analysis_id}/video")
def get_video(
    analysis_id: str,
    context: Annotated[ServiceContext, Depends(get_context)],
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> Response:
    record = require_record(context, analysis_id)
    return serve_object(context, record.video_key, range_header)


@router.get("/{analysis_id}/overlay")
def get_overlay(
    analysis_id: str,
    context: Annotated[ServiceContext, Depends(get_context)],
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> Response:
    """Serve the overlay, or start rendering it and say so.

    Rendering is never done during analysis, so the first request for an overlay is the one
    that causes it to exist. The client is told it is being prepared and polls, exactly as it
    already polls the analysis itself.
    """
    record = require_record(context, analysis_id)

    if record.overlay is ArtifactState.READY and record.overlay_key is not None:
        return serve_object(context, record.overlay_key, range_header)

    if record.overlay is ArtifactState.FAILED:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "the overlay could not be rendered for this analysis"
        )

    if record.overlay is ArtifactState.ABSENT:
        if record.report is None:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"no overlay can be rendered yet; the analysis is {record.state.value}",
            )
        context.store.update(
            analysis_id,
            lambda current: current.with_changes(overlay=ArtifactState.PREPARING),
        )
        renderer = context.renderer
        context.queue.submit(lambda: renderer.run(analysis_id))

    # Not an error: 202 with a body saying so, because the client's next move is to wait
    # and ask again, exactly as it already does for the analysis itself.
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "state": ArtifactState.PREPARING.value,
            "message": "the overlay is being prepared",
        },
    )
