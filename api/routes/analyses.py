"""Uploading a climb, following its analysis, and deleting it.

Upload is the one handler with real work in it: the file is streamed to storage a chunk at a
time and the size limit is enforced as it goes, so an oversized upload is refused part-way
rather than after the service has already received and buffered all of it.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from api.context import ServiceContext, get_context
from api.logging import get_logger
from api.models import AnalysisRecord
from api.services.records import RecordNotFoundError
from api.services.storage import CHUNK_BYTES, UploadTooLargeError

router = APIRouter(prefix="/analyses", tags=["analyses"])


def require_record(context: ServiceContext, analysis_id: str) -> AnalysisRecord:
    try:
        return context.store.get(analysis_id)
    except RecordNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such analysis") from None


def _validate_format(context: ServiceContext, upload: UploadFile) -> str:
    """Return the accepted suffix, or refuse.

    Extension *and* declared content type are both checked, because either alone is trivially
    wrong: an extension is a rename away from a lie, and browsers routinely send
    `application/octet-stream` for a file they decline to identify.
    """
    settings = context.settings
    suffix = PurePath(upload.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported video format {suffix or '(none)'}; "
            f"supported formats are {settings.describes_supported_formats()}",
        )
    if upload.content_type and upload.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported content type {upload.content_type}; "
            f"supported formats are {settings.describes_supported_formats()}",
        )
    return suffix


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_analysis(
    response: Response,
    context: Annotated[ServiceContext, Depends(get_context)],
    file: Annotated[UploadFile, File(description="the climbing video to analyze")],
) -> dict[str, Any]:
    """Accept a video, store it, and enqueue the analysis. Returns before any of it runs."""
    if not context.healthy:
        # Accepting an upload the service cannot process would mean storing someone's video
        # only to fail it — worse than refusing it now.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "the analysis model is unavailable; no analyses are being accepted",
        )

    suffix = _validate_format(context, file)
    settings = context.settings
    writer = context.storage.writer(suffix=suffix, max_bytes=settings.max_upload_bytes)

    try:
        while chunk := await file.read(CHUNK_BYTES):
            writer.write(chunk)
        stored = writer.commit()
    except UploadTooLargeError:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"the video exceeds the {settings.max_upload_mb} MB limit",
        ) from None
    except BaseException:
        writer.abort()
        raise

    record = AnalysisRecord.create(
        filename=file.filename or f"upload{suffix}",
        video_key=stored.key,
        size_bytes=stored.size_bytes,
        retention=settings.retention,
    )
    context.store.add(record)

    worker = context.worker
    context.queue.submit(lambda: worker.run(record.id))

    get_logger().info(
        "analysis accepted",
        extra={"analysis_id": record.id, "size_bytes": record.size_bytes},
    )
    location = f"/analyses/{record.id}"
    response.headers["Location"] = location
    return {"id": record.id, "state": record.state.value, "location": location}


@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: str, context: Annotated[ServiceContext, Depends(get_context)]
) -> dict[str, Any]:
    return require_record(context, analysis_id).to_status()


@router.get("/{analysis_id}/report")
def get_report(
    analysis_id: str, context: Annotated[ServiceContext, Depends(get_context)]
) -> dict[str, Any]:
    """The full report, and only once there is one.

    A partial report is never served. Half an analysis rendered in a UI is indistinguishable
    from a finished one that found nothing.
    """
    record = require_record(context, analysis_id)
    if record.report is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"the report is not available yet; the analysis is {record.state.value}",
        )
    return record.report.to_dict()


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(
    analysis_id: str, context: Annotated[ServiceContext, Depends(get_context)]
) -> Response:
    """Remove the analysis, its video, and any rendered artifact.

    Media first, then the record: a crash between the two leaves a record pointing at absent
    media, which reports as not found, rather than a video on disk that nothing remembers.
    """
    import contextlib

    from api.services.cleanup import delete_media

    record = require_record(context, analysis_id)
    delete_media(context.storage, record)
    # A concurrent sweep may already have removed it; the client's outcome is the same.
    with contextlib.suppress(RecordNotFoundError):
        context.store.delete(analysis_id)
    get_logger().info("analysis deleted", extra={"analysis_id": analysis_id})
    return Response(status_code=status.HTTP_204_NO_CONTENT)
