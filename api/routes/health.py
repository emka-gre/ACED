"""Service health and the provenance of the model behind every answer it gives.

Health here means "can this service analyze a video", which is exactly one question: did the
model bundle load. A service that answers 200 while holding no model would have every upload
fail after the client had already waited for it.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status

from api.context import ServiceContext, get_context

router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    response: Response, context: Annotated[ServiceContext, Depends(get_context)]
) -> dict[str, Any]:
    if not context.healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "reason": "the model bundle could not be loaded",
            "bundle": None,
        }
    return {"status": "ok", "bundle": context.bundle_provenance()}
