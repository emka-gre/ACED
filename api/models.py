"""The analysis record: what the service knows about one uploaded climb.

An immutable snapshot. Updates go through the store, which replaces the record wholesale
under a lock, so a reader in a request thread can never observe a record halfway through
being mutated by a worker thread — it holds an object that was consistent at the moment it
was handed over.

Timestamps are timezone-aware UTC throughout. A naive datetime here would be a retention bug
waiting for a daylight-saving boundary.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from aced.report.types import AnalysisReport


class AnalysisState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (AnalysisState.SUCCEEDED, AnalysisState.FAILED)


class ArtifactState(StrEnum):
    """Where a rendered artifact is in its life. Overlays are made on request, so `ABSENT`
    is the normal resting state and not an error."""

    ABSENT = "absent"
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"


def new_id() -> str:
    """An opaque identifier. Random, not sequential: an analysis id is the only thing
    guarding a stranger's climbing video, and a countable id is not a guard."""
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    id: str
    state: AnalysisState
    created_at: datetime
    expires_at: datetime
    filename: str
    video_key: str
    size_bytes: int
    progress: float = 0.0
    bundle_id: str | None = None
    bundle_version: str | None = None
    report: AnalysisReport | None = None
    failure_reason: str | None = None
    overlay: ArtifactState = ArtifactState.ABSENT
    overlay_key: str | None = None

    @classmethod
    def create(
        cls,
        *,
        filename: str,
        video_key: str,
        size_bytes: int,
        retention: timedelta,
        now: datetime | None = None,
    ) -> AnalysisRecord:
        created = now or utcnow()
        return cls(
            id=new_id(),
            state=AnalysisState.PENDING,
            created_at=created,
            expires_at=created + retention,
            filename=filename,
            video_key=video_key,
            size_bytes=size_bytes,
        )

    def with_changes(self, **changes: Any) -> AnalysisRecord:
        return replace(self, **changes)

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or utcnow()) >= self.expires_at

    def to_status(self) -> dict[str, Any]:
        """The status resource.

        The report appears only on success, and the failure reason only on failure. A
        pending analysis carrying a `report: null` field invites a client to render it.
        """
        payload: dict[str, Any] = {
            "id": self.id,
            "state": self.state.value,
            "progress": round(self.progress, 4),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "filename": self.filename,
            "overlay": self.overlay.value,
        }
        if self.bundle_id is not None:
            payload["model"] = {"bundle_id": self.bundle_id, "version": self.bundle_version}
        if self.state is AnalysisState.SUCCEEDED and self.report is not None:
            payload["report"] = self.report.to_dict()
        if self.state is AnalysisState.FAILED:
            payload["failure_reason"] = self.failure_reason
        return payload


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Something the storage backend is holding, addressed by its content."""

    key: str
    size_bytes: int
    digest: str = field(default="")
