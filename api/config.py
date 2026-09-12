"""Service configuration, read once from the environment.

A frozen dataclass rather than pydantic-settings: the settings are a dozen scalars, and
adding a dependency to parse them would cost more than it saves. Everything has a default
that works on a laptop, so `uvicorn api.app:app` runs with no configuration at all.

Nothing here reaches for the environment at call time. The settings object is built at
startup and handed to the parts that need it, so a test constructs one directly instead of
mutating `os.environ` and hoping nothing cached the old value.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

#: Containers that hold video we can decode. Checked by extension *and* by the browser's
#: declared content type, because either alone is trivially wrong: an extension is a rename
#: away from a lie, and some browsers send `application/octet-stream` for everything.
ALLOWED_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})

ALLOWED_CONTENT_TYPES = frozenset(
    {
        "video/mp4",
        "video/quicktime",
        "video/x-m4v",
        "video/x-msvideo",
        "video/x-matroska",
        "video/webm",
        # Sent by browsers and curl that decline to guess. The extension check still applies.
        "application/octet-stream",
    }
)

#: Where `npm run build` leaves the frontend, relative to the repository root. Assets are
#: served from here when they exist and ignored when they do not, so the API runs identically
#: with or without a built frontend — which is what the whole test suite does.
DEFAULT_WEB_ROOT = Path(__file__).resolve().parents[1] / "web" / "dist"

DEFAULT_MAX_UPLOAD_MB = 200
DEFAULT_RETENTION_HOURS = 24
DEFAULT_WORKERS = 2


def _int_from(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the service needs to know that is not code."""

    storage_root: Path = Path("storage")
    #: `None` means "resolve the default bundle at startup" — see `default_bundle_path`.
    bundle_path: Path | None = None
    #: Directory holding the built frontend. `None` falls back to `DEFAULT_WEB_ROOT`; a
    #: directory with no `index.html` in it simply means no frontend is served.
    web_root: Path | None = None
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    retention: timedelta = timedelta(hours=DEFAULT_RETENTION_HOURS)
    worker_count: int = DEFAULT_WORKERS
    allowed_extensions: frozenset[str] = field(default=ALLOWED_EXTENSIONS)
    allowed_content_types: frozenset[str] = field(default=ALLOWED_CONTENT_TYPES)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        environment = os.environ if env is None else env
        bundle = environment.get("ACED_BUNDLE")
        web_root = environment.get("ACED_WEB_ROOT")
        return cls(
            storage_root=Path(environment.get("ACED_STORAGE_ROOT", "storage")),
            bundle_path=Path(bundle) if bundle else None,
            web_root=Path(web_root) if web_root else None,
            max_upload_bytes=_int_from(environment, "ACED_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)
            * 1024
            * 1024,
            retention=timedelta(
                hours=_int_from(environment, "ACED_RETENTION_HOURS", DEFAULT_RETENTION_HOURS)
            ),
            worker_count=_int_from(environment, "ACED_WORKERS", DEFAULT_WORKERS),
        )

    @property
    def max_upload_mb(self) -> int:
        return self.max_upload_bytes // (1024 * 1024)

    @property
    def resolved_web_root(self) -> Path:
        return self.web_root if self.web_root is not None else DEFAULT_WEB_ROOT

    def describes_supported_formats(self) -> str:
        """The formats named in a rejection, so the message tells the client what to send."""
        return ", ".join(sorted(self.allowed_extensions))
