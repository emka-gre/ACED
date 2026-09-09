"""What a request handler is allowed to reach.

Lives apart from `api.app` so that the routes can depend on it without depending on the
application that mounts them — importing the app from a route module and the route module
from the app is a cycle, and the failure it produces (a partially initialised module with no
`router` in it) is opaque enough to be worth designing out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from aced.inference.bundle import BundleError, ModelBundle
from api.config import Settings
from api.services.cleanup import CleanupScheduler
from api.services.queue import JobQueue
from api.services.records import AnalysisStore
from api.services.storage import LocalStorage
from api.services.worker import AnalysisWorker, OverlayRenderer


@dataclass
class ServiceContext:
    """Everything a request handler is allowed to reach."""

    settings: Settings
    store: AnalysisStore
    storage: LocalStorage
    queue: JobQueue
    bundle: ModelBundle | None = None
    bundle_error: str | None = None
    cleanup: CleanupScheduler | None = field(default=None, repr=False)

    @property
    def healthy(self) -> bool:
        return self.bundle is not None

    def require_bundle(self) -> ModelBundle:
        if self.bundle is None:
            raise BundleError(self.bundle_error or "no model bundle is loaded")
        return self.bundle

    @property
    def worker(self) -> AnalysisWorker:
        return AnalysisWorker(store=self.store, storage=self.storage, bundle=self.require_bundle)

    @property
    def renderer(self) -> OverlayRenderer:
        return OverlayRenderer(store=self.store, storage=self.storage, bundle=self.require_bundle)

    def bundle_provenance(self) -> dict[str, Any] | None:
        return self.bundle.manifest.provenance if self.bundle is not None else None


def get_context(request: Request) -> ServiceContext:
    context: ServiceContext = request.app.state.context
    return context
