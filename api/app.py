"""The FastAPI application and everything it holds.

The service is assembled in one place and handed to the routes as a context object rather
than reached for through module-level globals. That is what lets a test build a service with
an immediate queue, a temporary storage root, and a stub bundle, and exercise the real
handlers against it.

The model bundle is loaded once at startup. If it cannot be loaded the service starts
anyway — and says it is unhealthy, and refuses uploads it could never process. Failing to
start would take the health endpoint down with it, leaving an operator with a container that
exits and no statement of why.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aced.inference.bundle import ModelBundle, default_bundle_path, load_bundle
from api.config import Settings
from api.context import ServiceContext
from api.logging import configure_logging, get_logger
from api.routes.analyses import router as analyses_router
from api.routes.health import router as health_router
from api.routes.media import router as media_router
from api.services.cleanup import CleanupScheduler, reconcile
from api.services.queue import InProcessJobQueue, JobQueue
from api.services.records import AnalysisStore
from api.services.sqlite import DATABASE_FILENAME, Database, SqliteAnalysisStore
from api.services.storage import LocalStorage

BundleLoader = Callable[[Settings], ModelBundle]


def _require(database: Database | None) -> Database:
    if database is None:  # pragma: no cover — only reachable through a programming error
        raise RuntimeError("no database was opened for the default store")
    return database


def load_configured_bundle(settings: Settings) -> ModelBundle:
    path = settings.bundle_path if settings.bundle_path is not None else default_bundle_path()
    return load_bundle(path)


def mount_frontend(app: FastAPI, settings: Settings) -> bool:
    """Serve the built web app from the API, when there is one to serve.

    Mounted last, so every API route is matched first and a future `/analyses` asset name
    could never shadow the resource. Routing in the frontend is hash-based, which is what
    makes this a static mount and not a rewrite rule: every deep link is still a GET of `/`,
    so there is no history fallback to configure and nothing to keep in sync.

    Absent assets are not an error. The API is useful on its own — the CLI parity suite, the
    whole test suite, and anyone driving it with curl never build the frontend — so a missing
    `index.html` means "no frontend today", logged once, not a failure to start.
    """
    from fastapi.staticfiles import StaticFiles

    root = settings.resolved_web_root
    if not (root / "index.html").is_file():
        get_logger().info("no built frontend to serve", extra={"web_root": str(root)})
        return False

    app.mount("/", StaticFiles(directory=root, html=True), name="web")
    get_logger().info("serving the web app", extra={"web_root": str(root)})
    return True


def create_app(
    settings: Settings | None = None,
    *,
    store: AnalysisStore | None = None,
    storage: LocalStorage | None = None,
    queue: JobQueue | None = None,
    bundle_loader: BundleLoader | None = None,
    start_cleanup: bool = True,
) -> FastAPI:
    resolved = settings or Settings.from_env()
    configure_logging()

    # The database lives beside the media, not among it: the storage root's direct children
    # are the objects reconciliation compares against the records, and the database file
    # must never be something that pass could mistake for an orphan.
    database = Database(resolved.storage_root / "db" / DATABASE_FILENAME) if store is None else None
    context = ServiceContext(
        settings=resolved,
        store=store if store is not None else SqliteAnalysisStore(_require(database)),
        storage=storage if storage is not None else LocalStorage(resolved.storage_root),
        queue=queue if queue is not None else InProcessJobQueue(resolved.worker_count),
    )
    load = bundle_loader or load_configured_bundle

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger = get_logger()
        try:
            context.bundle = load(resolved)
            logger.info("model bundle loaded", extra=context.bundle_provenance() or {})
        except Exception as error:
            context.bundle = None
            context.bundle_error = str(error)
            logger.exception("model bundle could not be loaded; service is unhealthy")

        # Before any request is served: whatever the previous process left half-done is
        # settled now, so no client ever polls a record that stopped moving when it died.
        failed, orphaned = reconcile(context.store, context.storage)
        if failed or orphaned:
            logger.info(
                "startup reconciliation",
                extra={"analyses_failed": len(failed), "media_removed": len(orphaned)},
            )

        if start_cleanup:
            context.cleanup = CleanupScheduler(context.store, context.storage)
            context.cleanup.start()
        try:
            yield
        finally:
            if context.cleanup is not None:
                context.cleanup.stop()
            context.queue.shutdown(wait=True)
            if database is not None:
                database.close()

    app = FastAPI(
        title="ACED",
        description="Arm-technique analysis for bouldering video.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.context = context

    app.include_router(health_router)
    app.include_router(analyses_router)
    app.include_router(media_router)
    mount_frontend(app, resolved)
    return app


app = create_app()
