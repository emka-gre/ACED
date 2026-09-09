"""The background worker: one analysis, start to terminal state.

This is the API's only entry into the pipeline, and it calls exactly what the CLI calls —
`aced.analysis.analyze` — which is what makes "the API's report matches the CLI's output for
the same video" true by construction rather than by test.

Every path out of `run` ends in a terminal state. An analysis that raises becomes `failed`
with a client-safe reason; an analysis whose record vanished mid-run (the client deleted it)
stops quietly. Nothing leaves a record in `running` forever, because a status resource that
never resolves is indistinguishable to a client from a service that has hung.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aced.analysis import analyze
from aced.inference.bundle import ModelBundle
from aced.pose.extractor import UnreadableVideoError
from api.errors import client_safe_reason
from api.logging import get_logger
from api.models import AnalysisRecord, AnalysisState, ArtifactState
from api.services.records import AnalysisStore, RecordNotFoundError
from api.services.storage import LocalStorage, Storage


class BundleProvider(Protocol):
    """Hands out the loaded bundle, or raises if there isn't one.

    A callable rather than the bundle itself so the worker holds no reference to a model that
    failed to load at startup: the health endpoint's "unhealthy" and the worker's "cannot
    run" are then the same fact, asked at different times.
    """

    def __call__(self) -> ModelBundle: ...


@dataclass(frozen=True, slots=True)
class AnalysisWorker:
    store: AnalysisStore
    storage: Storage
    bundle: BundleProvider

    def run(self, analysis_id: str) -> None:
        """Analyze one record. Called on a worker thread; never raises to its caller."""
        logger = get_logger()
        try:
            record = self.store.get(analysis_id)
        except RecordNotFoundError:
            logger.info("analysis vanished before it started", extra={"analysis_id": analysis_id})
            return

        try:
            self._mark_running(analysis_id)
            source = self._source_path(record)
            bundle = self.bundle()

            analysis = analyze(source, bundle, on_progress=lambda f: self._progress(analysis_id, f))

            self.store.update(
                analysis_id,
                lambda current: current.with_changes(
                    state=AnalysisState.SUCCEEDED,
                    progress=1.0,
                    report=analysis.report,
                    bundle_id=bundle.manifest.id,
                    bundle_version=bundle.manifest.version,
                    failure_reason=None,
                ),
            )
            logger.info(
                "analysis succeeded",
                extra={
                    "analysis_id": analysis_id,
                    "faults": analysis.report.summary.fault_count,
                    "bundle": bundle.manifest.id,
                },
            )
        except RecordNotFoundError:
            logger.info("analysis was deleted while running", extra={"analysis_id": analysis_id})
        except BaseException as error:
            # The diagnostics go to the log with the traceback; the client gets a sentence.
            logger.exception("analysis failed", extra={"analysis_id": analysis_id})
            self._mark_failed(analysis_id, client_safe_reason(error))

    def _source_path(self, record: AnalysisRecord) -> Path:
        path = self.storage.local_path(record.video_key)
        if path is None:
            raise UnreadableVideoError(f"stored video {record.video_key} is no longer available")
        return path

    def _mark_running(self, analysis_id: str) -> None:
        self.store.update(
            analysis_id,
            lambda current: current.with_changes(state=AnalysisState.RUNNING, progress=0.0),
        )

    def _progress(self, analysis_id: str, fraction: float) -> None:
        """Record progress, but never move a record that has already finished.

        The callback fires from inside the pipeline, and a delete or a failure can land
        between two frames. Guarding inside the update keeps the check and the write under
        the store's lock.
        """

        def change(current: AnalysisRecord) -> AnalysisRecord:
            if current.state is not AnalysisState.RUNNING:
                return current
            return current.with_changes(progress=min(max(fraction, 0.0), 1.0))

        try:
            self.store.update(analysis_id, change)
        except RecordNotFoundError:
            raise

    def _mark_failed(self, analysis_id: str, reason: str) -> None:
        try:
            self.store.update(
                analysis_id,
                lambda current: current.with_changes(
                    state=AnalysisState.FAILED, failure_reason=reason
                ),
            )
        except RecordNotFoundError:
            return


@dataclass(frozen=True, slots=True)
class OverlayRenderer:
    """Renders an analysis's overlay video on request.

    The landmarks and frame classes are recomputed rather than persisted: keeping them would
    mean a second on-disk format and a second thing to expire, for an artifact most analyses
    never ask for. The cost is that requesting an overlay re-runs pose extraction — which is
    the same trade the design already makes by rendering lazily at all. If overlays turn out
    to be the common case, persisting the frame classes alongside the report is the change to
    make, and it is confined to this class.
    """

    store: AnalysisStore
    storage: LocalStorage
    bundle: BundleProvider

    def run(self, analysis_id: str) -> None:
        from aced.render.overlay import render_overlay

        logger = get_logger()
        try:
            record = self.store.get(analysis_id)
        except RecordNotFoundError:
            return

        destination = self.storage.reserve(".mp4")
        try:
            source = self.storage.local_path(record.video_key)
            if source is None:
                raise UnreadableVideoError(f"stored video {record.video_key} is unavailable")

            analysis = analyze(source, self.bundle())
            render_overlay(source, analysis.landmarks, analysis.frame_classes, destination)

            self.store.update(
                analysis_id,
                lambda current: current.with_changes(
                    overlay=ArtifactState.READY, overlay_key=destination.name
                ),
            )
            logger.info("overlay rendered", extra={"analysis_id": analysis_id})
        except RecordNotFoundError:
            destination.unlink(missing_ok=True)
        except BaseException:
            logger.exception("overlay rendering failed", extra={"analysis_id": analysis_id})
            destination.unlink(missing_ok=True)
            try:
                self.store.update(
                    analysis_id,
                    lambda current: current.with_changes(overlay=ArtifactState.FAILED),
                )
            except RecordNotFoundError:
                return
