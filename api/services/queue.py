"""The work queue between a request and the analysis it started.

An interface with a thread-pool implementation for development, as the design calls for:
single-machine deployment needs no broker, and swapping in Celery or RQ later touches this
module and nothing else.

Jobs are submitted as callables. The queue's contract is that a job which raises does not
take the worker down and does not stop the next job from running — failure isolation is a
property of the queue, not something every job has to remember to implement.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol

from api.logging import get_logger

Job = Callable[[], None]


class JobQueue(Protocol):
    def submit(self, job: Job) -> None: ...

    def shutdown(self, *, wait: bool = True) -> None: ...


class InProcessJobQueue:
    """A bounded pool of worker threads in the API process.

    Bounded on purpose: analysis is CPU-bound through MediaPipe and TensorFlow, so more
    threads than the machine can run concurrently would slow every analysis down and make
    progress reporting a lie.
    """

    def __init__(self, workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="aced-worker")
        self._pending: set[Future[None]] = set()

    def submit(self, job: Job) -> None:
        future = self._pool.submit(self._guard(job))
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)

    def _guard(self, job: Job) -> Job:
        """Swallow anything the job raises, having logged it.

        A future's exception is only seen by whoever awaits it, and nobody awaits these. A
        job that fails must therefore log here, or the failure is silent — and the pool must
        stay usable, which it does as long as the exception does not escape the callable.
        """

        def guarded() -> None:
            try:
                job()
            except BaseException:
                get_logger().exception("job raised and was contained by the queue")

        return guarded

    def drain(self, timeout: float | None = None) -> None:
        """Block until submitted jobs finish. For tests and for orderly shutdown."""
        from concurrent.futures import wait

        wait(list(self._pending), timeout=timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


class ImmediateJobQueue:
    """Runs each job on the calling thread, before `submit` returns.

    For tests that want the analysis finished by the time the upload response is read. It
    deliberately keeps the same isolation contract: a raising job is logged and contained,
    so a test exercising failure sees the same record states production would produce.
    """

    def __init__(self) -> None:
        self.submitted = 0

    def submit(self, job: Job) -> None:
        self.submitted += 1
        try:
            job()
        except BaseException:
            get_logger().exception("job raised and was contained by the queue")

    def shutdown(self, *, wait: bool = True) -> None:
        return None
