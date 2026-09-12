"""Where uploaded videos and rendered artifacts live.

Content-addressed: an object's key is the SHA-256 of its bytes, so storing the same climb
twice stores one file, and a key cannot be turned into someone else's video by guessing at
it. The digest is computed while the stream is consumed rather than by re-reading the
finished file.

The interface is deliberately small — write a stream, open one, ask its size, delete it — so
an S3-backed implementation is a drop-in. `local_path` is the one concession to reality:
OpenCV decodes from a filesystem path, and a backend that has none would have to materialise
a temporary file to render an overlay. It is declared as returning `None` for exactly that
case rather than pretending every backend is a directory.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO, Protocol

from api.models import StoredObject

#: Chunk size for streaming reads and writes. Large enough that a 200 MB upload is a couple
#: of hundred iterations, small enough that a dozen concurrent uploads are not a memory
#: problem.
CHUNK_BYTES = 1024 * 1024


class UploadTooLargeError(ValueError):
    """The stream exceeded the configured limit and was abandoned part-way."""


class ObjectNotFoundError(KeyError):
    """No object is stored under that key."""


class ObjectWriter(Protocol):
    """An object being written incrementally, so an upload is never held in memory whole."""

    def write(self, chunk: bytes) -> None: ...

    def commit(self) -> StoredObject: ...

    def abort(self) -> None: ...


class Storage(Protocol):
    def writer(self, *, suffix: str, max_bytes: int) -> ObjectWriter: ...

    def store(self, chunks: Iterable[bytes], *, suffix: str, max_bytes: int) -> StoredObject: ...

    def open(self, key: str) -> BinaryIO: ...

    def size(self, key: str) -> int: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...

    def local_path(self, key: str) -> Path | None: ...

    def keys(self) -> set[str]: ...


class LocalObjectWriter:
    """Writes to a temporary file, then names it by the digest of what was written.

    The limit is enforced per chunk, so an oversized upload is abandoned the moment it
    crosses the line rather than after the whole thing has arrived — and the partial file is
    removed, because a rejected upload must not leave a video on disk.
    """

    def __init__(self, root: Path, incoming: Path, *, suffix: str, max_bytes: int) -> None:
        self._root = root
        self._suffix = suffix
        self._max_bytes = max_bytes
        self._digest = hashlib.sha256()
        self._written = 0
        self._path = incoming / f"{uuid.uuid4().hex}.part"
        self._handle: BinaryIO | None = self._path.open("wb")

    @property
    def bytes_written(self) -> int:
        return self._written

    def write(self, chunk: bytes) -> None:
        if self._handle is None:
            raise RuntimeError("writer is closed")
        if not chunk:
            return
        self._written += len(chunk)
        if self._written > self._max_bytes:
            self.abort()
            raise UploadTooLargeError(f"upload exceeds the {self._max_bytes} byte limit")
        self._digest.update(chunk)
        self._handle.write(chunk)

    def commit(self) -> StoredObject:
        if self._handle is None:
            raise RuntimeError("writer is closed")
        self._handle.close()
        self._handle = None

        digest = self._digest.hexdigest()
        key = f"{digest}{self._suffix}"
        destination = self._root / key
        if destination.exists():
            # Identical bytes are already stored: keep the original, drop the duplicate.
            self._path.unlink(missing_ok=True)
        else:
            shutil.move(str(self._path), str(destination))
        return StoredObject(key=key, size_bytes=self._written, digest=digest)

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._path.unlink(missing_ok=True)


class LocalStorage:
    """Files under a directory, in one flat namespace keyed by digest."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._incoming = self._root / "incoming"
        self._incoming.mkdir(exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def writer(self, *, suffix: str, max_bytes: int) -> LocalObjectWriter:
        return LocalObjectWriter(self._root, self._incoming, suffix=suffix, max_bytes=max_bytes)

    def store(self, chunks: Iterable[bytes], *, suffix: str, max_bytes: int) -> StoredObject:
        """Consume a whole stream and commit it — the synchronous convenience path."""
        writer = self.writer(suffix=suffix, max_bytes=max_bytes)
        try:
            for chunk in chunks:
                writer.write(chunk)
        except BaseException:
            writer.abort()
            raise
        return writer.commit()

    def _resolve(self, key: str) -> Path:
        """Reject anything that is not a bare filename directly under the root.

        Keys come from records the service itself wrote, but this is the one place a path is
        built from a string, and a traversal here would serve arbitrary files off the host.
        """
        candidate = (self._root / key).resolve()
        if candidate.parent != self._root.resolve():
            raise ObjectNotFoundError(key)
        return candidate

    def open(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        if not path.is_file():
            raise ObjectNotFoundError(key)
        return path.open("rb")

    def size(self, key: str) -> int:
        path = self._resolve(key)
        if not path.is_file():
            raise ObjectNotFoundError(key)
        return path.stat().st_size

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except ObjectNotFoundError:
            return False

    def delete(self, key: str) -> None:
        try:
            self._resolve(key).unlink(missing_ok=True)
        except ObjectNotFoundError:
            return

    def local_path(self, key: str) -> Path | None:
        try:
            path = self._resolve(key)
        except ObjectNotFoundError:
            return None
        return path if path.is_file() else None

    def keys(self) -> set[str]:
        """Every object held, by key — what reconciliation compares against the records.

        Files only, directly under the root. `incoming/` holds partial uploads that a
        writer is still filling, and anything else that is a directory is not an object.
        """
        return {child.name for child in self._root.iterdir() if child.is_file()}

    def reserve(self, suffix: str) -> Path:
        """A path for something an external tool will write.

        Overlay rendering is OpenCV writing to a filename, so it cannot go through a writer.
        The name is random rather than content-addressed because the content does not exist
        yet — which is why this is a `LocalStorage` method and not part of `Storage`.
        """
        return self._root / f"{uuid.uuid4().hex}{suffix}"


def iter_range(
    handle: BinaryIO, start: int, end: int, chunk_bytes: int = CHUNK_BYTES
) -> Iterator[bytes]:
    """Yield bytes `[start, end]` inclusive — the way HTTP counts a range."""
    handle.seek(start)
    remaining = end - start + 1
    try:
        while remaining > 0:
            chunk = handle.read(min(chunk_bytes, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk
    finally:
        handle.close()
