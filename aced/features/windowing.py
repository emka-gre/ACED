"""Incremental segmentation of a feature stream into fixed-length overlapping windows.

Ported from the research code's `Sliding_Window`, with the same three behaviours preserved
exactly, because the models were trained on windows built this way:

  runs        a gap in frame indices ends a run; runs are windowed independently and no
              window ever spans a gap
  padding     a run shorter than one window becomes a single window zero-padded to length
  flush       a run whose length leaves frames uncovered after the last stride-aligned
              window gets one final window ending on the run's last frame, so the tail is
              never dropped

The difference from the research code is that frames arrive one at a time and windows come
out as soon as their last frame lands, rather than the whole video being materialised as a
DataFrame first. Batch analysis is a loop over `push`; a future real-time path is the same
loop fed by a live camera.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

DEFAULT_WINDOW_LENGTH = 60
DEFAULT_STRIDE = 20


@dataclass(frozen=True, slots=True)
class Window:
    """One classification unit: `length` timesteps of the selected features.

    `values` is always `(length, n_features)` including any zero padding, because that is
    the shape the models expect. `real_length` says how much of it is actual data, and
    `frame_start`/`frame_end` locate that data in the source video — padded positions belong
    to no frame and must not vote in frame-level attribution.
    """

    values: npt.NDArray[np.float64]
    frame_start: int
    frame_end: int
    real_length: int
    feature_names: tuple[str, ...]

    @property
    def padded(self) -> bool:
        return self.real_length < self.values.shape[0]

    @property
    def frame_indices(self) -> tuple[int, ...]:
        """Source frame indices of the real (non-padded) positions, in order."""
        return tuple(range(self.frame_start, self.frame_start + self.real_length))

    def flatten(self) -> npt.NDArray[np.float64]:
        """Row-major flattening to `(length · n_features,)`.

        Time-major: every feature at t0, then every feature at t1. This is the layout the
        flat-input models were fitted against, and it must line up with `flat_column_names`.
        """
        return self.values.reshape(-1)

    def flat_column_names(self) -> tuple[str, ...]:
        return flat_column_names(self.feature_names, self.values.shape[0])


def flat_column_names(feature_names: Sequence[str], length: int) -> tuple[str, ...]:
    """`<feature>_t<index>` columns in time-major order — feature varies fastest."""
    return tuple(f"{name}_t{t}" for t in range(length) for name in feature_names)


@dataclass
class Windower:
    """Stateful windower fed one frame at a time.

    Frames must arrive in increasing index order. A frame whose index is not exactly one
    more than the previous one ends the current run — that is how a stretch of undetected
    frames becomes a run boundary rather than a silent splice.
    """

    feature_names: tuple[str, ...]
    length: int = DEFAULT_WINDOW_LENGTH
    stride: int = DEFAULT_STRIDE

    _buffer: list[list[float]] = field(default_factory=list, init=False, repr=False)
    _run_start: int | None = field(default=None, init=False, repr=False)
    _last_index: int | None = field(default=None, init=False, repr=False)
    _next_start: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.length <= 0 or self.stride <= 0:
            raise ValueError("window length and stride must be positive")
        if not self.feature_names:
            raise ValueError("windower needs at least one feature")

    def push(self, frame_index: int, values: Sequence[float]) -> list[Window]:
        """Add one frame's features; return any windows that this frame completed."""
        if len(values) != len(self.feature_names):
            raise ValueError(
                f"frame {frame_index} has {len(values)} values, expected {len(self.feature_names)}"
            )
        if self._last_index is not None and frame_index <= self._last_index:
            raise ValueError(
                f"frames must arrive in increasing order: {frame_index} followed {self._last_index}"
            )

        emitted: list[Window] = []
        if self._last_index is not None and frame_index != self._last_index + 1:
            emitted.extend(self._end_run())

        if self._run_start is None:
            self._run_start = frame_index
        self._buffer.append(list(values))
        self._last_index = frame_index

        # Emit every stride-aligned window whose final frame has now arrived.
        while self._next_start + self.length <= len(self._buffer):
            emitted.append(self._window(self._next_start, self.length))
            self._next_start += self.stride

        return emitted

    def close(self) -> list[Window]:
        """Signal end of stream; return any window still owed for the trailing frames."""
        emitted = self._end_run()
        self._last_index = None
        return emitted

    def _end_run(self) -> list[Window]:
        """Finish the current run: emit its padded or flush window, then reset."""
        emitted: list[Window] = []
        count = len(self._buffer)

        if count and count < self.length:
            # Short run: one zero-padded window. No aligned window can have been emitted.
            emitted.append(self._window(0, count))
        elif count > self.length:
            # Long run: cover any tail the stride-aligned windows left behind.
            last_valid_start = count - self.length
            last_aligned_start = (last_valid_start // self.stride) * self.stride
            if last_aligned_start < last_valid_start:
                emitted.append(self._window(last_valid_start, self.length))
        # count == self.length needs nothing: the aligned window at offset 0 covered it.

        self._buffer.clear()
        self._run_start = None
        self._next_start = 0
        return emitted

    def _window(self, offset: int, real_length: int) -> Window:
        assert self._run_start is not None
        rows = self._buffer[offset : offset + real_length]
        values = np.zeros((self.length, len(self.feature_names)), dtype=np.float64)
        values[:real_length] = np.asarray(rows, dtype=np.float64)

        frame_start = self._run_start + offset
        return Window(
            values=values,
            frame_start=frame_start,
            frame_end=frame_start + real_length - 1,
            real_length=real_length,
            feature_names=self.feature_names,
        )
