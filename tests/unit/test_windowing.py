"""Windowing: run splitting, padding, stride alignment, the flush window, and flattening.

These are the paths the research code exercises least and where a port is most likely to
drift, so they are pinned individually rather than only through end-to-end parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from aced.features.windowing import Window, Windower, flat_column_names

FEATURES = ("a", "b")


def feed(windower: Windower, indices: list[int]) -> list[Window]:
    """Push one frame per index, with values that encode the frame number."""
    windows: list[Window] = []
    for index in indices:
        windows.extend(windower.push(index, [float(index), float(index) * 2]))
    windows.extend(windower.close())
    return windows


def test_run_longer_than_one_window_strides_by_twenty() -> None:
    # 140 frames at length 60, stride 20: starts at 0, 20, 40, 60, 80 and nothing else,
    # because 140 - 60 = 80 is exactly stride-aligned.
    windows = feed(Windower(FEATURES), list(range(140)))
    assert [window.frame_start for window in windows] == [0, 20, 40, 60, 80]
    assert all(window.real_length == 60 for window in windows)
    assert all(not window.padded for window in windows)


def test_trailing_frames_get_a_flush_window() -> None:
    # 145 frames: aligned windows end at frame 139, so frames 140-144 would be lost without
    # a final window starting at 145 - 60 = 85.
    windows = feed(Windower(FEATURES), list(range(145)))
    assert [window.frame_start for window in windows] == [0, 20, 40, 60, 80, 85]
    assert windows[-1].frame_end == 144
    assert not windows[-1].padded


def test_run_shorter_than_one_window_is_padded() -> None:
    windows = feed(Windower(FEATURES), list(range(35)))
    assert len(windows) == 1
    window = windows[0]
    assert window.values.shape == (60, 2)
    assert window.real_length == 35
    assert window.padded
    assert (window.frame_start, window.frame_end) == (0, 34)
    # The reported range covers only real frames, and the tail is zeros.
    assert np.all(window.values[35:] == 0.0)
    assert len(window.frame_indices) == 35


def test_run_of_exactly_one_window_is_emitted_once() -> None:
    windows = feed(Windower(FEATURES), list(range(60)))
    assert len(windows) == 1
    assert not windows[0].padded


def test_gap_splits_runs_and_no_window_spans_it() -> None:
    # Frames 0-40 and 55-120: a short run then a long one.
    indices = list(range(41)) + list(range(55, 121))
    windows = feed(Windower(FEATURES), indices)

    covered = [set(window.frame_indices) for window in windows]
    missing = set(range(41, 55))
    assert all(not (frames & missing) for frames in covered), "a window spanned the gap"

    first_run = [window for window in windows if window.frame_start < 41]
    assert len(first_run) == 1 and first_run[0].padded, "the 41-frame run should be padded"
    assert all(window.frame_start >= 55 for window in windows[1:])


def test_windows_are_emitted_as_their_last_frame_arrives() -> None:
    windower = Windower(FEATURES)
    for index in range(59):
        assert windower.push(index, [0.0, 0.0]) == [], f"emitted early at frame {index}"

    emitted = windower.push(59, [0.0, 0.0])
    assert len(emitted) == 1, "the first window should land exactly on frame 60"

    for index in range(60, 79):
        assert windower.push(index, [0.0, 0.0]) == []
    assert len(windower.push(79, [0.0, 0.0])) == 1


def test_close_drains_buffered_frames() -> None:
    windower = Windower(FEATURES)
    for index in range(30):
        windower.push(index, [0.0, 0.0])
    drained = windower.close()
    assert len(drained) == 1 and drained[0].real_length == 30


def test_close_on_an_empty_stream_produces_nothing() -> None:
    assert Windower(FEATURES).close() == []


def test_frames_must_arrive_in_order() -> None:
    windower = Windower(FEATURES)
    windower.push(5, [0.0, 0.0])
    with pytest.raises(ValueError, match="increasing order"):
        windower.push(5, [0.0, 0.0])


def test_wrong_number_of_values_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        Windower(FEATURES).push(0, [1.0])


class TestFlattening:
    def test_column_order_is_time_major(self) -> None:
        names = flat_column_names(("a", "b", "c"), length=2)
        assert names == ("a_t0", "b_t0", "c_t0", "a_t1", "b_t1", "c_t1")

    def test_eight_features_over_sixty_frames_give_480_columns(self) -> None:
        names = flat_column_names(tuple(f"f{i}" for i in range(8)), length=60)
        assert len(names) == 480
        assert names[0] == "f0_t0" and names[-1] == "f7_t59"

    def test_flattened_values_line_up_with_their_columns(self) -> None:
        windows = feed(Windower(FEATURES), list(range(60)))
        window = windows[0]
        flat = window.flatten()
        columns = window.flat_column_names()

        assert flat.shape == (120,)
        lookup = dict(zip(columns, flat, strict=True))
        # Frame 7 carried values (7, 14) by construction.
        assert lookup["a_t7"] == 7.0
        assert lookup["b_t7"] == 14.0
