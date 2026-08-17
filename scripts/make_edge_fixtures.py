"""Derive the windowing edge-case fixtures from the reference landmark CSV.

Task 2.2. The archive contains exactly one video, so the awkward inputs — the ones the
research code exercises least and is therefore most likely to disagree on — have to be
constructed. Slicing real landmark rows keeps the values realistic while making the frame
index pattern exactly what each case needs.

The three cases, at window length 60 and stride 20:

  gap             two contiguous runs separated by missing frame indices, so the windower
                  must split them and never span the boundary
  short_run       fewer frames than one window, so the run is zero-padded to 60 and its
                  reported range must cover only the real frames
  stride_unaligned  a run whose length leaves frames uncovered after the last stride-aligned
                  window, so a flush window ending on the final frame must be emitted

Run:
    python scripts/make_edge_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "landmarks"
REFERENCE = FIXTURES / "reference_62_B.csv"

WINDOW = 60
STRIDE = 20


def _renumber(frame: pd.DataFrame, start: int) -> pd.DataFrame:
    """Reindex a slice to consecutive frame numbers beginning at `start`."""
    out = frame.reset_index(drop=True).copy()
    out["frame"] = range(start, start + len(out))
    return out


def build_gap(source: pd.DataFrame) -> pd.DataFrame:
    """Frames 0-40 and 55-120: two runs, a fourteen-frame hole between them.

    The first run is shorter than a window and the second is longer, so one fixture covers
    padding and striding on either side of a boundary.
    """
    first = _renumber(source.iloc[:41], start=0)
    second = _renumber(source.iloc[41:107], start=55)
    return pd.concat([first, second], ignore_index=True)


def build_short_run(source: pd.DataFrame) -> pd.DataFrame:
    """35 frames — below the 60-frame window, so the single window is zero-padded."""
    return _renumber(source.iloc[:35], start=0)


def build_stride_unaligned(source: pd.DataFrame) -> pd.DataFrame:
    """145 frames.

    last_valid_start = 145 - 60 = 85, which is not a multiple of the stride: aligned windows
    start at 0, 20, 40, 60, 80 and end at frame 139, leaving frames 140-144 uncovered until
    the flush window at start 85 is emitted.
    """
    length = 145
    assert (length - WINDOW) % STRIDE != 0, "fixture must be stride-unaligned to be worth having"
    return _renumber(source.iloc[:length], start=0)


BUILDERS = {
    "edge_gap.csv": build_gap,
    "edge_short_run.csv": build_short_run,
    "edge_stride_unaligned.csv": build_stride_unaligned,
}


def main() -> int:
    source = pd.read_csv(REFERENCE)
    for name, build in BUILDERS.items():
        frame = build(source)
        path = FIXTURES / name
        frame.to_csv(path, index=False)
        runs = (frame["frame"].diff() != 1).sum()
        print(
            f"{name}: {len(frame)} frames, {runs} run(s), "
            f"indices {frame['frame'].iloc[0]}-{frame['frame'].iloc[-1]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
