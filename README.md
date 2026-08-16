# ACED

**A**utomated **C**limbing **E**rror **D**etection — a climber uploads a bouldering video and
gets back a report saying what went wrong with their arm technique, when it went wrong, and why.

The analysis method comes from a bachelor thesis: MediaPipe pose extraction → nine retained
landmarks → eighteen engineered geometric features → an eight-feature subset → 60-frame windows
at stride 20 → a binary LSTM or SVM classifier. This repository is the application built around
that method. See [THESIS_ARCHIVE.md](THESIS_ARCHIVE.md) for where the research lives and what
was imported from it.

## What it does and does not do

It detects **arm technique** — principally arms held bent where a straight arm would save
forearm strength. That is the whole of its scope.

It does not judge footwork, hip position, or route reading. It analyses one climber; footage
with several people in frame is unsupported. It works from 2D pose, which is sensitive to camera
angle — an arm bent toward the wall can project as straight. The models were trained on a small
set of climbers in one gym, labelled by one person. Every report states this, and results are
framed as observations rather than coaching instruction.

## Layout

```
aced/        the analysis package: pose, features, inference, report, render, cli
api/         FastAPI service and background worker
web/         Vite + React + TypeScript frontend
models/      versioned model bundles (weights + scaler + manifest)
fixtures/    parity fixtures — recorded landmark and reference data
tests/       unit, parity, and API suites
scripts/     one-off migration and reference-generation scripts
openspec/    the change proposal, design, specs, and task list
```

## Setup

Python is pinned to **3.11** (`.python-version`). mediapipe 0.10.21 publishes no wheels above
CPython 3.12, and the dependency set is pinned to what the models were fitted under — do not
widen it without re-establishing numerical parity.

```sh
make install     # creates .venv and installs the package with dev extras
make check       # lint, type check, tests
```

## Running it

```sh
aced analyze path/to/climb.mov -o report.json     # headless, same report the service produces
```

```sh
make api         # FastAPI service
make worker      # background analysis worker
cd web && npm run dev
```

Analysis runs as a background job: `POST /analyses` returns `202` with an id, `GET /analyses/{id}`
reports state and progress, and serves the report once it succeeds. A thirty-second climb is
around nine hundred frames of pose inference and takes minutes, not seconds.

## Parity with the research pipeline

The packaged pipeline is required to reproduce the research pipeline's output numerically. The
parity suite (`make parity`) checks each stage — features, windowing, scaling, predictions —
against recorded reference output, and runs from committed landmark CSVs with no MediaPipe
dependency. The pose stage itself is checked against the reference video, which is not committed;
those tests are marked `slow` and skip when it is absent.

One deliberate change from the research code: the LSTM's `StandardScaler` is loaded from the
model bundle instead of being refit from a training CSV at inference. This is numerically a
no-op — the persisted scaler's parameters are identical to the refit result — and it removes a
silent dependency on a 15 MB data file staying unchanged.

## Data and retention

Uploaded videos are footage of identifiable people. Each analysis carries an expiry, expired
media is deleted by a cleanup job, and a climber can delete an analysis and its video at any
time. Nothing derived from a real climb belongs in this repository.
