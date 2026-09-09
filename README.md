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
web/         Vite + React + TypeScript frontend (its own npm scripts and tests)
models/      versioned model bundles (weights + scaler + manifest)
fixtures/    parity fixtures — recorded landmark and reference data
tests/       unit, end-to-end, parity, and API suites
scripts/     one-off migration and reference-generation scripts
docs/        pipeline.md — each stage, what it preserves, where real-time attaches
             follow-ups.md — what is knowingly unfinished, and why
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

### No local Python 3.11, or on Apple Silicon

`tensorflow==2.16.2` was pinned against an Intel Mac and has no macOS arm64 wheel, and
`mediapipe==0.10.21` has no Linux arm64 wheel either — so on an Apple Silicon Mac neither a
native install nor a plain arm64 container works. `Dockerfile` runs the pinned environment as
`linux/amd64` instead (Docker Desktop emulates this transparently; nothing extra to pass):

```sh
docker compose build dev
docker compose run --rm dev bash            # interactive shell in the pinned env
docker compose run --rm dev pytest -m "not slow"
docker compose run --rm dev pytest -m parity
```

The repo is bind-mounted into the container, so edits made on the host take effect without
rebuilding; only a change to `pyproject.toml` does.

### The frontend

`web/` is a separate toolchain and runs on the host, not in the Python container — the image
carries no Node. It needs Node 20 or newer.

```sh
cd web
npm install
npm run check    # eslint, prettier, tsc, and the vitest suite
```

## Running it

```sh
aced analyze path/to/climb.mov -o report.json     # headless, same report the service produces
aced analyze climb.mov --overlay overlay.mp4      # also render the classified skeleton
aced analyze climb.mov --bundle models/svm-arm-technique/v1
```

With no `-o` the report goes to stdout and everything else to stderr, so `aced analyze
climb.mov | jq .summary` works. With no `--bundle` the newest shipped
`lstm-arm-technique` bundle is used, or whatever `ACED_BUNDLE` points at. The overlay is
rendered only when asked for, and only after the report is safely written.

Exit codes: `0` analyzed (including the valid outcome of no climber being found), `2` bad
arguments, `3` unreadable video, `4` bundle could not be loaded, `5` the report was written
but the overlay failed.

```sh
docker compose up api                   # the service in its pinned environment, on :8000
cd web && npm install && npm run dev    # the web app, on :5173
```

`docker compose up api` is the supported way to run it: the API and its analysis workers in
one container, on the same pinned linux/amd64 environment the tests run under. It is healthy
once `/health` answers 200, which happens only after the model bundle loads — the container's
healthcheck is that endpoint, so "started" and "able to analyze" are not confused. Uploads go
to a named volume, not into your working copy.

`make api` runs the same service from a local venv instead, for a machine that can install the
pinned dependencies natively.

The dev server proxies `/analyses` and `/health` to the API, so the browser sees one origin
and the service needs no CORS configuration for a case that only exists in development.
`ACED_API_URL` points the proxy somewhere other than `localhost:8000`.

Analysis runs as a background job: `POST /analyses` returns `202` with an id, `GET /analyses/{id}`
reports state and progress, and serves the report once it succeeds. A thirty-second climb is
around nine hundred frames of pose inference and takes minutes, not seconds.

| | |
|---|---|
| `POST /analyses` | upload a video, `202` with the analysis id |
| `GET /analyses/{id}` | state, progress, and the report once it succeeded |
| `GET /analyses/{id}/report` | the report alone; `409` until it exists |
| `GET /analyses/{id}/video` | the original, with byte-range support for seeking |
| `GET /analyses/{id}/overlay` | the skeleton overlay; the first request starts rendering it and answers `202` |
| `DELETE /analyses/{id}` | removes the record, the video, and any rendered artifact |
| `GET /health` | service health and the loaded bundle's provenance; `503` when no bundle loaded |

The queue is in-process for development — a bounded thread pool inside the API — so there is
no broker and no separate worker command to run. It sits behind an interface, and swapping in
Celery or RQ later touches [api/services/queue.py](api/services/queue.py) and nothing else.

Configuration is environment variables, all optional: `ACED_BUNDLE` (bundle directory),
`ACED_STORAGE_ROOT`, `ACED_MAX_UPLOAD_MB`, `ACED_RETENTION_HOURS`, `ACED_WORKERS`.

[docs/pipeline.md](docs/pipeline.md) describes each stage of the analysis, which research
semantics it preserves, where it deliberately differs, and where a real-time path would
attach. [docs/follow-ups.md](docs/follow-ups.md) lists what is knowingly unfinished —
authentication, the missing thesis archive, and the costs accepted along the way.

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

### Last recorded run — 2026-09-10, `lstm-arm-technique` v1.0.0

`docker compose run --rm dev pytest -m parity` → **17 passed, 9 skipped**.

Passing, from committed fixtures: feature values against the recorded reference and against
hand-computed values, run splitting across gaps, padding, stride alignment and the flush
window, both bundles' window classifications, LSTM probabilities, the SVM inventing no
confidence, the two backends returning identical result shapes, nothing being fitted at
inference, and frame-level attribution.

Skipped, all nine for the same reason — the reference video is not on this machine (see
[THESIS_ARCHIVE.md](THESIS_ARCHIVE.md)): the pose stage against the recorded landmarks, the
CLI's report against the library's, and the API's report against the CLI's.

**So the claim this run supports is narrower than "parity holds".** Everything downstream of
pose extraction is pinned to the research pipeline's recorded output. The pose stage itself —
MediaPipe over real frames — is pinned only when the archive is present. Restore it and set
`ACED_THESIS_ARCHIVE` to re-run the full suite; inside the container it must also be mounted,
since the default path is under the host's home directory.

## Data and retention

Uploaded videos are footage of identifiable people in a gym. That makes them personal data,
and it shapes several things in this repository that would otherwise look like over-engineering.

**Retention is a field on the record from the moment it exists.** Every analysis carries an
expiry (`ACED_RETENTION_HOURS`, 24 by default). A cleanup job sweeps expired analyses and
deletes the stored video, any rendered overlay, and the record itself. Media is deleted before
the record, so a crash between the two leaves a record pointing at absent media — which reports
as not found — rather than a video on disk that nothing remembers to delete.

**A climber can delete their own analysis at any time**, before expiry, through
`DELETE /analyses/{id}` or the button in the web app. It removes the video, the overlay and the
report together.

**An analysis id is the only thing guarding a stranger's video.** Ids are random, not
sequential, for that reason. There is no authentication in this version: anyone holding a link
can view that analysis. That is a deliberate scope decision for a thesis deliverable, not an
oversight — but it means the links should be treated as secrets, and it is the first thing to
change before this is exposed to anyone beyond its author.

**Stored media never enters the repository.** Uploads live in a Docker volume rather than a
bind-mounted directory, so footage cannot accumulate in a working copy where it could be
committed or picked up by a file-sync client. Every video extension and every CSV outside
`fixtures/` is gitignored. The thesis reference video is not committed and must not be.

**Derived data is still personal data.** Pose landmarks and features describe an identifiable
person's body and movement; `fixtures/` contains landmark CSVs from one consenting climber and
is the deliberate exception. Nothing else derived from a real climb belongs here.

**What the service does not do:** no analytics, no third-party requests, no logging of video
content. Logs carry analysis ids, sizes and states — never frames, landmarks or filenames from
which a person could be identified.
