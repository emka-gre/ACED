## Context

The thesis code establishes a validated method: MediaPipe Pose → 9 retained landmarks → 18 engineered geometric features → 8 selected features → 60-frame windows at stride 20 → binary LSTM/SVM classification of arm technique. That method works; the code around it does not survive contact with production.

Concretely, what the existing code does and what must be preserved:

| Stage | Thesis implementation | Status |
|---|---|---|
| Pose | `mp.solutions.pose`, all 33 landmarks with `x,y,z,visibility`; frames with no detection are **dropped**, so `frame` indices have gaps | Preserve semantics, make gaps explicit |
| Landmark subset | Keep `{0, 11, 12, 13, 14, 15, 16, 23, 24}` (nose, shoulders, elbows, wrists, hips) | Preserve |
| Features | 18 features from 2D `(x, y)` only: 4 angles, 4 signed vertical offsets, 10 distances | Preserve exactly |
| Selection | 8 features: `left/right_elbow_angle`, `right_wrist_shoulder_dist`, `left_head_elbow_dist`, `right_head_wrist_dist`, `right_hip_elbow_dist`, `left_wrist_shoulder_dist`, `left_elbow_height_rel_shoulder` | Preserve, move to model bundle metadata |
| Windowing | Contiguous-frame runs split; runs ≤ 60 frames zero-padded to 60; longer runs strided by 20 with a final flush window | Preserve, including padding and flush behavior |
| Flattening | Row-major `(N, 60, F)` → `(N, 60·F)`, columns named `{feature}_t{t}` ordered t-major | Preserve — the SVM consumes flat columns, the LSTM reshapes back |
| Scaling (LSTM) | **Refit `StandardScaler` from `ML_pipeline/df_X_m_train_3split_2.csv` at inference** | **Defect — replace with persisted scaler; verified numerically identical** |
| Scaling (SVM) | None applied externally — the saved artifact is `Pipeline(StandardScaler, SVC)` and scales internally | Preserve; bundle declares scaling as internal |
| Prediction | LSTM sigmoid ≥ 0.5 → 1 (good), else 0 (bad); SVM `.predict()` on raw flat features (`probability=False`, no `predict_proba`) | Preserve threshold, make it configurable |
| Attribution | Per-flagged-segment z-score of each aggregated feature against the good-segment mean/std; top 3 contributors | Preserve as the primary explanation mechanism |
| Frame mapping | Majority vote across overlapping windows covering each frame, ties → good | Preserve |

The label convention is `1 = good technique`, `0 = fault`. This is inverted relative to intuition and is a persistent source of bugs; the new code names it explicitly rather than passing bare ints around.

## Goals / Non-Goals

**Goals:**

- A pip-installable `aced` package whose inference output is numerically identical to the thesis pipeline on a reference video, verified by test.
- Model artifacts that are self-describing: weights, fitted scaler, feature schema, and window parameters travel together and are version-stamped.
- An asynchronous upload → analyze → report flow that survives a multi-minute analysis without HTTP timeouts.
- A report that localizes faults in time and explains them in language a climber understands, not feature names and z-scores.
- An incremental core: features and windows are computed from a stream of frames, so a future real-time path reuses the same code rather than forking it.

**Non-Goals:**

- Real-time / live webcam feedback. Designed for, not built here.
- Retraining, model improvement, or new fault classes. The shipped models classify arm technique only; hip position, footwork, and other faults are future capabilities.
- Multi-climber video. `mp.solutions.pose` returns a single pose and exposes no signal about other people in frame, so there is no way to detect the condition without changing the pose backend. Multi-person video is out of scope, and the limitation is stated in the report's disclosed limitations rather than flagged per-video. Detecting it would require MediaPipe Tasks `PoseLandmarker` with `num_poses > 1` or a separate person detector; that is a future change, and it would put pose parity back in play.
- User accounts, social features, coach dashboards.
- 3D pose lifting. The thesis explicitly works in 2D and inherits its camera-angle sensitivity; that limitation is surfaced to the user, not solved.

## Decisions

### Model bundle format over loose files

A bundle is a directory (`models/<name>/<version>/`) containing `model.keras` or `model.joblib`, optionally `scaler.joblib`, and `bundle.json` with the feature list in order, window length and stride, landmark subset, decision threshold, model kind, input shape, scaling mode, training date, and validation metrics. Loading validates that the feature schema in the bundle matches what the feature extractor produces, and refuses to run on mismatch.

*Why:* the current defect is possible only because the scaler is disconnected from the model. Making the scaler part of the artifact makes the defect unrepresentable. The schema check turns "silently wrong predictions from column misalignment" — which `reindex(columns=..., fill_value=0.0)` in the thesis code actively hides — into a startup error.

*Scaling is a declared mode, not an assumption.* `scaling: "external"` means the bundle ships `scaler.joblib` and the pipeline applies it before adapting — the LSTM case. `scaling: "internal"` means the model artifact scales its own input and the pipeline must pass raw features — the SVM case, whose saved artifact is `Pipeline(StandardScaler, SVC)`. A bundle declaring `external` without a scaler file fails to load; a bundle declaring `internal` and shipping one also fails, rather than leaving it ambiguous which scaler wins. Getting this wrong double-scales the SVM and produces plausible, wrong predictions with no error.

### Confidence is backend-dependent and declared

The LSTM emits a sigmoid probability and its confidence is that value. The shipped `SVC` was fitted with `probability=False`, so it has no calibrated probability — only `decision_function` margins on an arbitrary scale. The bundle declares `confidence: "probability" | "margin" | "none"`, and result objects carry an optional confidence plus its kind. Report and UI copy state a numeric confidence only when the kind is `probability`.

*Why:* squashing an uncalibrated SVM margin through a sigmoid to satisfy a "0 to 1 confidence" field would manufacture a number that looks like a probability and is not one. Refitting with `probability=True` is retraining, which this change excludes.

*Alternative considered:* MLflow or a model registry. Rejected as disproportionate for two static models that change rarely; a directory and a JSON file are inspectable and need no service.

### Backend Python, frontend TypeScript, split at an HTTP boundary

FastAPI serves the API; a Vite/React SPA consumes it. No server-side rendering, no Python templating.

*Why:* MediaPipe, Keras, and the feature code are Python and stay there. The report UI — a video player with a seekable fault timeline — is genuinely interactive and belongs in a real frontend framework. A clean HTTP boundary is also what makes the future real-time path viable: browser-side MediaPipe Tasks can produce landmarks and post them to a lightweight `/infer` endpoint without touching the upload flow.

*Alternative considered:* keeping Streamlit. Rejected — no control over video playback, no async job model, and no path to real-time.

### Analysis as a background job with a polled status resource

`POST /analyses` stores the video, creates an analysis record in `pending`, enqueues work, and returns `202` with the analysis ID. `GET /analyses/{id}` returns state (`pending` / `running` / `succeeded` / `failed`), a progress fraction, and the report when complete. The worker is a separate process consuming a queue.

*Why:* a 30-second climb at 30 fps is ~900 frames of MediaPipe inference plus LSTM prediction plus SHAP plus two rendered videos — minutes, not seconds. Doing that in a request handler blocks a worker and times out at every proxy in the path.

*Alternative considered:* WebSocket or SSE progress streaming. Polling is chosen for v1 because it is trivially resumable — the user can close the tab and come back — and progress granularity here is coarse anyway. SSE can be layered on later without changing the resource model.

*Alternative considered:* Celery + Redis. Deferred: the queue is behind a small interface with an in-process implementation for development, so single-machine deployment needs no broker. Swapping in Celery or RQ later touches one module.

### Explanations from z-score attribution, with SHAP as an optional enrichment

The per-fault explanation is generated from the aggregated-feature z-score comparison against good segments in the same climb, mapped to templated natural language keyed on feature name (`left_elbow_angle` → "left arm bent"). SHAP runs only when explicitly requested.

*Why:* the z-score method is self-contained, fast, deterministic, and explains a fault relative to the climber's own better moments — which is what a coach does. `shap.GradientExplainer` on an LSTM is slow, adds a heavy dependency to the hot path, and in the thesis code fails often enough that it is already wrapped in a bare `except`. Making it optional keeps the common path fast and honest.

*Trade-off:* z-scores are contrastive within a single video. A climb where every move is bad has no good baseline; the thesis code silently falls back to the all-segments mean. The new code detects this case and says the baseline is weak rather than presenting a confident explanation derived from nothing.

### Incremental frame pipeline

`PoseExtractor` yields `FrameLandmarks` per frame rather than returning a DataFrame; `FeatureExtractor` maps a `FrameLandmarks` to a feature vector; `Windower` is a stateful object fed frames one at a time that emits windows as they complete. Batch video analysis is a `for` loop over this, not a separate code path.

*Why:* this is the entire cost of keeping the real-time door open, and it is close to zero. It also removes the `df.apply(lambda row: ...)` per-row Python call that dominates feature extraction time today.

### Report structure decoupled from model internals

The report is a typed structure — `AnalysisReport { video_meta, summary, segments[], faults[], quality_flags[] }` — serialized to JSON, with `faults[]` carrying frame range, time range, confidence, dominant feature, contributing features, and rendered explanation text. The frontend renders only this; it never sees feature vectors or model outputs.

*Why:* it lets the model backend change (SVM → LSTM → something later) without touching the API contract or the UI, and it gives the CLI and the API the same output by construction.

### Video storage with explicit retention

Uploads go to a content-addressed store behind a `Storage` interface (local filesystem for development, S3-compatible in deployment). Each analysis carries an expiry; a cleanup job deletes expired videos and rendered artifacts.

*Why:* these are videos of identifiable people in gyms. Unbounded retention of personal data is a liability, and "we'll add deletion later" never happens. Retention is a field on the record from day one.

## Risks / Trade-offs

- ~~**The shipped `scaler.pkl` may not be the scaler fitted for the shipped LSTM**~~ → **Resolved before implementation.** `training/LTSM/scaler.pkl` is a `StandardScaler` with `n_features_in_ = 480`, `feature_names_in_` equal to the 480 `<feature>_t<index>` columns in emitted order, and `n_samples_seen_ = 1588`. Refitting from `ML_pipeline/df_X_m_train_3split_2.csv` (shape 1588×480) reproduces `mean_` and `scale_` with a maximum absolute difference of `0.0`, and `training/LTSM/ltsm_training.ipynb` fits its scaler on exactly that file. The scaler is the training scaler; no offline refit is required, and swapping the runtime refit for a load changes no prediction. `bundle.json` records this provenance.

- **The parity reference does not exist yet and must be generated** → The archive contains no recorded pipeline output, so the reference predictions have to be produced by running the thesis pipeline once, which needs Python 3.11 with mediapipe 0.10.21 and TensorFlow. That environment is a prerequisite for the parity harness, not a step within it. Most parity coverage does not need it: `labeling/csv_pose/mediapipe/` holds 139 raw pose CSVs, so features, windowing, scaling, and prediction can all be pinned from CSV input with no MediaPipe dependency at all. Only the pose stage itself requires the video and the pinned environment.

- **Numerical parity with the thesis pipeline may fail on edge cases even when the common path matches** → Parity tests cover the awkward inputs specifically: a video with frames where no pose is detected (gap handling), a clip shorter than one 60-frame window (padding), and a clip whose length is not a multiple of the stride (flush window). These are exactly the paths the thesis code exercises least.

- **2D pose is camera-angle sensitive** — an arm bent toward the wall can project as straight, which the thesis introduction names as a known limitation → Do not paper over it. The report carries quality flags (climber too small in frame, low landmark visibility, likely unfavorable angle, multiple people detected) and the UI states plainly when confidence is degraded.

- **The models were trained on one gym, a small set of climbers, and one labeler** → Report copy frames output as observations about arm position, never as authoritative coaching. Model metadata including validation metrics and known population is exposed in the UI, not buried.

- **MediaPipe and TensorFlow are heavy and version-fragile; MediaPipe is pinned at 0.10.21 with numpy 1.26.4** → Pin exactly, lock the environment, and containerize the worker. Do not attempt a dependency upgrade as part of this change; getting numerical parity and a dependency bump entangled makes both undiagnosable. Two constraints follow from the artifacts and must be encoded in `pyproject.toml`: mediapipe 0.10.21 publishes no wheels above CPython 3.12, so `requires-python = ">=3.11,<3.13"` (the thesis used 3.11); and `LSTM_model.keras` was written by Keras 3.12.1, so the Keras/TensorFlow pin must be able to read a Keras 3 archive.

- **SVM inference on flattened 480-column windows and LSTM on `(N, 60, 8)` need different input shapes from the same feature stream** → The bundle declares which shape the backend wants; the adapter is one function selected at load time, not branching scattered through the pipeline.

- **Rendering two overlay videos per analysis is expensive and may exceed the value it delivers** → Render lazily on first request rather than eagerly during analysis, and treat the simulation video (skeleton on blank background) as optional.

## Migration Plan

1. Archive the thesis repository as-is (tagged, stored outside the app repo) — it is the record of the research and the only source of the training data.
2. Extract the models and scaler; build and verify the model bundles against the thesis pipeline on a reference video. **Gate:** parity confirmed before proceeding. The scaler half of this gate is already cleared (see Risks); what remains is confirming end-to-end predictions once the reference output has been generated.
3. Build the `aced` package with tests, verifying parity continuously.
4. Build the API and worker against the package.
5. Build the frontend against the API.

Rollback is not applicable — nothing is being replaced in production. The thesis code remains in its archive throughout.

## Open Questions

- Which model ships as the default backend? The LSTM is the more developed line of work, but the SVM is faster and the thesis contains no head-to-head comparison on a common test split. Resolve by evaluating both bundles on the held-out validation set during migration and recording the result in `bundle.json`. Note the tiebreaker: only the LSTM yields a calibrated confidence, and the report leans on per-fault confidence, so the SVM would ship a visibly thinner report at equal accuracy.
- Which SVM artifact is the one the thesis reports on? Two exist — `training/MODELS/SVM_model` (1.9 MB, named by the migration task) and `training/SVM/SVM_model` (2.5 MB). Confirm against the training notebook before building the SVM bundle.
- What deployment target — a single container on a VPS, or a managed platform? Affects the storage and queue implementations chosen, though not the interfaces.
- Should a climber be able to correct the model's verdict on a segment? Cheap to add, and it is the only realistic source of new labeled data for future retraining. Deferred, but the report schema reserves room for it.
