## 1. Repository Foundation

- [x] 1.1 Archive the thesis zip outside the app repo (tagged copy + a short `THESIS_ARCHIVE.md` recording where the datasets, notebooks, and labeling tools live); remove `bachelor-project-main.zip` from the working tree
- [x] 1.2 Create the repo skeleton: `aced/` package, `api/`, `web/`, `models/`, `tests/`, `fixtures/`
- [x] 1.3 Add `.gitignore` covering virtualenvs, `__pycache__`, `.DS_Store`, `*.csv`, model checkpoints outside `models/`, uploaded media, and build output
- [x] 1.4 Add `pyproject.toml` with `requires-python = ">=3.11,<3.13"` (mediapipe 0.10.21 publishes no wheels above CPython 3.12; the thesis used 3.11) and pinned runtime dependencies (mediapipe 0.10.21, numpy 1.26.4, opencv-python 4.11.0.86, pandas, scikit-learn, tensorflow/keras at a version that reads a Keras 3.12 archive, joblib) and dev dependencies (pytest, ruff, mypy); generate a lock file
- [x] 1.4a Commit a `.python-version` (done) and verify `LSTM_model.keras` loads and `SVM_model` unpickles in the locked environment before building anything on top of them
- [x] 1.5 Configure ruff and mypy; add `Makefile` or task runner entries for `test`, `lint`, `format`
- [x] 1.6 Add CI running lint, type check, and the test suite on push
- [x] 1.7 Write the top-level `README.md`: what ACED does, its scope and limitations, setup, and how to run the API, worker, and web app

## 2. Reference Fixtures and Parity Harness

- [x] 2.1 Commit `labeling/csv_pose/mediapipe/bad/pose_data_essai_62_B_Ema.csv` (619 frames, 33 landmarks, pairs with the archive's only video `video_62_B_Ema.MOV`) as the primary parity fixture — landmark-level input pins features, windowing, scaling, and prediction with no MediaPipe dependency and no video in git
- [x] 2.1a Generation script written (`scripts/generate_reference.py`); features and SVM stages recorded, LSTM and pose stages pending. Set up the Python 3.11 reference environment (mediapipe 0.10.21 + TensorFlow) and run the thesis pipeline once on `video_62_B_Ema.MOV` to record the reference outputs — pose CSV, feature CSV, segment CSV, per-window predictions and probabilities; the archive contains no recorded output, so this generates it. Keep the video outside git; commit only the recorded CSVs
- [x] 2.2 Construct three edge-case fixtures as landmark CSVs derived from the reference pose data (the archive has only one video, so these are built, not extracted): a sequence with a frame-index gap, a sequence shorter than 60 frames, and a sequence whose length is not a multiple of the stride
- [x] 2.3 Write the parity harness that compares packaged-pipeline output against the recorded reference at each stage (landmarks, features, windows, predictions) with explicit tolerances; stages downstream of pose SHALL be runnable from the CSV fixture alone
- [x] 2.4 Record hand-computed expected values for a handful of feature frames so feature correctness is pinned independently of the thesis code

## 3. Model Bundles

- [x] 3.1 Define the `bundle.json` schema: bundle id, version, model kind, input shape, ordered feature list, window length and stride, landmark subset, decision threshold, **scaling mode** (`external` with a shipped scaler, or `internal` to the model artifact), **confidence kind** (`probability` or `none`), training date, validation metrics, scaler provenance
- [x] 3.2 Write the migration script that assembles a bundle from a thesis model file plus, for externally scaled bundles, a fitted scaler
- [x] 3.3 Build the LSTM bundle from `LSTM_model.keras` + `training/LTSM/scaler.pkl` as `scaling: external`, `confidence: probability`; **verify** its predictions on the reference video match the thesis pipeline's recorded output. The scaler itself is already verified: its `mean_`/`scale_` are bitwise identical to a refit from `df_X_m_train_3split_2.csv`, the file `ltsm_training.ipynb` fitted on — record that provenance in `bundle.json`
- [x] 3.4 Not triggered — 3.3 passed parity on both bundles. If 3.3 fails parity, the cause is downstream of the scaler (feature or windowing divergence) — diagnose before proceeding rather than refitting. **This gates all downstream work**
- [x] 3.5 Build the SVM bundle from `training/MODELS/SVM_model` as `scaling: internal` (the artifact is `Pipeline(StandardScaler, SVC)` and the thesis feeds it raw features), `confidence: none` (`probability=False`); confirm first which of the two SVM artifacts the thesis reports on, then verify parity the same way
- [ ] 3.6 Evaluate both bundles on the held-out validation set, record metrics in each `bundle.json`, and choose the default backend — noting that only the LSTM supplies per-fault confidence
- [x] 3.7 Implement `BundleLoader`: loads model + manifest + declared scaler, validates completeness and that the declared scaling mode matches the bundle's contents, exposes provenance, fails loudly on missing or contradictory components

## 4. Pose Extraction

- [x] 4.1 Define `FrameLandmarks` (frame index, timestamp, named landmarks with x/y/z/visibility) and `VideoMeta` (fps, frame count, dimensions, fps-defaulted flag)
- [x] 4.2 Implement `PoseExtractor` as a generator over MediaPipe results, retaining only the nine model landmarks
- [x] 4.3 Preserve original frame indices across undetected frames so downstream gap detection works
- [x] 4.4 Raise a distinct `UnreadableVideoError` for undecodable files, separate from the no-pose-detected outcome
- [x] 4.5 Compute quality signals: undetected-frame proportion and mean landmark visibility. Multi-person detection is **out of scope** — `mp.solutions.pose` returns a single pose and exposes no signal about other people in frame; the limitation is disclosed in the report instead (7.8)
- [x] 4.6 Tests: detected/undetected frames, gap preservation, unreadable file, missing fps default, quality signals

## 5. Features and Windowing

- [x] 5.1 Port the eighteen feature computations into `aced/features/`, correcting the misspelled method names while keeping emitted feature names byte-identical
- [x] 5.2 Replace non-finite results (coincident landmarks, out-of-domain `arccos`) with zero, and cover it with tests
- [x] 5.3 Implement `FeatureExtractor.from_frame()` operating on one `FrameLandmarks` — no DataFrame `apply`
- [x] 5.4 Implement bundle-driven feature subset selection with a hard failure on any missing declared feature
- [x] 5.5 Implement `Windower` as an incremental, stateful object: contiguous-run detection, 60-frame windows at stride 20, zero-padding for short runs, trailing flush window, `close()` to drain
- [x] 5.6 Track and expose each window's real (non-padded) frame range
- [x] 5.7 Implement the flattened `<feature>_t<index>` representation in the exact column order the flat-input models expect
- [x] 5.8 Tests: feature parity against fixtures and hand-computed values, run splitting across gaps, padding, stride alignment, flush window, flatten column order, incremental emission timing

## 6. Inference

- [x] 6.1 Define `PredictedClass` as a named enum and confine the numeric-encoding mapping to the model adapter; model confidence is optional and carries its declared kind
- [x] 6.2 Implement the sequence-input adapter (windows with time and feature axes preserved)
- [x] 6.3 Implement the flat-input adapter (windows in flattened column layout)
- [x] 6.4 Implement `InferencePipeline`: apply the bundle's scaler only when the manifest declares external scaling, adapt, predict, apply the bundle threshold, return one result per window with class, frame range, and confidence when the backend provides one
- [x] 6.5 Validate the extractor's feature schema against the bundle at load and refuse to run on mismatch — no reindexing, truncation, or zero-fill
- [x] 6.6 Implement frame-level majority-vote attribution, ties to sound, excluding padded positions
- [x] 6.7 Tests: no fitting occurs at inference, schema mismatch is rejected, empty window set, threshold behavior, both backends return identical result shapes, vote and tie handling
- [x] 6.7a Tests for scaling mode: an internally scaled bundle receives unscaled input (guarding against double-scaling the SVM), an externally scaled bundle receives scaled input, and a manifest contradicting its contents fails to load
- [x] 6.8 Parity test: reference video window classifications match the recorded reference

## 7. Report Generation

- [x] 7.1 Define the report types (`AnalysisReport`, `Segment`, `Fault`, `QualityFlag`, `Summary`) with JSON serialization and no model-internal fields
- [x] 7.2 Implement fault merging: consecutive/overlapping faulty windows into distinct faults with frame and time ranges plus aggregated confidence
- [x] 7.3 Implement summary computation (duration, window count, faulty proportion, fault count)
- [x] 7.4 Implement z-score attribution against the within-climb sound baseline: dominant feature, value, baseline value, deviation, ranked top three contributors
- [x] 7.5 Detect the absent-sound-baseline case, mark attributions low-reliability, and state it in the report — do not silently fall back to the all-window mean
- [x] 7.6 Implement the feature-to-language templates and the explanation renderer; assert in tests that no feature identifiers or statistical terms appear in rendered text
- [x] 7.7 Populate quality flags from pose-extraction signals (undetected-frame proportion, mean visibility) plus the too-short-video case
- [x] 7.8 Attach model scope, limitations, and bundle version to every report, including that only a single climber is analyzed and that multi-person footage is unsupported
- [ ] 7.9 Port skeleton-overlay video rendering into `aced/render/`, coloring frames by assigned class, invoked on demand rather than during analysis
- [x] 7.10 Tests: fault merging (adjacent, separated, none), summary arithmetic, attribution ranking, missing baseline, explanation text hygiene, quality flags, serialization round-trip

## 8. CLI

- [ ] 8.1 Implement `aced analyze <video>` writing the report as JSON, with options for bundle selection and overlay rendering
- [ ] 8.2 Non-zero exit and a clear message on unreadable input or bundle load failure
- [ ] 8.3 End-to-end test: CLI on the reference video produces the expected report

## 9. API Service

- [ ] 9.1 Scaffold the FastAPI app, configuration, structured logging, and error handling
- [ ] 9.2 Define the `Storage` interface with a local-filesystem implementation; content-address stored media
- [ ] 9.3 Define the `JobQueue` interface with an in-process implementation for development
- [ ] 9.4 Define the analysis record model and persistence (id, state, progress, expiry, bundle version, report, artifact references)
- [ ] 9.5 Implement `POST /analyses`: streaming upload with format and size validation before acceptance, returns 202 with id and location
- [ ] 9.6 Implement the worker: consumes the queue, runs the pipeline, reports progress, writes the report, marks terminal state
- [ ] 9.7 Implement `GET /analyses/{id}`: state, progress, report when succeeded; 404 for unknown
- [ ] 9.8 Implement `GET /analyses/{id}/report`: full report as JSON; explicit not-yet-available response when not succeeded
- [ ] 9.9 Implement media endpoints for the original and overlay videos with byte-range support
- [ ] 9.10 Implement on-request overlay rendering with a preparing state until the artifact exists
- [ ] 9.11 Implement `DELETE /analyses/{id}` removing report, video, and artifacts
- [ ] 9.12 Implement the expiry cleanup job
- [ ] 9.13 Implement `GET /health` reporting service health and loaded bundle provenance; report unhealthy when the bundle fails to load
- [ ] 9.14 Ensure failures are isolated: a crashed job lands in a terminal failed state and the service keeps serving
- [ ] 9.15 Return client-safe failure reasons while logging diagnostics server-side
- [ ] 9.16 API tests: accepted upload, rejected format, oversized upload, concurrent analyses, status transitions, report gating, range requests, deletion, expiry cleanup, health, worker-crash isolation

## 10. Web App

- [ ] 10.1 Scaffold Vite + React + TypeScript with lint, format, and test setup
- [ ] 10.2 Generate or hand-write the typed API client from the report and analysis schemas
- [ ] 10.3 Build the upload view: file selection and drag-drop, client-side format and size validation, upload progress, retry on failure
- [ ] 10.4 Build the status view: polled state and progress, addressable by analysis id so a climber can leave and return
- [ ] 10.5 Build the report view: summary, fault list with times and explanations, quality flags shown prominently
- [ ] 10.6 Build the video player with a fault-marked timeline; selecting a fault seeks to it, playback entering a fault indicates which one
- [ ] 10.7 Add the overlay toggle preserving playback position
- [ ] 10.8 Handle the empty-fault-list case with an explicit "no faults detected" statement
- [ ] 10.9 Display analysis scope, model version, and limitations with every report
- [ ] 10.10 Implement deletion with a confirmation step
- [ ] 10.11 Make the layout usable at phone width with no horizontal scrolling
- [ ] 10.12 Frontend tests for upload validation, status polling, fault seeking, and the empty and failed states

## 11. Integration and Delivery

- [ ] 11.1 End-to-end test: upload the reference video through the API, poll to completion, and assert the report matches the CLI's output for the same video
- [ ] 11.2 Containerize the API and worker with the pinned environment; document local `docker compose` startup
- [ ] 11.3 Document the retention policy and the personal-data implications of storing gym videos
- [ ] 11.4 Write `docs/pipeline.md` describing each stage, the preserved thesis semantics, and where the real-time path would attach
- [ ] 11.5 Verify the full parity suite passes against the shipped default bundle and record the result in the README
