## Why

A bachelor thesis produced a working method for detecting arm-technique errors in bouldering — MediaPipe pose extraction, engineered biomechanical features, sliding-window segmentation, and trained LSTM/SVM classifiers that reach the "straight arms" (long-arm) judgement a coach would make. But it exists only as research code: a 1.1 GB repository containing a committed virtualenv (~51k files), ~400 MB of intermediate CSVs, hardcoded absolute paths from the author's laptop, three near-duplicate pipeline modules, zero tests, and a Streamlit demo. Nothing about it is deployable, and one construction makes it fragile in a way that would eventually make it wrong: at inference the LSTM's `StandardScaler` is **refit from a training CSV** on every run rather than loading the scaler fitted during training, so predictions depend on a 15 MB data file staying present and byte-identical, with no check that it has. The refit currently reproduces the training scaler exactly (verified: refitting from `ML_pipeline/df_X_m_train_3split_2.csv` gives `mean_` and `scale_` identical to `training/LTSM/scaler.pkl`, and `ltsm_training.ipynb` fitted on that same file), so today's predictions are correct — but nothing in the code enforces that, and the failure mode is silent.

The research is done; the product is not. This change extracts the validated ML core into a clean, tested Python package and builds a real application around it: a climber uploads a video and receives a report that says what went wrong, when it went wrong, and why.

## What Changes

**Extract and clean the ML core into an installable `aced` package**

- Consolidate the three duplicate pipelines (`ml_pipeline_ltsm.py`, `ml_pipeline_MVS.py`, `app_mvs copy.py`) into one configurable inference pipeline with a pluggable model backend (LSTM or SVM).
- **BREAKING (vs. thesis code)**: the inference-time scaler refit is removed. The scaler fitted at training time is persisted alongside the model and loaded from a versioned model bundle. Training CSVs are no longer a runtime dependency. This is numerically a no-op today — the persisted scaler is verified identical to the refit result — so it costs no parity and removes the failure mode.
- Replace every hardcoded absolute path and `sys.path.append('..')` hack with package-relative resource loading and explicit configuration.
- Preserve the validated pipeline semantics exactly: 9 retained MediaPipe landmarks, the 18 engineered geometric features, the 8-feature selected subset, 60-frame windows with stride 20, zero-padding for short segments, and majority-vote frame attribution. Numerical parity with the thesis pipeline is a verified acceptance criterion, not an assumption.
- Fix the API surface without changing the math: the `Feature` class's misspelled methods (`compute_poition`, `compute_ditance`) get correct names, feature output names stay byte-identical so persisted models remain valid.
- Add a pytest suite covering feature computation, windowing edge cases (segment shorter than window, frame-index gaps, trailing partial stride), scaler/model loading, and end-to-end inference on a short fixture video.

**Build the application**

- A FastAPI backend exposing video upload, asynchronous analysis with progress, and report retrieval. Analysis runs in a background worker, not in the request cycle — a 30-second climb takes far longer than any sane HTTP timeout to process.
- A React + TypeScript frontend: upload, live processing status, and a report view with an annotated video player where flagged segments are seekable on the timeline.
- A report model that goes beyond a binary verdict: per-segment good/bad classification with confidence, temporal location in both frames and seconds, the dominant contributing feature per flagged segment, and a plain-language explanation ("left elbow held at 94° through 0:12–0:15 — arm bent where a straight arm would save forearm strength").
- Skeleton-overlay video generation, carried over from the thesis code, rendered as a downloadable/streamable artifact of the analysis.
- A CLI (`aced analyze <video>`) that runs the same pipeline headlessly and emits the same report as JSON.

**Establish the repository**

- Build the app repo from scratch with proper hygiene: no virtualenv, no datasets, `.gitignore`, dependency management, formatting/lint config, CI running the test suite.
- Import only the trained LSTM and SVM models plus the newly persisted scaler as versioned model artifacts. Thesis notebooks, intermediate CSVs, labeling tools, and raw datasets are archived outside the app repository — they are research history, not runtime dependencies.
- **Real-time feedback is explicitly out of scope for this change**, but the pipeline is designed so a streaming path is additive: the feature extractor and windowing operate incrementally rather than assuming a complete video is in memory.

## Capabilities

### New Capabilities

- `pose-extraction`: Extracting per-frame skeletal landmarks from a climbing video, retaining the relevant joint subset, and handling frames where no pose is detected.
- `technique-features`: Computing the engineered biomechanical features (joint angles, relative heights, inter-joint distances) from landmarks, and segmenting a frame sequence into fixed-length overlapping windows.
- `technique-inference`: Loading a versioned model bundle (weights + fitted scaler + feature schema) and classifying windows as technically sound or faulty, with confidence scores.
- `analysis-report`: Turning window-level predictions into a climber-facing report — temporal localization of faults, per-fault dominant feature attribution, plain-language explanations, and overall session summary.
- `video-analysis-api`: HTTP API for uploading a video, tracking analysis progress asynchronously, and retrieving the resulting report and rendered artifacts.
- `analysis-web-app`: Browser interface for upload, processing status, and report review with a seekable annotated video timeline.

### Modified Capabilities

None — this is a greenfield application repository.

## Impact

- **New code**: `aced/` Python package (pose, features, inference, report, CLI), `api/` FastAPI service, `web/` React frontend, `tests/`.
- **Imported assets**: `training/MODELS/LSTM_model.keras` (Keras 3.12.1 format, input shape `(60, 8)`), `training/MODELS/SVM_model` (a scikit-learn `Pipeline` of `StandardScaler` + `SVC(class_weight='balanced')`, 480 flat inputs), and `training/LTSM/scaler.pkl` (`StandardScaler`, 480 features, 1588 samples) from the thesis archive, repackaged as versioned model bundles with an accompanying feature schema and metadata (training date, feature list, window parameters, validation metrics).
- **Imported fixtures**: raw MediaPipe pose CSVs from `labeling/csv_pose/mediapipe/`, notably `pose_data_essai_62_B_Ema.csv` (619 frames), which pairs with the archive's single video `video_62_B_Ema.MOV` and serves as the parity fixture for everything downstream of MediaPipe.
- **Retired**: the Streamlit prototype (`streamlit/`), all duplicate pipeline modules, the committed `bachelor_project/` virtualenv, and ~400 MB of intermediate CSVs.
- **Dependencies**: mediapipe, opencv-python, numpy, pandas, scikit-learn, tensorflow/keras, joblib, shap for the core; fastapi, uvicorn, and a task queue for the API; React/TypeScript/Vite for the frontend.
- **Resolved before proposal**: the persisted `scaler.pkl` has been confirmed to be the scaler fitted for the shipped LSTM — its parameters are bitwise identical to a refit from `df_X_m_train_3split_2.csv`, which is the file `ltsm_training.ipynb` fitted on. No offline refit migration is needed.
- **Risk**: the two backends scale differently. The LSTM consumes externally scaled input; the SVM carries its own `StandardScaler` inside its pipeline and is fed raw features. The bundle manifest must declare which, or the SVM gets double-scaled silently.
- **Risk**: `SVC` was trained with `probability=False`, so the SVM backend has no calibrated probability — only a decision-function margin. Confidence is therefore backend-dependent.
- **Storage**: uploaded videos and rendered overlays require a storage strategy and retention policy; videos of identifiable people are personal data.
