# The pipeline

How a video becomes a report, what each stage preserves from the research it was ported
from, and where a real-time path would attach.

```
video ─► PoseExtractor ─► FeatureExtractor ─► Windower ─► InferencePipeline ─► AnalysisReport
         landmarks         18 features        60 frames    sound / fault        faults, in
         per frame         per frame          stride 20    per window           plain language
```

Every stage consumes one frame at a time. Batch analysis is a loop over the same code a live
camera would feed; nothing here materialises the whole video first.

## Why the shape matters

The thesis produced a working method — MediaPipe landmarks, engineered biomechanical
features, sliding-window segmentation, trained LSTM and SVM classifiers. It existed as
research code: three near-duplicate pipeline modules, a committed virtualenv, hardcoded
absolute paths, no tests.

The port's job was to keep the numbers identical while making the method deployable. Where
this code differs from the research code, it is because the research code was doing something
that could silently produce a wrong answer — each of those is called out below.

---

## 1. Pose extraction — `aced/pose/`

MediaPipe Pose over each frame, keeping **nine** of its thirty-three landmarks: nose, both
shoulders, elbows, wrists, and hips. Those nine are what the models were fitted on.

`static_image_mode=False`, every other parameter left at its default. This is not a
preference — it is what the parity reference was generated under, and a recorded reference
only means something against the settings that produced it.

**Frame indices are preserved across undetected frames.** A frame where no pose was found
emits no record, so consecutive records can differ by more than one. That gap is meaningful:
the windower splits on it rather than bridging across it, and a window that spanned a
detection gap would be built from two moments that never followed each other.

**Changed from the research code:** an unreadable file returned an empty DataFrame, making a
corrupt video indistinguishable from a video with no climber in it. `UnreadableVideoError`
keeps those apart, because the first is a failure and the second is a valid answer.

**Out of scope:** multi-person detection. `mp.solutions.pose` returns a single pose and
exposes no signal about anyone else in frame, so there is nothing to detect *with*. The
report discloses the limitation instead of pretending to check for it.

## 2. Features — `aced/features/compute.py`

Eighteen biomechanical quantities per frame: joint angles, inter-landmark distances, and
vertical offsets. Emitted feature **names are byte-identical** to the research code's,
including where the port corrected a misspelled method name behind them — the names are part
of the model contract, and the models index into them by position.

Non-finite results — coincident landmarks, an out-of-domain `arccos` — become zero, exactly
as before.

Only 2D is used. `z` is carried for rendering and quality checks, never for inference: the
research method is 2D, and the camera-angle sensitivity that follows from that is disclosed in
every report rather than papered over.

**Changed:** features are computed per `FrameLandmarks` rather than through a DataFrame
`apply`, which removes a per-row Python call from the hot path and is what makes the streaming
shape possible.

## 3. Windowing — `aced/features/windowing.py`

Sixty-frame windows at stride 20, with three behaviours preserved exactly because the models
were trained on windows built this way:

- **runs** — a gap in frame indices ends a run; runs are windowed independently
- **padding** — a run shorter than one window becomes one window, zero-padded to length
- **flush** — a run whose tail is left uncovered gets a final window ending on its last frame

Each window records its real (non-padded) frame range, so padded positions can be excluded
from frame-level attribution rather than voting as though they were data.

**Changed:** the windower is a stateful object fed one frame at a time and emitting windows as
they complete, rather than a function over a materialised DataFrame.

## 4. Inference — `aced/inference/`

A **bundle** is a directory carrying the model, its manifest, and its scaler when it has one.
The manifest declares the feature list and order, window geometry, decision threshold, and two
things that carry real weight:

- `scaling`: `external` means this pipeline scales before the model sees the data; `internal`
  means the artifact scales its own input and must be handed raw features. The shipped SVM is
  a `Pipeline(StandardScaler, SVC)` and would be **silently double-scaled** otherwise —
  producing plausible, confident, wrong predictions.
- `confidence`: `probability` or `none`. The SVM is `probability=False`, so it emits no
  calibrated score, and the report says nothing about confidence rather than squashing a
  decision-function margin through a sigmoid to fill the field.

The label convention inherited from the research is `1 = sound, 0 = fault` — inverted relative
to intuition, and a standing source of bugs in code that passed bare integers around. It is
mapped to a named class at the model adapter and nothing downstream sees the integers again.

Frame-level class comes from a majority vote across the windows covering each frame, ties to
sound, padded positions excluded.

**Changed — this is the defect the port exists to fix.** The research pipeline **refit** the
LSTM's `StandardScaler` from a 15 MB training CSV on every inference run. Predictions
therefore depended on that file staying present and byte-identical, with nothing checking that
it had. Today's refit reproduces the training scaler exactly — verified, max absolute
difference `0.0` — so the predictions were correct; nothing enforced it, and the failure mode
was silent. The fitted scaler now ships inside the bundle, and a manifest that contradicts its
own contents fails to load.

The extractor's feature schema is validated against the bundle at load, by **name and order**,
with no reindexing, truncation or zero-fill. The research code's
`reindex(columns=..., fill_value=0.0)` turned a column misalignment into a wrong prediction;
here it is a load-time error.

## 5. Report — `aced/report/`

Consecutive faulty windows merge into distinct faults with frame and time ranges. Each fault
is attributed by comparing its features against the **climber's own sound windows in the same
climb** — a z-score against a within-climb baseline, which is what a coach does: your bad
moments relative to your good ones.

**Changed:** when a climb has no sound window at all, the research code silently fell back to
the all-window mean, producing a confident explanation derived from nothing. This detects that
case, marks the attribution unreliable, and says so.

Explanations are rendered from templates keyed on feature name. A test asserts that no feature
identifier and no statistical vocabulary appears in any rendered string — a climber reads
"your left arm stayed straight while you reached", never `left_elbow_angle` or a z-score.

Quality flags (incomplete detection, low visibility, too short) and the model's scope,
version, metrics and limitations travel with **every** report, not on a separate page.

The report is the only thing downstream ever sees. No feature vectors, no window arrays, no
model objects cross that line.

## 6. Delivery

`aced analyze` and the API worker are two callers of one function,
[`aced.analysis.analyze`](../aced/analysis.py). That is why the API's report equals the CLI's
by construction rather than by test — though `tests/e2e/test_api_cli_parity.py` asserts it
anyway, exactly.

The skeleton overlay is rendered **on request**, never during analysis: a full decode and
re-encode is the most expensive operation in the system and is wasted on the analyses nobody
watches back.

---

## Where real-time would attach

Every stage above already consumes one frame at a time, which is the entire cost of keeping
this door open, and it was close to zero.

A live path replaces stage 1 and nothing else. Browser-side MediaPipe Tasks produces landmarks
in the page; those post to a small `/infer` endpoint that feeds the same `FeatureExtractor` and
the same `Windower` instance held per session. Windows complete as frames arrive, and
`InferencePipeline.classify` runs on each one as it does now.

What would need thought, and is not free:

- **Attribution needs a baseline.** The z-score compares against sound windows from the same
  climb, which do not exist yet at the start of one. A live path would either withhold
  explanations until enough sound windows accumulate, or say plainly that early feedback is
  uncalibrated — the same honesty the missing-baseline case already gets.
- **Report building is terminal.** Fault merging and the summary describe a finished climb.
  Live output is a stream of window verdicts; the report stays a thing you get at the end.
- **The upload flow is untouched.** A real-time path is an addition beside it, not a
  replacement — the recorded path is what produces something you can watch back.
