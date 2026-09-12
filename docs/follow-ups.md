# Follow-ups

What is knowingly unfinished, now that the `climbing-technique-analysis-app` change is
complete (89/89). Nothing here is a defect in what was delivered — each item is either a
scope decision that was made deliberately, a dependency that is missing from this machine, or
a claim that could be made stronger. They are written down so that the next person, including
future us, does not have to rediscover them.

Ordered roughly by what would bite first.

---

## Before this is used by anyone but its author

- [ ] **Add authentication.** There is none. An analysis id is random and unguessable, but
      anyone holding a link can view that climb, its video, and its report — and delete it.
      That is defensible for a thesis deliverable running on a laptop; it is not defensible
      the moment a second person's video is on the server. Until then, treat analysis links as
      secrets. See the README's *Data and retention* section, which says so plainly.

- [x] ~~**Decide what serves the frontend in production.**~~ Done: the API serves the built
      assets. `Dockerfile` gained a Node stage that builds `web/dist` on the host's own
      architecture — its output is JavaScript, so emulating linux/amd64 for it would cost time
      and buy nothing — and the `service` stage copies the result in. The mount is registered
      after every API route, so an asset can never shadow a resource, and a missing build is
      not an error: the API runs identically without one, which is how the tests and the CLI
      run it. `docker compose up api` is now the whole application on one origin.

- [ ] **Confirm the phone layout by hand.** Task 10.11 asks for a usable layout at phone width
      with no horizontal scrolling. It is written that way — one column that widens,
      `overflow-wrap` on every free-text field, thumb-sized controls — but **jsdom does not do
      layout**, so no test asserts it. This is the only deliverable in the whole change that
      only a human at 375px can confirm.

## Blocked on the thesis archive

The archive is not on this machine — see [THESIS_ARCHIVE.md](../THESIS_ARCHIVE.md). The
`ML_pipeline` splits were recovered from iCloud Drive, which unblocked task 3.6, but
`training/`, `streamlit/` and `video_62_B_Ema.MOV` are still missing.

- [ ] **Restore the archive and re-run `pytest -m parity` in full.** Nine tests skip without
      it: the pose stage against the recorded landmarks, the CLI's report against the
      library's, and the API's report against the CLI's. Everything downstream of pose
      extraction is already pinned against committed fixtures — what is missing is MediaPipe
      over real frames. The README records the narrower claim the last run supports; restoring
      the archive is what lets it record the whole one.

- [ ] **`scripts/build_bundles.py` and `scripts/verify_artifacts.py` cannot run** without it
      either. That matters only if a bundle needs rebuilding, but if one ever does, the
      archive is a prerequisite and not a step.

## Known costs, accepted for now

- [ ] **Requesting an overlay re-runs pose extraction.** `OverlayRenderer` recomputes
      landmarks rather than persisting them, because keeping them would mean a second on-disk
      format and a second thing to expire, for an artifact most analyses never ask for. If
      overlays turn out to be the common case, persisting frame classes alongside the report
      is the change — and it is confined to that one class.

- [ ] **The queue is in-process.** A bounded thread pool inside the API: no broker to run, and
      single-machine deployment needs none. It sits behind `JobQueue`, so swapping in Celery or
      RQ touches [api/services/queue.py](../api/services/queue.py) and adds a `worker` service
      to `docker-compose.yml`. Nothing else moves.

- [ ] **Records live in memory.** `InMemoryAnalysisStore` behind the `AnalysisStore` interface.
      A restart forgets every analysis, though the stored media survives in the volume and is
      then orphaned — the cleanup job will not find it, because the records that named it are
      gone. Worth fixing at the same time as persistence, not before.

- [ ] **`aced analyze --quiet` is not silent.** TensorFlow writes CUDA and oneDNN notices to
      stderr as it initialises. stdout stays pure JSON so piping is unaffected. Silencing it
      means setting `TF_CPP_MIN_LOG_LEVEL` process-wide, which is a global side effect the CLI
      should probably not impose without being asked.

- [ ] **The SHAP extra is declared but unused.** `pyproject.toml` carries an `explain` extra
      pinning `shap>=0.45`, and nothing imports it. The design names SHAP as an optional
      enrichment on top of the z-score attribution, never on the hot path. Either implement it
      behind that extra or drop the extra — an installable option that does nothing is worse
      than neither.

## Model and method

- [ ] **The models are a small-population artifact.** Trained on a handful of climbers in one
      gym, labelled by one person. The reports say so, and the metrics from task 3.6 are
      measured on 178 held-out windows from that same population — which bounds what they can
      tell you about anyone else. Widening that means new data and new labelling, which is
      research, not engineering.

- [ ] **The two backends are statistically indistinguishable.** On the held-out split the SVM
      scores marginally higher (F1 0.902 against 0.885) but the two agree on 171 of 178
      windows and an exact McNemar test gives p = 0.45. The LSTM ships as default because it is
      the only backend supplying per-fault confidence. A larger validation set could separate
      them; today's cannot. `scripts/evaluate_bundles.py` re-runs the comparison.

- [ ] **Multi-person footage is unsupported and undetectable.** `mp.solutions.pose` returns a
      single pose and exposes no signal about anyone else in frame, so there is nothing to
      detect *with*. The limitation is disclosed in every report rather than checked for.
      Changing this means a different pose backend.
