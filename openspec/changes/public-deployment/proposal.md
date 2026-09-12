## Why

The application works. `docker compose up api` serves the API, its analysis workers and the
web app from one origin, and a climber can upload a video and read a report. It runs on a
laptop and it is reachable from a phone on the same wifi.

None of that is the same as being reachable from the internet, and three things stand between
the two.

**The upload endpoint is open.** Anyone who finds the URL can POST a 200 MB video and make the
server run minutes of MediaPipe and TensorFlow. That is not a privacy nitpick — it is an
unauthenticated, unmetered, compute-expensive endpoint, and it is the first thing that would
be found and used.

**An analysis id is the only thing guarding a climber's video.** Ids are random and
unguessable, which is why this was acceptable on a laptop. On a public host it means a link
forwarded once is access granted forever, to the video, the report, and the delete button.

**Records live in memory.** A restart forgets every analysis while its media survives in the
volume — orphaned, because the cleanup job can only delete what a record names. On a laptop
that is a nuisance. On a server that restarts for deployments, it is an accumulating pile of
other people's climbing videos that nothing will ever remove.

Two further things make a deployment awkward rather than unsafe: the image is **7.6 GB**,
which rules out most hosting and makes every deploy slow; and there is no TLS, which for a
service handling video of identifiable people is not optional.

## What Changes

- **Ownership.** An upload returns an owner token alongside the analysis id. Reading the
  status, the report, the media, or deleting the analysis requires it. An unowned request is
  answered as if the analysis does not exist, so a probe cannot confirm an id.
- **Abuse limits.** Uploads are rate-limited per client, and the service refuses new work when
  its queue is saturated rather than accepting an upload it will not get to for an hour.
- **Persistence.** Analysis records survive a restart, and startup reconciles what it finds:
  an analysis interrupted mid-run becomes failed rather than staying `running` forever, and
  media no record names is deleted.
- **A smaller image.** `tensorflow-cpu` and `opencv-python-headless` in place of the full
  packages. This is a dependency change against a pinned set, so it is gated on the parity
  suite: if the numbers move, it does not ship.
- **TLS and a production compose file**, with configuration and secrets kept out of the
  repository.
- **A consent step and a privacy notice.** A climber uploading footage of identifiable people
  is told what is stored, for how long, and how to delete it, before they upload rather than
  after.

## Impact

- Affected specs: `video-analysis-api` (ownership, limits, persistence), `analysis-web-app`
  (token handling, consent, privacy notice), and a new `service-operations` capability for
  deployment, configuration and reconciliation.
- Affected code: `api/` throughout, `web/src/api/` and the upload and report views,
  `pyproject.toml`, `Dockerfile`, `docker-compose.yml`.
- **Not in scope:** user accounts. A token per analysis is what the current resource model
  supports, and accounts would mean a user model, a login flow, password handling and email —
  a larger project that this one does not need in order to be safely reachable.
- **Not in scope:** horizontal scaling. The queue stays in-process; a second machine is a
  different change, and nothing here forecloses it.
