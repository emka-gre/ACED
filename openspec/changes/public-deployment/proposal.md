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

- **Accounts.** A climber registers with an email address and a password, signs in, and gets
  a session. Analyses belong to the account that uploaded them; nobody else can see them.
  Sign-in is a cookie, so the browser sends it on every request including the ones a
  `<video>` element makes on its own.
- **Ownership.** Every analysis endpoint requires a signed-in user who owns the analysis. A
  request from anyone else is answered as if the analysis does not exist. A signed-in climber
  can list their own analyses.
- **Abuse limits.** Uploads are rate-limited per account, registration is rate-limited per
  client and can be closed behind an invite code, and the service refuses new work when its
  queue is saturated rather than accepting an upload it will not get to for an hour.
- **Persistence.** Accounts, sessions and analyses live in SQLite and survive a restart.
  Startup reconciles what it finds: an analysis interrupted mid-run becomes failed rather than
  staying `running` forever, and media no record names is deleted.
- **A smaller image.** `tensorflow-cpu` and `opencv-python-headless` in place of the full
  packages. This is a dependency change against a pinned set, so it is gated on the parity
  suite: if the numbers move, it does not ship.
- **TLS and a production compose file**, with configuration and secrets kept out of the
  repository.
- **A consent step and a privacy notice.** A climber uploading footage of identifiable people
  is told what is stored, for how long, and how to delete it, before they upload rather than
  after — and can delete their account and everything under it.

## Impact

- Affected specs: `video-analysis-api` (accounts, ownership, limits, persistence),
  `analysis-web-app` (sign-in, my analyses, consent, account deletion), and a new
  `service-operations` capability for deployment, configuration and reconciliation.
- Affected code: `api/` throughout, `web/src/api/` and every view, `pyproject.toml`
  (a password-hashing dependency), `Dockerfile`, `docker-compose.yml`.
- **Not in scope:** email. No verification on sign-up, no password reset by email, no
  notifications. A forgotten password has no self-service recovery in this version; the
  design says what that means and what the operator can do.
- **Not in scope:** sharing a report with someone who is not its owner. A coach reads it over
  the climber's shoulder or from their phone. A read-only share link is a natural follow-up
  once accounts exist, and it is the size of one small change.
- **Not in scope:** horizontal scaling. The queue stays in-process; a second machine is a
  different change, and nothing here forecloses it.
