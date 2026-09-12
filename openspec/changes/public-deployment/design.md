## Context

The application is complete and runs locally. This change is about what has to be true before
it is reachable from the internet, and the constraints are mostly not about features.

The pinned dependency set is the binding one. `mediapipe==0.10.21`, `numpy==1.26.4`,
`tensorflow==2.16.2` are what the models were fitted under, and the parity suite exists to
prove the port reproduces the research numerically. Any dependency change here is subject to
that suite, and nine of its tests currently skip because the thesis archive is missing — so
the evidence available is weaker than it will look. That is stated plainly rather than
assumed away.

## Goals / Non-Goals

**Goals**

- A stranger who finds the URL cannot read someone else's climb, and cannot make the server
  do expensive work without limit.
- A restart loses nothing and leaks nothing.
- The image is small enough to deploy on ordinary hosting, with the numbers unchanged.
- A climber knows what happens to their video before they upload it.

**Non-goals**

- User accounts, sessions, password reset, email. Out of scope; see the proposal.
- Multi-machine deployment. The queue stays in-process.
- Any change to the analysis itself. Not one line of `aced/` should need to move.

## Decisions

### An owner token per analysis, not user accounts

`POST /analyses` returns `{id, token}`. The token is a random secret; only its hash is stored.
Every subsequent request for that analysis must present it.

*Why:* it fits the resource model that already exists. An analysis is already an anonymous,
expiring, self-contained thing addressed by an opaque id — attaching a secret to it is a
small change, where accounts would introduce a user model that nothing else in the system
needs.

*Trade-off:* a climber who loses the token loses the analysis. That is a real cost and the
web app has to carry it — it keeps the token in `localStorage` keyed by analysis id, and the
link it shows contains the token so it can be bookmarked and shared deliberately.

*Alternative considered:* signed cookies. Rejected — a cookie is per-browser, and the shareable
link is the feature the polling design exists for.

### Unauthorized reads answer 404, not 403

A request for an analysis without the right token gets the same answer as a request for an id
that never existed.

*Why:* 403 confirms the id exists. Given that ids are the addressing scheme, an endpoint that
distinguishes "wrong token" from "no such analysis" is an oracle for enumerating them.

### Media URLs carry the token as a query parameter

JSON endpoints take `Authorization: Bearer <token>`. The video and overlay cannot: a `<video>`
element issues its own requests and cannot be given a header, which is also why range requests
had to be implemented in the first place.

*Why this way:* the alternative is a cookie scoped to the media path, which reintroduces the
per-browser problem for the one case where the URL is most likely to be shared with a coach.

*Trade-off, stated because it is real:* tokens in URLs end up in server logs and browser
history. The mitigation is that these are our own logs, the structured logger already records
no URLs, and the token grants access to one expiring analysis rather than an account.

### SQLite for records, behind the interface that already exists

`AnalysisStore` is already a Protocol with an in-memory implementation. This adds a second
one. No caller changes.

*Why SQLite:* one file, no service to run, and it matches a deployment that is one container
on one machine. Postgres would be the answer for multiple workers; that is the change this one
deliberately does not make.

*Startup reconciliation:* an analysis found in `running` at startup is marked failed — the
worker that owned it died with the previous process, and nothing will resume it. Media with no
record is deleted. Both are the same principle the worker already follows: nothing is left in
a non-terminal state forever.

### The image shrinks by swapping two packages, gated on parity

`tensorflow` → `tensorflow-cpu`, `opencv-python` → `opencv-python-headless`. Neither changes
what is computed: `tensorflow-cpu` is the same wheels without the CUDA payload, and the
headless OpenCV build drops the GUI module this service never imports — it still decodes,
encodes and draws.

*Why it is still a risk:* it is a change to a pinned set, and the whole point of pinning was
that numerical parity was established under those exact versions. So the swap is not complete
until `pytest -m parity` passes against the shipped default bundle, and the result is recorded
in the README the way the last run was. If the numbers move, the change is reverted and the
image stays large.

*What weakens this:* the archive is missing, so the pose-stage parity tests skip. The evidence
covers everything downstream of pose extraction, which is where a TensorFlow change would
show, and does not cover MediaPipe over real frames, which is where the OpenCV change could.
That gap is real and goes in the README rather than being glossed.

### Caddy in front, for TLS

A reverse proxy container that obtains and renews certificates automatically and forwards to
the API.

*Why Caddy:* automatic certificate management with three lines of configuration. nginx with
certbot is the same outcome with a renewal cron job to forget about.

### Rate limiting in the application, not the proxy

Upload rate limiting lives in the API, keyed on client address, with the queue depth as a
second gate.

*Why not the proxy:* the limit that matters is not requests per second, it is *analyses in
flight* — a single upload can occupy a worker for minutes. Only the application knows that.

## Risks / Trade-offs

- **A lost token is a lost analysis.** There is no recovery path by design: recovery would
  require knowing who owns what, which is the user model this change avoids. The web app
  storing the token locally and surfacing the full link is the mitigation, and it is partial.
- **The parity evidence for the dependency swap is incomplete** while the archive is missing.
  Recorded, not hidden.
- **Rate limiting by client address is crude** behind a proxy or NAT — a gym's wifi is one
  address. The limit is therefore set to something a real user will not hit rather than
  something tight, and the queue-depth gate is what actually protects the machine.
- **SQLite serializes writes.** With a handful of concurrent analyses this is irrelevant; it
  would not be with fifty, and that is the point at which the store changes again.
- **Deploying at all creates an obligation** that running locally does not: video of
  identifiable people, on a machine, under someone's control. The consent step and the privacy
  notice are part of this change for that reason, not as decoration.

## Migration Plan

1. Ownership, persistence and limits, with tests — the service still runs locally throughout.
2. The dependency swap, **gated**: parity run and recorded before it is kept.
3. TLS, the production compose file, and the deployment note.
4. Consent and the privacy notice in the web app.

Steps 1 and 4 are independent of 2 and 3; if the dependency swap fails parity, everything else
still ships and the image stays 7.6 GB.
