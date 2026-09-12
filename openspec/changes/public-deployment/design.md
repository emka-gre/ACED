## Context

The application is complete and runs locally. This change is about what has to be true before
it is reachable from the internet, and the constraints are mostly not about features.

The pinned dependency set is the binding one. `mediapipe==0.10.21`, `numpy==1.26.4`,
`tensorflow==2.16.2` are what the models were fitted under, and the parity suite exists to
prove the port reproduces the research numerically. Any dependency change here is subject to
that suite, and nine of its tests currently skip because the thesis archive is missing — so
the evidence available is weaker than it will look. That is stated plainly rather than
assumed away.

The other constraint is what this service does not have: no email. It cannot verify an
address, send a reset link, or notify anyone. Every decision about accounts below is shaped by
that, and the alternative — standing up email delivery — is deliberately out of scope.

## Goals / Non-Goals

**Goals**

- A climber's video and report are visible to that climber and nobody else.
- A stranger who finds the URL cannot make the server do expensive work without limit.
- A restart loses nothing and leaks nothing.
- The image is small enough to deploy on ordinary hosting, with the numbers unchanged.
- A climber knows what happens to their video before they upload it, and can remove
  everything — account included — when they want to.

**Non-goals**

- Email, and everything that depends on it: verification, reset, notifications.
- Sharing a report with a non-owner. A follow-up, once accounts exist.
- Multi-machine deployment. The queue stays in-process.
- Any change to the analysis itself. Not one line of `aced/` should need to move.

## Decisions

### Accounts: email and password, no verification

A climber registers with an email address and a password. The address is an identifier and a
login name; nothing is ever sent to it, and it is not verified. Passwords are hashed with
Argon2id and only the hash is stored.

*Why email as the login name:* it is the one identifier people reliably remember, and it
keeps the door open for verification and reset later without a migration. Not verifying it
now is an honest consequence of having no mail delivery, and it is cheap: an unverified
address costs the service nothing, since nothing is sent to it.

*Why Argon2id:* it is the current recommendation for password hashing, `argon2-cffi` is a
small, maintained dependency with no native build on the platforms this runs on, and there is
no reason to choose a weaker scheme on a service that will hold real passwords.

*What a forgotten password means:* there is no self-service recovery. The account's analyses
are not lost — an operator with database access can set a new password — but the climber
cannot do it alone. The sign-in screen says so rather than implying a reset flow that does not
exist. This is the cost of no email, and it is the one to revisit first if the service ever
gets it.

### Sessions in a cookie, stored server-side

Signing in creates a session row and sets an `HttpOnly`, `Secure`, `SameSite=Lax` cookie
carrying only the session id. Signing out deletes the row. Sessions expire.

*Why a cookie and not a bearer token:* a `<video>` element issues its own requests and cannot
be given a header. A cookie goes with every request the browser makes, which means the media
endpoints are protected by exactly the same mechanism as the JSON ones, and nothing sensitive
ever appears in a URL. This is the single biggest simplification accounts buy over
per-analysis tokens.

*Why server-side rather than signed:* sign-out and account deletion have to actually end the
session, not just ask the browser to forget it. A row can be deleted; a signed cookie is valid
until it expires wherever a copy exists.

*CSRF:* the API is same-origin with the web app and every state-changing route is a `POST` or
`DELETE` from script, never a form. `SameSite=Lax` keeps a cross-site page from riding the
cookie on those, and the multipart upload — the one request a cross-site form could
plausibly forge — additionally requires a custom header that a form cannot set.

### Ownership: analyses belong to the account that uploaded them

Every analysis carries an `owner_id`. Every analysis endpoint requires a session, loads the
record, and compares. A mismatch is answered exactly as an unknown id.

*Why 404 and not 403:* a 403 confirms the id exists. Given that ids are the addressing scheme,
an endpoint that distinguishes "not yours" from "no such analysis" is an oracle for enumerating
them. Answering identically costs nothing and closes it.

*A list endpoint:* `GET /analyses` returns the signed-in climber's own analyses, newest first.
Accounts make this possible for the first time — before, nothing could say whose anything was —
and it is what lets a climber find a report without having kept the link.

### Abuse limits: per account, plus a gate on the queue

Upload rate limiting is keyed on the account. Registration is keyed on client address and can
be closed entirely behind an invite code (`ACED_SIGNUP_CODE`). Independently of either, the
service refuses an upload when pending-plus-running analyses are at the configured maximum —
before storing anything.

*Why per account:* a per-address limit is crude behind a gym's wifi, where one address is
everybody. An account is the right unit, and accounts are what this change adds.

*Why the invite code:* open registration on a compute-expensive service means the abuse
vector moves from "upload a video" to "create an account, then upload a video". A code is the
smallest thing that closes that door, and leaving it unset opens it again for a deployment
that wants to be public.

*Why the queue gate is separate and in the application:* the limit that protects the machine
is not requests per second, it is analyses in flight — one upload occupies a worker for
minutes. Only the application knows the queue depth, and refusing before storage is what
keeps a rejected upload from leaving a file behind.

### SQLite, behind the interfaces that already exist

`AnalysisStore` is already a Protocol with an in-memory implementation; this adds a SQLite one
and two more stores beside it, for accounts and sessions, in the same file. No caller of the
analysis store changes.

*Why SQLite:* one file, no service to run, and it matches a deployment that is one container
on one machine. Postgres would be the answer for multiple workers; that is the change this one
deliberately does not make.

*Startup reconciliation:* an analysis found in `running` at startup is marked failed — the
worker that owned it died with the previous process, and nothing will resume it. Media with no
record is deleted. Both are the same principle the worker already follows: nothing is left in
a non-terminal state forever.

*Account deletion:* removes the account, its sessions, every analysis it owns, and all their
media, in that order of visibility — the record set that names media goes last, so a crash
mid-way leaves media that the next startup's reconciliation will find and remove.

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
the API. `Secure` cookies require it — a session cookie that the browser will only send over
HTTPS is only useful if there is HTTPS.

*Why Caddy:* automatic certificate management with three lines of configuration. nginx with
certbot is the same outcome with a renewal cron job to forget about.

## Risks / Trade-offs

- **A forgotten password needs an operator.** No email means no self-service reset. Stated on
  the sign-in screen, and the deployment note says how an operator does it.
- **Unverified email addresses.** Anyone can register as anyone. Nothing is sent to the
  address and nothing depends on it being real, so the consequence is only that it cannot be
  trusted as contact information — which it is not used as.
- **The parity evidence for the dependency swap is incomplete** while the archive is missing.
  Recorded, not hidden.
- **SQLite serializes writes.** With a handful of concurrent analyses this is irrelevant; it
  would not be with fifty, and that is the point at which the store changes again.
- **A registration wall is friction.** A climber who wanted a one-off report now creates an
  account first. That is the trade for their video being theirs alone, and the sign-up form is
  two fields.
- **Deploying at all creates an obligation** that running locally does not: video of
  identifiable people, on a machine, under someone's control. The consent step, the privacy
  notice and account deletion are part of this change for that reason, not as decoration.

## Migration Plan

1. Persistence first — accounts need it — with the analysis suite passing unchanged against
   SQLite.
2. Accounts, sessions and ownership, with tests. The service still runs locally throughout;
   `docker compose up api` gets a sign-in screen.
3. Limits and the invite code.
4. The dependency swap, **gated**: parity run and recorded before it is kept.
5. TLS, the production compose file, and the deployment note.
6. Consent, the privacy notice, and account deletion in the web app.

Step 4 is independent of everything else; if the swap fails parity, the rest still ships and
the image stays 7.6 GB.
