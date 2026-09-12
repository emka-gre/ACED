## 1. Persistence

- [x] 1.1 `SqliteAnalysisStore` implementing `AnalysisStore`; records including the report serialised as JSON; one file under the storage root, schema created on first open
- [x] 1.2 Startup reconciliation: mark `running` records failed with a client-safe reason; delete media no record names
- [x] 1.3 Make SQLite the default store; keep the in-memory store for tests
- [x] 1.4 Tests: records survive a store reopen, interrupted analysis is failed at startup, orphaned media is removed, the whole API suite passes unchanged against the SQLite store

## 2. Accounts and sessions

- [ ] 2.1 Add `argon2-cffi`; account model (id, email, password hash, created); `AccountStore` and `SessionStore` on the same SQLite file
- [ ] 2.2 `POST /auth/register`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`; minimum password length enforced; wrong address and wrong password answer identically
- [ ] 2.3 Session cookie: `HttpOnly`, `Secure`, `SameSite=Lax`, session id only; server-side row with expiry; a `require_session` dependency for the rest of the API
- [ ] 2.4 Invite code: `ACED_SIGNUP_CODE`, required at registration when set
- [ ] 2.5 CSRF: state-changing routes require a custom header a cross-site form cannot set; the upload especially
- [ ] 2.6 Tests: register and sign in, duplicate address, wrong credentials indistinguishable, logout invalidates the cookie server-side, expired session rejected, invite code enforced, password never appears in a log or response, CSRF header required

## 3. Ownership

- [ ] 3.1 `owner_id` on the analysis record; set at upload from the session
- [ ] 3.2 Every analysis endpoint — status, report, video, overlay, delete — requires a session and an owner match; a mismatch answers exactly as an unknown id
- [ ] 3.3 `GET /analyses`: the signed-in account's analyses, newest first, with state, dates and summary line; no report bodies
- [ ] 3.4 `DELETE /auth/me`: remove the account, its sessions, every owned analysis and all media, records last
- [ ] 3.5 Tests: unauthenticated upload rejected with nothing stored, owner reads including media ranges, another account's analysis is not-found, listing shows only one's own, account deletion removes everything and frees the address

## 4. Abuse limits

- [ ] 4.1 Per-account upload rate limit with a configurable window, answering 429 with `Retry-After`, rejected before the body is read
- [ ] 4.2 Per-address registration rate limit
- [ ] 4.3 Queue-depth gate: refuse an upload when pending + running analyses are at the configured maximum, before storing anything
- [ ] 4.4 Surface the limits and the current queue depth in `/health`
- [ ] 4.5 Tests: each limit trips and resets, saturation refuses before storage, limits appear in health, a rejected upload leaves no record and no file

## 5. Image size — gated

- [ ] 5.1 Swap `tensorflow` → `tensorflow-cpu` and `opencv-python` → `opencv-python-headless` in `pyproject.toml`; regenerate the lock
- [ ] 5.2 Rebuild the image and record the size before and after
- [ ] 5.3 **Gate:** run `pytest -m parity` against the shipped default bundle; record the result in the README including which tests skipped and why. If anything that previously passed now fails, revert 5.1 and stop this section
- [ ] 5.4 Run the full fast suite and the end-to-end suite under the new dependencies

## 6. Deployment

- [ ] 6.1 `docker-compose.prod.yml`: the service plus a Caddy reverse proxy with automatic TLS, storage on a named volume, restart policy
- [ ] 6.2 `.env.example` with every configurable value and no real ones; the production compose reads from `.env`
- [ ] 6.3 `docs/deployment.md`: host requirements (RAM, disk, a DNS name), bring-up, update, backup of the storage volume, complete removal, and how an operator resets a forgotten password
- [ ] 6.4 Verify on a real host or a local Caddy with an internal CA that HTTPS is served, HTTP redirects, and the session cookie is only sent over HTTPS

## 7. Web app

- [ ] 7.1 Sign-in and registration screens; the invite-code field when the server requires it; the forgotten-password statement
- [ ] 7.2 Session handling: `GET /auth/me` on load; every request sends the cookie and the CSRF header; a 401 mid-visit returns to sign-in with a reason, never as an analysis error
- [ ] 7.3 My analyses: the list as the signed-in landing screen, newest first; opening one; another account's link explained rather than reported as absent
- [ ] 7.4 Upload screen: what is stored, the retention period, deletion available; an acknowledgement checkbox that gates the upload
- [ ] 7.5 Report view: state when this analysis and its video will be deleted automatically
- [ ] 7.6 Account deletion with a confirmation naming what goes; signed out afterwards
- [ ] 7.7 Tests: signed-out visitor sees sign-in, registration signs in, expired session returns to sign-in, list shows own analyses, upload blocked without acknowledgement, retention shown, account deletion confirms and signs out

## 8. Close

- [ ] 8.1 Update `docs/follow-ups.md`: tick authentication and persistence; note sharing and email as the natural next items
- [ ] 8.2 Update the README's data-and-retention section: analyses belong to accounts, account deletion, and the no-self-service-reset consequence, plainly
