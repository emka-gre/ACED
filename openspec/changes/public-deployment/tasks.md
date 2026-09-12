## 1. Ownership

- [ ] 1.1 Generate an owner token at upload, store its hash on the record, and return it alongside the id
- [ ] 1.2 Require the token on every analysis endpoint: `Authorization: Bearer` for JSON, a query parameter for media
- [ ] 1.3 Answer an unowned or wrongly-owned request exactly as an unknown id — same status, same body
- [ ] 1.4 Tests: token issued, owner reads, wrong token and no token are indistinguishable from not-found, media by query parameter, deletion requires ownership, token hash never appears in any response

## 2. Abuse limits

- [ ] 2.1 Per-client upload rate limit with a configurable window, answering 429 with `Retry-After`, rejected before the body is read
- [ ] 2.2 Queue-depth gate: refuse an upload when pending + running analyses are at the configured maximum, before storing anything
- [ ] 2.3 Surface both limits and the current queue depth in `/health`
- [ ] 2.4 Tests: rate limit trips and resets, saturation refuses before storage, limits appear in health, a rejected upload leaves no record and no file

## 3. Persistence

- [ ] 3.1 `SqliteAnalysisStore` implementing `AnalysisStore`; records including the report serialised as JSON; one file under the storage root
- [ ] 3.2 Startup reconciliation: mark `running` records failed with a client-safe reason; delete media no record names
- [ ] 3.3 Make it the default store; keep the in-memory store for tests
- [ ] 3.4 Tests: records survive a store reopen, interrupted analysis is failed at startup, orphaned media is removed, the API suite passes unchanged against the SQLite store

## 4. Image size — gated

- [ ] 4.1 Swap `tensorflow` → `tensorflow-cpu` and `opencv-python` → `opencv-python-headless` in `pyproject.toml`; regenerate the lock
- [ ] 4.2 Rebuild the image and record the size before and after
- [ ] 4.3 **Gate:** run `pytest -m parity` against the shipped default bundle; record the result in the README including which tests skipped and why. If anything that previously passed now fails, revert 4.1 and stop this section
- [ ] 4.4 Run the full fast suite and the end-to-end suite under the new dependencies

## 5. Deployment

- [ ] 5.1 `docker-compose.prod.yml`: the service plus a Caddy reverse proxy with automatic TLS, storage on a named volume, restart policy
- [ ] 5.2 `.env.example` with every configurable value and no real ones; the production compose reads from `.env`
- [ ] 5.3 `docs/deployment.md`: host requirements (RAM, disk, a DNS name), bring-up, update, backup of the storage volume, complete removal
- [ ] 5.4 Verify on a real host or a local Caddy with an internal CA that HTTPS is served and HTTP redirects

## 6. Consent and notice

- [ ] 6.1 Upload screen: state what is stored, the retention period, and that deletion is available; an acknowledgement checkbox that gates the upload
- [ ] 6.2 Report view: state when this analysis and its video will be deleted automatically
- [ ] 6.3 Web app: keep the owner token in the browser keyed by analysis id; send it on every request; handle the no-token case with the explanation the spec requires, not "not found"
- [ ] 6.4 Web app: offer the full link including the token, described as granting access
- [ ] 6.5 Tests: upload blocked without acknowledgement, token stored and sent, token-bearing link opens and stores, no-token case explained, retention shown

## 7. Close

- [ ] 7.1 Update `docs/follow-ups.md`: tick authentication and persistence; note what remains
- [ ] 7.2 Update the README's data-and-retention section to describe ownership and the no-recovery-of-a-lost-token consequence plainly
