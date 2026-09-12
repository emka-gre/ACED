## ADDED Requirements

### Requirement: Accounts and sessions

The API SHALL let a climber register with an email address and a password, sign in to obtain
a session, and sign out. Passwords SHALL be stored only as a modern password hash, and a
session SHALL be carried in a cookie that scripts cannot read and that the browser sends only
over HTTPS.

#### Scenario: Registration

- **WHEN** a client registers with an email address and a password that meets the minimum
  length
- **THEN** an account is created and the client is signed in
- **AND** the password is never stored or logged in clear

#### Scenario: Duplicate registration

- **WHEN** a client registers with an address that already has an account
- **THEN** the request is rejected with a client error
- **AND** the response does not reveal anything else about the existing account

#### Scenario: Sign-in

- **WHEN** a client signs in with a matching address and password
- **THEN** a session is created and set as an `HttpOnly`, `Secure`, `SameSite` cookie

#### Scenario: Wrong credentials

- **WHEN** a client signs in with an address that does not exist or a password that does not
  match
- **THEN** the request is rejected with the same status and message in either case

#### Scenario: Sign-out ends the session

- **WHEN** a signed-in client signs out
- **THEN** the session is removed on the server
- **AND** a later request presenting the same cookie is treated as unauthenticated

#### Scenario: Sessions expire

- **WHEN** a session is older than the configured lifetime
- **THEN** it is no longer accepted

#### Scenario: Registration closed by invite code

- **WHEN** an invite code is configured and a registration does not present it
- **THEN** the registration is rejected
- **AND** no account is created

### Requirement: Analysis ownership

Every analysis SHALL belong to the account that uploaded it. Every request concerning an
analysis SHALL require a signed-in owner, and a request from anyone else SHALL be answered as
if the analysis does not exist.

#### Scenario: Upload requires sign-in

- **WHEN** an unauthenticated client uploads a video
- **THEN** the request is rejected as unauthenticated
- **AND** no analysis record is created and no video is stored

#### Scenario: Owner reads their analysis

- **WHEN** a signed-in client requests an analysis it uploaded
- **THEN** the request is served, including the video and overlay with byte ranges

#### Scenario: Another account's analysis

- **WHEN** a signed-in client requests an analysis uploaded by a different account
- **THEN** the API responds exactly as it would for an identifier that does not exist

#### Scenario: Unauthenticated read of an analysis

- **WHEN** an unauthenticated client requests any analysis resource
- **THEN** the request is rejected as unauthenticated
- **AND** the response does not indicate whether the identifier exists

#### Scenario: Listing one's own analyses

- **WHEN** a signed-in client lists analyses
- **THEN** it receives only the analyses it uploaded, newest first, with state and summary
- **AND** never a report body, a video, or anything belonging to another account

#### Scenario: Deletion requires ownership

- **WHEN** a client attempts to delete an analysis it does not own
- **THEN** the analysis is not deleted
- **AND** the response is the same as for an unknown analysis

### Requirement: Account deletion

A signed-in climber SHALL be able to delete their account, and doing so SHALL remove every
analysis they own together with its media.

#### Scenario: Account removed with everything under it

- **WHEN** a signed-in client deletes their account
- **THEN** the account, its sessions, its analyses, and all their stored media are removed
- **AND** the address can be registered again afterwards

### Requirement: Abuse limits

The service SHALL limit how much work a client can cause: uploads rate-limited per account,
registrations rate-limited per client address, and new analyses refused when the queue is
saturated.

#### Scenario: Upload rate exceeded

- **WHEN** an account uploads more videos in a period than the configured limit allows
- **THEN** the request is rejected with a client error stating when it may retry
- **AND** no analysis record is created and no video is stored

#### Scenario: Registration rate exceeded

- **WHEN** a client address registers more accounts in a period than the configured limit
- **THEN** further registrations from it are rejected until the period passes

#### Scenario: Queue saturated

- **WHEN** an upload arrives while the number of analyses awaiting or in progress is at the
  configured maximum
- **THEN** the request is rejected with a message saying the service is busy
- **AND** the rejection happens before the upload is stored, not after

#### Scenario: Limits are visible

- **WHEN** the health endpoint is requested
- **THEN** it reports the configured limits and the current queue depth

### Requirement: Records survive a restart

Accounts, sessions and analysis records SHALL be persisted, and the service SHALL reconcile
its state at startup so that no analysis is left in a non-terminal state and no media is left
with nothing that names it.

#### Scenario: State outlives a restart

- **WHEN** the service restarts
- **THEN** accounts can still sign in, unexpired sessions are still valid, and analyses
  created before the restart are still addressable with their reports and media

#### Scenario: Interrupted analysis is failed at startup

- **WHEN** the service starts and finds an analysis recorded as running
- **THEN** that analysis is marked failed with a client-safe reason
- **AND** it is not restarted, because the process that owned it is gone

#### Scenario: Orphaned media removed at startup

- **WHEN** the service starts and finds stored media that no record names
- **THEN** that media is deleted
