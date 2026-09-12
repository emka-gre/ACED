## ADDED Requirements

### Requirement: Analysis ownership

An upload SHALL return an owner token alongside the analysis identifier, and every subsequent
request concerning that analysis SHALL require it. The service SHALL store only a hash of the
token, so that a compromised store does not yield access to the analyses it describes.

#### Scenario: Token issued at upload

- **WHEN** a video is accepted for analysis
- **THEN** the response carries an owner token as well as the analysis identifier
- **AND** the token is not recoverable from any later response

#### Scenario: Owner reads their analysis

- **WHEN** a client requests an analysis presenting its owner token
- **THEN** the request is served

#### Scenario: Request without the owner token

- **WHEN** a client requests an analysis without presenting its owner token, or presents a
  token that does not match
- **THEN** the API responds as it would for an identifier that does not exist
- **AND** the response does not distinguish a wrong token from an unknown analysis

#### Scenario: Media access by an owner

- **WHEN** a client requests the video or overlay of an analysis it owns
- **THEN** the request is served, including for a byte range
- **AND** the token may be presented in the request URL, because a media element cannot send
  request headers

#### Scenario: Deletion requires ownership

- **WHEN** a client attempts to delete an analysis without its owner token
- **THEN** the analysis is not deleted
- **AND** the response is the same as for an unknown analysis

### Requirement: Abuse limits

The service SHALL limit how much work an unauthenticated client can cause, by rate-limiting
uploads per client and by refusing new analyses when the queue is saturated.

#### Scenario: Upload rate exceeded

- **WHEN** a client uploads more videos in a period than the configured limit allows
- **THEN** the request is rejected with a client error stating when it may retry
- **AND** no analysis record is created and no video is stored

#### Scenario: Queue saturated

- **WHEN** an upload arrives while the number of analyses awaiting or in progress is at the
  configured maximum
- **THEN** the request is rejected with a message saying the service is busy
- **AND** the rejection happens before the upload is stored, not after

#### Scenario: Limits are configurable

- **WHEN** the service starts
- **THEN** the upload rate limit and the maximum queue depth are read from configuration
- **AND** their values appear in the health response so an operator can confirm what is
  running

### Requirement: Records survive a restart

Analysis records SHALL be persisted, and the service SHALL reconcile its state at startup so
that no analysis is left in a non-terminal state and no media is left with nothing that names
it.

#### Scenario: Analyses outlive a restart

- **WHEN** the service restarts
- **THEN** analyses created before the restart are still addressable
- **AND** their reports, media and expiries are unchanged

#### Scenario: Interrupted analysis is failed at startup

- **WHEN** the service starts and finds an analysis recorded as running
- **THEN** that analysis is marked failed with a client-safe reason
- **AND** it is not restarted, because the process that owned it is gone

#### Scenario: Orphaned media removed at startup

- **WHEN** the service starts and finds stored media that no record names
- **THEN** that media is deleted
