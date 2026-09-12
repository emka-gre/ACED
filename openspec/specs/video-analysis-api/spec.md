# video-analysis-api Specification

## Purpose
TBD - created by archiving change climbing-technique-analysis-app. Update Purpose after archive.

## Requirements

### Requirement: Video upload

The API SHALL accept a climbing video upload, validate its format and size before accepting it, store it, and create an analysis record identified by an opaque identifier.

#### Scenario: Accepted upload

- **WHEN** a client uploads a supported video file within the size limit
- **THEN** the API responds with an accepted status, the analysis identifier, and the location of the analysis resource
- **AND** the analysis record is created in the pending state

#### Scenario: Unsupported format

- **WHEN** a client uploads a file whose format is not supported
- **THEN** the API rejects the request with a client error naming the supported formats
- **AND** no analysis record is created

#### Scenario: Oversized upload

- **WHEN** a client uploads a file exceeding the configured size limit
- **THEN** the API rejects the request with a client error stating the limit
- **AND** the request is rejected without buffering the entire file in memory

### Requirement: Asynchronous analysis

Analysis SHALL run outside the request-response cycle. The upload response SHALL NOT wait for analysis to complete.

#### Scenario: Upload returns promptly

- **WHEN** a video is accepted for analysis
- **THEN** the API responds without waiting for pose extraction or inference to finish
- **AND** the analysis is enqueued for background processing

#### Scenario: Concurrent analyses

- **WHEN** several analyses are submitted at once
- **THEN** each is tracked independently by its own identifier
- **AND** the state of one does not affect the reported state of another

### Requirement: Analysis status and progress

The API SHALL expose an analysis resource reporting its state — pending, running, succeeded, or failed — and, while running, a progress indication.

#### Scenario: Analysis in progress

- **WHEN** a client requests an analysis that is being processed
- **THEN** the response reports the running state and a progress value between 0 and 1

#### Scenario: Analysis complete

- **WHEN** a client requests an analysis that has finished successfully
- **THEN** the response reports the succeeded state and includes the report

#### Scenario: Analysis failed

- **WHEN** processing fails
- **THEN** the response reports the failed state with a client-safe reason
- **AND** internal diagnostic details are logged but not returned to the client

#### Scenario: Unknown analysis

- **WHEN** a client requests an identifier that does not exist
- **THEN** the API responds with a not-found status

### Requirement: Report retrieval

The API SHALL serve the completed report as JSON conforming to the report structure, and SHALL NOT serve a partial report for an analysis that has not succeeded.

#### Scenario: Report served

- **WHEN** a client requests the report of a succeeded analysis
- **THEN** the full report is returned as JSON

#### Scenario: Report not yet available

- **WHEN** a client requests the report of a pending or running analysis
- **THEN** the API responds indicating the report is not yet available
- **AND** no partial report is returned

### Requirement: Media artifact access

The API SHALL serve the original video and, when generated, the annotated overlay video for a completed analysis, supporting range requests so that a client can seek within them.

#### Scenario: Seekable playback

- **WHEN** a client requests a byte range of an analysis video
- **THEN** the API serves the requested range with a partial-content status

#### Scenario: Overlay generated on request

- **WHEN** a client requests the overlay video for an analysis that has none
- **THEN** the API begins rendering it and reports the artifact as being prepared
- **AND** the artifact becomes retrievable once rendering completes

### Requirement: Failure isolation

A failure in processing one analysis SHALL NOT prevent the service from accepting or processing others, and SHALL leave the failed analysis in a terminal failed state rather than pending indefinitely.

#### Scenario: Worker crash during processing

- **WHEN** processing of an analysis terminates unexpectedly
- **THEN** that analysis is eventually reported as failed
- **AND** the service continues accepting and processing other analyses

### Requirement: Retention and deletion

Each analysis SHALL carry an expiry. Expired videos and rendered artifacts SHALL be deleted automatically, and a client SHALL be able to delete an analysis and its media before expiry.

#### Scenario: Client deletes an analysis

- **WHEN** a client deletes an analysis
- **THEN** the stored video, rendered artifacts, and report are removed
- **AND** subsequent requests for that analysis report it as not found

#### Scenario: Expired analysis cleaned up

- **WHEN** an analysis passes its expiry
- **THEN** its stored video and rendered artifacts are deleted by the cleanup process

### Requirement: Service health and model provenance

The API SHALL expose an endpoint reporting service health and the identity, version, and validation metrics of the loaded model bundle.

#### Scenario: Health queried

- **WHEN** the health endpoint is requested
- **THEN** the response reports service health and the loaded bundle's identifier and version

#### Scenario: Model bundle unavailable at startup

- **WHEN** the configured model bundle cannot be loaded
- **THEN** the service reports itself unhealthy
- **AND** it does not accept analyses it cannot process
