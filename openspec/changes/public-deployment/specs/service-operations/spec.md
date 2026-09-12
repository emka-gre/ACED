## ADDED Requirements

### Requirement: Production deployment

The service SHALL be deployable to a single host as one command, served over TLS, with
configuration and secrets supplied from outside the repository.

#### Scenario: Deployed with TLS

- **WHEN** the production compose file is brought up on a host with a public name
- **THEN** the service is reachable over HTTPS with a valid certificate
- **AND** plain HTTP redirects to it

#### Scenario: Secrets outside the repository

- **WHEN** the deployment is configured
- **THEN** every secret and every host-specific value comes from the environment or an
  untracked file
- **AND** the repository contains an example of that file with no real values in it

#### Scenario: Deployment documented

- **WHEN** an operator reads the deployment note
- **THEN** it says what host resources are needed, how to bring the service up, how to update
  it, and how to remove it and its stored data entirely

### Requirement: Image size

The service image SHALL be small enough to deploy on ordinary hosting, and any dependency
change made to achieve that SHALL be shown not to change the analysis.

#### Scenario: Dependency swap verified by parity

- **WHEN** a dependency is replaced by a lighter variant
- **THEN** the parity suite is run against the shipped default bundle before the change is
  kept
- **AND** the result, including which tests could not run and why, is recorded in the README

#### Scenario: Parity failure reverts the change

- **WHEN** the parity suite fails after a dependency change
- **THEN** the dependency change is reverted
- **AND** the image size is accepted as it was

### Requirement: Operational visibility

The service SHALL report enough about itself for an operator to know it is working, without
exposing anything about the people whose video it holds.

#### Scenario: Health reports limits and load

- **WHEN** the health endpoint is requested
- **THEN** it reports the configured limits and the number of analyses awaiting or in
  progress

#### Scenario: Logs carry no personal data

- **WHEN** the service logs an event about an analysis
- **THEN** the log carries the analysis identifier, sizes and states
- **AND** never the uploaded filename, a frame, a landmark, or an owner token
