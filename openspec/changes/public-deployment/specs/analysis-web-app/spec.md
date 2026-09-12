## ADDED Requirements

### Requirement: Token custody

The web app SHALL retain the owner token for each analysis it starts, use it for every
request concerning that analysis, and present a link containing it so the climber can return
to or deliberately share the analysis.

#### Scenario: Token retained after upload

- **WHEN** an upload succeeds
- **THEN** the app stores the analysis identifier and its owner token in the browser
- **AND** subsequent requests for that analysis carry the token

#### Scenario: Returning to an analysis

- **WHEN** a climber opens a link to an analysis that includes its token
- **THEN** the analysis is shown
- **AND** the token is stored so the plain link works on that browser afterwards

#### Scenario: Analysis opened without a token

- **WHEN** a climber opens a link to an analysis whose token the browser does not hold
- **THEN** the app states that this analysis cannot be opened from this browser without its
  original link
- **AND** it does not present this as the analysis being absent

#### Scenario: Shareable link offered

- **WHEN** an analysis has finished
- **THEN** the app offers a link containing the token, described as granting access to the
  video and report

### Requirement: Informed upload

The web app SHALL tell a climber what happens to an uploaded video — what is stored, for how
long, and how to remove it — before the upload begins, and SHALL require acknowledgement that
they may share the footage.

#### Scenario: Notice shown before upload

- **WHEN** the upload screen is shown
- **THEN** it states what is stored, the retention period, and that deletion is available at
  any time

#### Scenario: Acknowledgement required

- **WHEN** a climber has selected a video but not acknowledged that they may share footage of
  the people in it
- **THEN** the upload cannot be started

#### Scenario: Retention stated on the report

- **WHEN** a report is shown
- **THEN** it states when the analysis and its video will be deleted automatically
