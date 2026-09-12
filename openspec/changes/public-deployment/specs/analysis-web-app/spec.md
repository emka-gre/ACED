## ADDED Requirements

### Requirement: Sign-in

The web app SHALL require a signed-in climber before showing anything but the sign-in and
registration screens, and SHALL keep them signed in across visits until they sign out or the
session expires.

#### Scenario: Signed-out visitor

- **WHEN** a visitor with no session opens the app
- **THEN** they are shown sign-in, with a way to register

#### Scenario: Registration

- **WHEN** a visitor registers with an address and a password
- **THEN** they are signed in and taken to the upload screen
- **AND** if an invite code is required, the form asks for it

#### Scenario: Forgotten password

- **WHEN** a climber looks for a way to reset a forgotten password
- **THEN** the sign-in screen states plainly that there is no self-service reset and what to
  do instead

#### Scenario: Session expired mid-visit

- **WHEN** a request fails because the session is no longer valid
- **THEN** the app returns the climber to sign-in with a message saying why
- **AND** does not present it as an error in the analysis

### Requirement: My analyses

The web app SHALL show a signed-in climber their own analyses, so a report can be found again
without having kept its link.

#### Scenario: Listing

- **WHEN** a signed-in climber opens the app
- **THEN** their analyses are listed newest first, with state, date, and a summary line for
  each that has finished
- **AND** selecting one opens it

#### Scenario: Another account's link

- **WHEN** a signed-in climber opens a link to an analysis they do not own
- **THEN** the app states that there is no analysis at this link for this account
- **AND** offers their own list

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

### Requirement: Account deletion

The web app SHALL let a climber delete their account, after confirming what that removes.

#### Scenario: Deleting the account

- **WHEN** a climber chooses to delete their account and confirms
- **THEN** the account and every analysis under it are removed
- **AND** the climber is signed out and returned to the sign-in screen
