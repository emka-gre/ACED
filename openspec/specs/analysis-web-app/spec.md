# analysis-web-app Specification

## Purpose
TBD - created by archiving change climbing-technique-analysis-app. Update Purpose after archive.

## Requirements

### Requirement: Video upload interface

The web app SHALL let a climber select or drag a video file, validate its format and size before uploading, and show upload progress.

#### Scenario: Upload started

- **WHEN** a climber selects a supported video
- **THEN** the upload begins and its progress is displayed

#### Scenario: Invalid file rejected before upload

- **WHEN** a climber selects a file of an unsupported format or exceeding the size limit
- **THEN** the app explains why the file cannot be used and does not start the upload

#### Scenario: Upload fails

- **WHEN** an upload fails due to a network or server error
- **THEN** the app reports the failure and offers to retry

### Requirement: Processing status

After upload, the web app SHALL display the analysis state and progress, and SHALL allow the climber to leave and return to a pending analysis without losing access to it.

#### Scenario: Analysis in progress

- **WHEN** an analysis is running
- **THEN** the app displays its progress and updates it as processing advances

#### Scenario: Returning to a pending analysis

- **WHEN** a climber reopens the address of an analysis that is still processing
- **THEN** the app resumes displaying its current state

#### Scenario: Analysis failed

- **WHEN** an analysis fails
- **THEN** the app explains what went wrong in plain language and offers to try another video

### Requirement: Report presentation

When an analysis succeeds, the web app SHALL present the summary, the detected faults with their times and plain-language explanations, and any quality flags.

#### Scenario: Report displayed

- **WHEN** an analysis succeeds
- **THEN** the app shows the overall summary and each detected fault with its time range and explanation

#### Scenario: No faults detected

- **WHEN** a report contains no faults
- **THEN** the app states that no arm-technique faults were detected rather than showing an empty list

#### Scenario: Quality flags surfaced

- **WHEN** a report carries quality flags
- **THEN** the app displays them prominently alongside the results
- **AND** states how they affect the reliability of the analysis

### Requirement: Seekable fault timeline

The web app SHALL play the analyzed video with a timeline marking the detected faults, and selecting a fault SHALL seek playback to the start of that fault.

#### Scenario: Fault selected

- **WHEN** a climber selects a detected fault
- **THEN** video playback seeks to the start of that fault's time range

#### Scenario: Playback reaches a fault

- **WHEN** playback enters a fault's time range
- **THEN** the app indicates which fault is currently being viewed

#### Scenario: Overlay toggled

- **WHEN** an annotated overlay video is available and the climber enables it
- **THEN** playback switches to the overlay while preserving the current playback position

### Requirement: Limitations communicated

The web app SHALL state that the analysis covers arm technique only, identify the model version used, and present results as observations rather than authoritative coaching instruction.

#### Scenario: Scope shown with results

- **WHEN** a report is displayed
- **THEN** the app states the scope of the analysis and the model version that produced it

### Requirement: Analysis deletion

The web app SHALL let a climber delete an analysis and its uploaded video, and SHALL confirm the action before performing it.

#### Scenario: Deletion confirmed

- **WHEN** a climber requests deletion and confirms
- **THEN** the analysis and its video are deleted and the climber is returned to the upload view

#### Scenario: Deletion cancelled

- **WHEN** a climber requests deletion and cancels the confirmation
- **THEN** nothing is deleted

### Requirement: Responsive layout

The web app SHALL be usable on a phone-sized screen, since climbers record and review climbs on their phones at the gym.

#### Scenario: Phone-sized viewport

- **WHEN** the app is opened on a phone-sized viewport
- **THEN** upload, status, video playback, and the fault list are all usable without horizontal scrolling
