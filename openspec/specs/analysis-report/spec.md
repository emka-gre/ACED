# analysis-report Specification

## Purpose
TBD - created by archiving change climbing-technique-analysis-app. Update Purpose after archive.

## Requirements

### Requirement: Report structure

The system SHALL produce, for each analyzed video, a report containing video metadata, an overall summary, the per-window classifications, a list of detected faults, quality flags, and the identity and version of the model bundle used. The report SHALL be serializable to JSON and SHALL NOT expose raw feature vectors or model outputs.

#### Scenario: Report generated

- **WHEN** an analysis completes successfully
- **THEN** a report containing all required sections is produced
- **AND** it serializes to JSON without loss

#### Scenario: Model internals not exposed

- **WHEN** a report is serialized
- **THEN** it contains no scaled feature values, model weights, or backend-specific fields

### Requirement: Overall summary

The report SHALL summarize the climb with the analyzed duration, the number of windows evaluated, the proportion classified as faulty, and the number of distinct faults detected.

#### Scenario: Summary reflects results

- **WHEN** a climb yields 20 windows of which 6 are classified faulty and these merge into 2 distinct faults
- **THEN** the summary reports 20 windows, a faulty proportion of 0.3, and 2 faults

### Requirement: Temporal fault localization

The system SHALL merge consecutive or overlapping faulty windows into distinct faults, and SHALL report each fault's start and end in both frame indices and seconds, together with a confidence derived from its constituent windows when the active backend produces one.

#### Scenario: Adjacent faulty windows merged

- **WHEN** three overlapping windows covering frames 100–160, 120–180, and 140–200 are all classified faulty
- **THEN** a single fault spanning frames 100–200 is reported
- **AND** its time range is derived from the video's frame rate

#### Scenario: Separated faults kept distinct

- **WHEN** faulty windows occur at frames 60–120 and again at frames 400–460 with sound windows between them
- **THEN** two distinct faults are reported

#### Scenario: No faults detected

- **WHEN** every window is classified as technically sound
- **THEN** the report contains an empty fault list
- **AND** the summary states that no arm-technique faults were detected

### Requirement: Fault attribution

For each fault, the system SHALL identify the features that deviate most from the climber's technically sound windows in the same video, reporting the dominant feature, its value, the sound-window baseline value, its deviation, and the ranked top contributing features.

#### Scenario: Dominant feature identified

- **WHEN** a fault's aggregated features are compared against the baseline from sound windows
- **THEN** the feature with the largest absolute standardized deviation is reported as dominant
- **AND** the top three contributing features are reported in ranked order

#### Scenario: Insufficient sound baseline

- **WHEN** a video contains no windows classified as technically sound
- **THEN** the report states that no within-climb baseline was available
- **AND** attributions are marked as low-reliability rather than presented as confident

### Requirement: Plain-language explanations

Each fault SHALL carry an explanation in language addressed to a climber, naming the body part, describing the deviation, and stating when it occurred, without exposing internal feature identifiers or statistical terms.

#### Scenario: Explanation rendered

- **WHEN** a fault's dominant feature is the left elbow angle with a value well below the sound baseline
- **THEN** the explanation names the left arm, describes it as bent rather than extended, and states the time range in which it occurred

#### Scenario: No internal terminology leaks

- **WHEN** any explanation is rendered
- **THEN** it contains no feature identifiers, no statistical measure names, and no model terminology

### Requirement: Quality flags

The report SHALL carry flags describing conditions that degrade confidence, including a high proportion of frames with no detected pose, low mean landmark visibility, and a video too short to fill one analysis window.

#### Scenario: Degraded input flagged

- **WHEN** more than the configured proportion of frames yielded no detected pose
- **THEN** the report carries a flag identifying incomplete pose detection
- **AND** the flag states how it affects the reliability of the results

#### Scenario: Video too short

- **WHEN** a video yields fewer frames than one analysis window
- **THEN** the report carries a flag stating the video was too short for reliable analysis
- **AND** any results produced from the padded window are marked low-reliability

### Requirement: Model limitations disclosed

The report SHALL state the scope of what the model detects and the conditions under which its judgements are less reliable, and SHALL identify the model bundle version that produced it.

#### Scenario: Scope stated

- **WHEN** a report is presented to a user
- **THEN** it states that the analysis covers arm technique only
- **AND** it identifies the model version and notes the camera-angle sensitivity of 2D pose analysis
- **AND** it states that only one climber is analyzed and that footage containing several people is not supported

### Requirement: Annotated video artifacts

The system SHALL be able to render a skeleton-overlay video of the analyzed climb in which each frame's skeleton is colored by its assigned class, and SHALL generate this artifact on request rather than as a mandatory part of every analysis.

#### Scenario: Overlay requested

- **WHEN** an overlay video is requested for a completed analysis
- **THEN** a video is produced with the skeleton drawn over the original footage
- **AND** frames assigned the faulty class are visually distinguished from sound frames

#### Scenario: Overlay not requested

- **WHEN** an analysis completes and no overlay has been requested
- **THEN** no overlay rendering is performed

### Requirement: Headless analysis command

The system SHALL provide a command-line interface that analyzes a local video file and writes the same report structure the service produces.

#### Scenario: CLI analysis

- **WHEN** the analyze command is run against a local video file
- **THEN** the report is written as JSON
- **AND** its content is equivalent to the report the service would produce for the same video and bundle

#### Scenario: CLI failure reporting

- **WHEN** the analyze command is run against an unreadable file
- **THEN** the command exits with a non-zero status and a message identifying the problem
