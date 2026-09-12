# pose-extraction Specification

## Purpose
TBD - created by archiving change climbing-technique-analysis-app. Update Purpose after archive.

## Requirements

### Requirement: Per-frame landmark extraction from video

The system SHALL extract 2D skeletal landmarks from each frame of a climbing video using MediaPipe Pose, and SHALL emit results incrementally as frames are processed rather than only after the whole video is read.

#### Scenario: Video with a detectable climber

- **WHEN** a video containing a single visible climber is processed
- **THEN** the extractor yields one landmark record per frame in which a pose was detected
- **AND** each record carries the source frame index, the frame timestamp in seconds, and `x`, `y`, `z`, and `visibility` for each retained landmark

#### Scenario: Consumer reads results while extraction is in progress

- **WHEN** a caller iterates the extractor's output
- **THEN** landmark records become available as each frame is processed
- **AND** the extractor does not require the full video's landmarks to be held in memory at once

### Requirement: Retained landmark subset

The extractor SHALL retain exactly the nine landmarks used by the trained models — nose (0), left and right shoulder (11, 12), left and right elbow (13, 14), left and right wrist (15, 16), and left and right hip (23, 24) — and SHALL discard the remaining MediaPipe landmarks.

#### Scenario: Landmark subset is applied

- **WHEN** MediaPipe returns all 33 landmarks for a frame
- **THEN** the emitted record contains only the nine retained landmarks, addressable by anatomical name

### Requirement: Frames without a detected pose

The extractor SHALL omit frames in which no pose is detected, and SHALL preserve the original frame index on every emitted record so that gaps in the frame sequence remain detectable by downstream stages.

#### Scenario: Intermittent detection failure

- **WHEN** poses are detected on frames 0–40 and 55–120 but not on frames 41–54
- **THEN** the extractor emits records for frames 0–40 and 55–120 only
- **AND** the emitted frame indices show the gap between 40 and 55

#### Scenario: No pose detected anywhere

- **WHEN** no pose is detected in any frame of the video
- **THEN** the extractor completes without error and yields no records
- **AND** it reports that no climber was detected

### Requirement: Unreadable or unsupported video input

The extractor SHALL fail with a distinct, actionable error when the video cannot be opened or decoded, and SHALL NOT return an empty result that is indistinguishable from a video containing no climber.

#### Scenario: Corrupt or unsupported file

- **WHEN** extraction is attempted on a file that cannot be opened as video
- **THEN** the extractor raises an error identifying the file as unreadable
- **AND** the error is distinguishable from the no-pose-detected outcome

### Requirement: Video metadata

The extractor SHALL report the source video's frame rate, frame count, and dimensions, and SHALL substitute a documented default frame rate when the container reports a missing or non-positive value.

#### Scenario: Frame rate available

- **WHEN** a video reports 30 fps
- **THEN** the extractor reports a frame rate of 30 and derives frame timestamps from it

#### Scenario: Frame rate missing or invalid

- **WHEN** a video reports a frame rate of zero or none
- **THEN** the extractor uses the documented default of 30 fps
- **AND** flags that timestamps are approximate

### Requirement: Input quality signals

The extractor SHALL report signals that indicate degraded analysis conditions, including the proportion of frames with no detected pose and the mean landmark visibility across retained landmarks.

#### Scenario: Poor visibility recording

- **WHEN** a video yields detected poses whose mean landmark visibility falls below the configured threshold
- **THEN** the extractor reports a low-visibility quality signal alongside its results
