# technique-features Specification

## Purpose
TBD - created by archiving change climbing-technique-analysis-app. Update Purpose after archive.

## Requirements

### Requirement: Biomechanical feature computation

The system SHALL compute eighteen geometric features from the retained 2D landmarks of a single frame: four joint angles (left and right elbow, left and right shoulder), four signed vertical offsets relative to the corresponding shoulder (left and right wrist, left and right elbow), and ten inter-joint distances.

#### Scenario: Features computed for a frame

- **WHEN** a frame's retained landmarks are supplied
- **THEN** the extractor returns all eighteen named features for that frame

#### Scenario: Angle convention

- **WHEN** the elbow angle is computed from shoulder, elbow, and wrist positions
- **THEN** the result is the angle at the elbow vertex expressed in degrees in the range 0–180
- **AND** a fully extended arm yields a value approaching 180

#### Scenario: Signed vertical offset convention

- **WHEN** a wrist-relative-to-shoulder offset is computed
- **THEN** the result is the signed difference of the vertical coordinates, preserving the sign convention of the trained models

### Requirement: Numerical parity with the validated pipeline

Computed feature values SHALL be numerically equivalent to those produced by the original research pipeline for the same landmark input, within floating-point tolerance, and feature names SHALL match exactly the names the trained models were fitted against.

#### Scenario: Parity against reference fixtures

- **WHEN** features are computed for a set of reference landmark frames with known expected values
- **THEN** every feature value matches the expected value within floating-point tolerance
- **AND** every feature name matches the expected name exactly

### Requirement: Degenerate geometry handling

The extractor SHALL produce a finite value for every feature, replacing values that are undefined or non-finite — arising from coincident landmarks, zero-length vectors, or out-of-domain angle arguments — with zero.

#### Scenario: Coincident landmarks

- **WHEN** two landmarks used for a distance or angle occupy the same position
- **THEN** the affected features are emitted as zero rather than as NaN or infinity
- **AND** no exception propagates to the caller

### Requirement: Model feature subset selection

The system SHALL select the ordered subset of features required by the active model bundle, and SHALL reject the input when a required feature is not among the computed features.

#### Scenario: Required subset present

- **WHEN** a bundle declares its eight selected features
- **THEN** the feature vector is assembled in the exact order the bundle declares

#### Scenario: Required feature unavailable

- **WHEN** a bundle declares a feature the extractor does not compute
- **THEN** loading fails with an error naming the missing feature
- **AND** the system does not substitute a default value

### Requirement: Contiguous-run segmentation

The system SHALL split a frame sequence into runs of consecutive frame indices, treating any gap in frame indices as a run boundary, and SHALL window each run independently.

#### Scenario: Sequence with a detection gap

- **WHEN** feature frames cover indices 0–40 and 55–120
- **THEN** two independent runs are windowed
- **AND** no window spans the gap between frame 40 and frame 55

### Requirement: Fixed-length overlapping windows

The system SHALL segment each run into windows of the length declared by the model bundle (60 frames), advancing by the declared stride (20 frames), and SHALL emit each window together with the frame index range it covers.

#### Scenario: Run longer than one window

- **WHEN** a run of 140 frames is windowed at length 60 and stride 20
- **THEN** windows begin at run offsets 0, 20, 40, 60, and 80
- **AND** each window covers exactly 60 frames

#### Scenario: Trailing frames not aligned to the stride

- **WHEN** a run's length leaves frames uncovered after the last stride-aligned window
- **THEN** one additional window ending at the run's final frame is emitted
- **AND** that window covers exactly 60 frames

#### Scenario: Run shorter than one window

- **WHEN** a run of 35 frames is windowed at length 60
- **THEN** a single window is emitted, zero-padded to 60 frames
- **AND** the window's reported frame range covers only the 35 real frames

#### Scenario: No windows produced

- **WHEN** a video yields no frames with detected poses
- **THEN** windowing produces no windows
- **AND** the caller is able to distinguish this from a windowing failure

### Requirement: Window flattening for flat-input models

The system SHALL provide a flattened representation of a window in row-major time-major order, with columns named `<feature>_t<index>`, matching the layout the trained flat-input models were fitted against.

#### Scenario: Flattened column layout

- **WHEN** a 60-frame window of 8 features is flattened
- **THEN** 480 columns are produced
- **AND** the column order groups all features of timestep 0 first, then all features of timestep 1, and so on

### Requirement: Incremental windowing

The windower SHALL accept feature frames one at a time and SHALL emit each window as soon as its final frame has been supplied, without requiring the full sequence in advance.

#### Scenario: Windows emitted during streaming

- **WHEN** feature frames are supplied one at a time to a windower configured at length 60 and stride 20
- **THEN** the first window is emitted upon supplying frame 60
- **AND** each subsequent window is emitted upon supplying its final frame

#### Scenario: Stream terminated

- **WHEN** the caller signals the end of the stream with frames buffered
- **THEN** the windower emits any final window required to cover the trailing frames, padded if necessary
