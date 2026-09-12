# technique-inference Specification

## Purpose
TBD - created by archiving change climbing-technique-analysis-app. Update Purpose after archive.

## Requirements

### Requirement: Self-contained model bundles

A model bundle SHALL contain the trained model, a manifest declaring the bundle version, model kind, ordered feature list, window length and stride, retained landmark subset, decision threshold, input shape expected by the model, scaling mode, confidence kind, training date, and validation metrics, and — when the manifest declares external scaling — the scaler fitted at training time. The system SHALL load all inference parameters from the bundle and SHALL NOT read training datasets at inference time.

#### Scenario: Bundle loaded

- **WHEN** a bundle directory is loaded
- **THEN** the model, the manifest, and any scaler the manifest declares are loaded together
- **AND** no training dataset file is read

#### Scenario: Incomplete bundle

- **WHEN** a bundle is missing its model, its manifest, or a scaler its manifest declares
- **THEN** loading fails with an error naming the missing component
- **AND** no partially initialized pipeline is returned

#### Scenario: Scaler is never fitted at inference

- **WHEN** any inference is performed
- **THEN** any scaler applies only its persisted fitted parameters
- **AND** no fitting operation occurs

### Requirement: Declared scaling mode

The manifest SHALL declare whether feature scaling is applied by the pipeline before the model receives its input or by the model artifact itself, and the system SHALL apply scaling exactly once accordingly. A bundle whose declared mode contradicts its contents SHALL fail to load.

#### Scenario: External scaling

- **WHEN** a bundle declares external scaling
- **THEN** the pipeline applies the bundle's fitted scaler to the feature values before adapting them to the model's input shape

#### Scenario: Internal scaling

- **WHEN** a bundle declares that the model artifact scales its own input
- **THEN** the pipeline supplies unscaled feature values
- **AND** no bundle-level scaler is applied

#### Scenario: Declaration contradicts contents

- **WHEN** a bundle declares external scaling but ships no scaler, or declares internal scaling and ships one
- **THEN** loading fails with an error describing the contradiction

### Requirement: Feature schema validation at load

The system SHALL verify at load time that the feature names and ordering the extractor produces match the bundle's declared feature list, and SHALL refuse to run on mismatch rather than reordering, truncating, or zero-filling columns.

#### Scenario: Schema matches

- **WHEN** the extractor's feature names and order equal the bundle's declared list
- **THEN** the pipeline initializes successfully

#### Scenario: Schema mismatch

- **WHEN** the extractor's features differ from the bundle's declared list in name, count, or order
- **THEN** loading fails with an error describing the difference
- **AND** no predictions are produced

### Requirement: Window classification

The system SHALL classify each window as technically sound or faulty and SHALL return, for every window, its predicted class, the frame range it covers, and a confidence whose kind the bundle declares.

#### Scenario: Windows classified

- **WHEN** a sequence of windows is submitted for inference
- **THEN** exactly one result is returned per window, in input order
- **AND** each result carries a class and a frame range

#### Scenario: Probability-producing backend

- **WHEN** the bundle declares a probability confidence kind
- **THEN** each result carries a confidence in the range 0 to 1
- **AND** the confidence is the model's own probability output, not a rescaled substitute

#### Scenario: Backend without calibrated probability

- **WHEN** the bundle declares that its model produces no calibrated probability
- **THEN** results carry no probability-valued confidence
- **AND** the absence is expressed explicitly rather than as a fabricated or defaulted number

#### Scenario: Threshold from the bundle

- **WHEN** a probability-producing model returns a score at or above the bundle's declared threshold
- **THEN** the window is classified as technically sound
- **AND** a score below the threshold classifies the window as faulty

#### Scenario: No windows to classify

- **WHEN** an empty sequence of windows is submitted
- **THEN** an empty result set is returned without error

### Requirement: Explicit class semantics

Predicted classes SHALL be represented by named values rather than bare integers throughout the system's interfaces, while preserving the trained models' underlying encoding in which the positive class denotes sound technique.

#### Scenario: Named classes at interface boundaries

- **WHEN** a classification result crosses a module, API, or report boundary
- **THEN** the class is expressed as a named value such as `sound` or `fault`
- **AND** the mapping to the model's numeric encoding is applied only at the model adapter

### Requirement: Multiple model backends

The system SHALL support both sequence-input and flat-input model backends, selecting the input adaptation from the bundle manifest, and SHALL expose an identical result interface regardless of backend.

#### Scenario: Sequence-input backend

- **WHEN** a bundle declares a sequence-input model
- **THEN** scaled windows are supplied with time and feature axes preserved

#### Scenario: Flat-input backend

- **WHEN** a bundle declares a flat-input model
- **THEN** windows are supplied in the flattened column layout the model was fitted against, scaled or unscaled according to the bundle's declared scaling mode

#### Scenario: Backend is transparent to callers

- **WHEN** the same video is analyzed with either backend
- **THEN** the shape and type of the returned results are identical
- **AND** fields the backend cannot populate, such as a calibrated confidence, are absent rather than differently typed

### Requirement: Frame-level attribution from overlapping windows

The system SHALL assign a class to each analyzed frame by majority vote across all windows covering that frame, resolving ties in favor of sound technique, and SHALL exclude padding positions from voting.

#### Scenario: Frame covered by several windows

- **WHEN** a frame is covered by three windows classified sound, sound, and faulty
- **THEN** the frame is assigned the sound class

#### Scenario: Tied vote

- **WHEN** a frame is covered by an equal number of sound and faulty windows
- **THEN** the frame is assigned the sound class

#### Scenario: Padding excluded

- **WHEN** a window contains zero-padded positions beyond the run's real frames
- **THEN** those positions contribute no votes to any frame

### Requirement: Inference parity with the validated pipeline

For a given reference video, the system's window classifications SHALL match those of the original research pipeline using the same model and correctly fitted scaler.

#### Scenario: Reference video parity

- **WHEN** the reference video is analyzed by the packaged pipeline
- **THEN** the per-window classifications equal the reference classifications recorded from the validated pipeline

### Requirement: Bundle provenance is retrievable

The system SHALL expose the active bundle's version, model kind, training date, and validation metrics so that they can be surfaced to users and recorded with each analysis.

#### Scenario: Provenance reported

- **WHEN** an analysis completes
- **THEN** the identifier and version of the bundle that produced it are recorded with the result
