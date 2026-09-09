/**
 * The API's wire types, mirroring `aced/report/types.py` and `api/models.py`.
 *
 * Hand-written rather than generated. The report structure is a dozen small shapes that
 * change only when the analysis contract changes, and a generator would add a build step and
 * a schema file to keep in sync for no coverage the compiler does not already give. The
 * responsibility that comes with that: when a `to_dict` on the Python side gains a field,
 * this file is where it lands.
 *
 * Everything here is what a climber is allowed to see. There are no feature vectors, window
 * arrays or model outputs in the contract, and there should never be.
 */

export type AnalysisState = 'pending' | 'running' | 'succeeded' | 'failed';

export type ArtifactState = 'absent' | 'preparing' | 'ready' | 'failed';

export type PredictedClass = 'sound' | 'fault';

export type QualityFlagKind = 'incomplete_pose_detection' | 'low_visibility' | 'video_too_short';

export interface QualityFlag {
  kind: QualityFlagKind;
  message: string;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  baseline: number;
  deviation: number;
}

export interface Attribution {
  /** False when the climb had no sound window to compare against. */
  reliable: boolean;
  dominant: FeatureContribution | null;
  top_contributors: FeatureContribution[];
}

export interface Segment {
  index: number;
  frame_start: number;
  frame_end: number;
  time_start: number;
  time_end: number;
  predicted: PredictedClass;
  /** Null under a backend that emits no calibrated probability. */
  confidence: number | null;
  low_reliability: boolean;
}

export interface Fault {
  frame_start: number;
  frame_end: number;
  time_start: number;
  time_end: number;
  confidence: number | null;
  attribution: Attribution;
  explanation: string;
}

export interface Summary {
  duration_seconds: number;
  window_count: number;
  faulty_window_count: number;
  faulty_proportion: number;
  fault_count: number;
  message: string;
}

export interface VideoInfo {
  duration_seconds: number;
  frame_count: number;
  fps: number;
  /** The container reported no frame rate, so every timestamp is approximate. */
  fps_defaulted: boolean;
  width: number;
  height: number;
}

export interface ModelInfo {
  bundle_id: string;
  version: string;
  trained_on: string | null;
  metrics: Record<string, number>;
}

export interface AnalysisReport {
  video: VideoInfo;
  summary: Summary;
  segments: Segment[];
  faults: Fault[];
  quality_flags: QualityFlag[];
  model: ModelInfo;
  scope: string;
  limitations: string[];
}

export interface AnalysisStatus {
  id: string;
  state: AnalysisState;
  progress: number;
  created_at: string;
  expires_at: string;
  filename: string;
  overlay: ArtifactState;
  model?: { bundle_id: string; version: string | null };
  /** Present only on success — a pending analysis carries no report field at all. */
  report?: AnalysisReport;
  /** Present only on failure. Already client-safe; the server logs the detail. */
  failure_reason?: string;
}

export interface CreatedAnalysis {
  id: string;
  state: AnalysisState;
  location: string;
}

export interface Health {
  status: 'ok' | 'unhealthy';
  bundle: {
    bundle: string;
    version: string;
    model_kind: string;
    trained_on: string | null;
    metrics: Record<string, number>;
  } | null;
  reason?: string;
}
