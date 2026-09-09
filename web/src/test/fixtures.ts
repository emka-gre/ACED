import type { AnalysisReport, AnalysisStatus, Fault } from '../api/types';

/** A fault with sensible defaults, so a test only states the part it cares about. */
export function fault(overrides: Partial<Fault> = {}): Fault {
  return {
    frame_start: 60,
    frame_end: 120,
    time_start: 2,
    time_end: 4,
    confidence: 0.82,
    explanation: 'Your left arm stayed straight while you reached.',
    attribution: {
      reliable: true,
      dominant: { feature: 'left_elbow_angle', value: 170, baseline: 120, deviation: 2.4 },
      top_contributors: [],
    },
    ...overrides,
  };
}

export function report(overrides: Partial<AnalysisReport> = {}): AnalysisReport {
  return {
    video: {
      duration_seconds: 20,
      frame_count: 600,
      fps: 30,
      fps_defaulted: false,
      width: 1080,
      height: 1920,
    },
    summary: {
      duration_seconds: 20,
      window_count: 28,
      faulty_window_count: 6,
      faulty_proportion: 0.214,
      fault_count: 2,
      message: 'Detected 2 arm-technique faults across 6 of 28 analyzed windows.',
    },
    segments: [],
    faults: [
      fault(),
      fault({
        frame_start: 300,
        frame_end: 360,
        time_start: 10,
        time_end: 12,
        explanation: 'Your right arm stayed straight through the move.',
      }),
    ],
    quality_flags: [],
    model: {
      bundle_id: 'lstm-arm-technique',
      version: '1.0.0',
      trained_on: '2026-04-07',
      metrics: { accuracy: 0.882, fault_recall: 0.91 },
    },
    scope: 'This analysis covers arm technique only.',
    limitations: ['Only one climber is analyzed.', '2D pose is sensitive to camera angle.'],
    ...overrides,
  };
}

export function status(overrides: Partial<AnalysisStatus> = {}): AnalysisStatus {
  return {
    id: 'abc123',
    state: 'succeeded',
    progress: 1,
    created_at: '2026-09-10T10:00:00+00:00',
    expires_at: '2026-09-11T10:00:00+00:00',
    filename: 'climb.mp4',
    overlay: 'absent',
    report: report(),
    ...overrides,
  };
}
