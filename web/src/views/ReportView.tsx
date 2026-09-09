import { useCallback, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';

import { FaultList } from '../components/FaultList';
import { ModelPanel } from '../components/ModelPanel';
import { QualityFlags } from '../components/QualityFlags';
import { VideoPlayer } from '../components/VideoPlayer';
import type { AnalysisReport, ArtifactState } from '../api/types';
import { formatPercent, formatTime } from '../format';

interface Props {
  id: string;
  report: AnalysisReport;
  overlay: ArtifactState;
  deleteControl: ReactNode;
}

/**
 * The finished analysis.
 *
 * Order is the argument: quality warnings, then the summary, then the climb with its faults,
 * then what the model is and is not. A climber reading top to bottom learns how much to trust
 * this before they read what it says.
 */
export function ReportView({ id, report, overlay, deleteControl }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);

  const activeFault = useMemo(() => {
    const index = report.faults.findIndex(
      (fault) => currentTime >= fault.time_start && currentTime <= fault.time_end,
    );
    return index === -1 ? null : index;
  }, [currentTime, report.faults]);

  const selectFault = useCallback(
    (index: number) => {
      const fault = report.faults[index];
      const video = videoRef.current;
      if (!fault || !video) return;
      video.currentTime = fault.time_start;
      setCurrentTime(fault.time_start);
    },
    [report.faults],
  );

  const { summary, video } = report;

  return (
    <section className="report">
      <QualityFlags flags={report.quality_flags} />

      <header className="summary">
        <h1>{summary.message}</h1>
        <dl className="summary-figures">
          <div>
            <dt>Length</dt>
            <dd>{formatTime(summary.duration_seconds)}</dd>
          </div>
          <div>
            <dt>Windows analyzed</dt>
            <dd>{summary.window_count}</dd>
          </div>
          <div>
            <dt>Flagged</dt>
            <dd>{formatPercent(summary.faulty_proportion)}</dd>
          </div>
        </dl>
        {video.fps_defaulted && (
          <p className="warning">
            This video reported no frame rate, so every time shown here is approximate.
          </p>
        )}
      </header>

      <VideoPlayer
        id={id}
        faults={report.faults}
        duration={video.duration_seconds}
        overlay={overlay}
        videoRef={videoRef}
        activeFault={activeFault}
        onTimeUpdate={setCurrentTime}
        onSelectFault={selectFault}
      />

      {activeFault !== null && (
        <p className="now-playing" aria-live="polite">
          Playing fault {activeFault + 1}: {report.faults[activeFault]?.explanation}
        </p>
      )}

      <FaultList faults={report.faults} activeFault={activeFault} onSelectFault={selectFault} />

      <ModelPanel model={report.model} scope={report.scope} limitations={report.limitations} />

      <footer className="report-footer">{deleteControl}</footer>
    </section>
  );
}
