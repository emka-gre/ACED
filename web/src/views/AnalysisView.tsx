import { useState } from 'react';

import { deleteAnalysis } from '../api/client';
import { useAnalysis } from '../usePolling';
import { ReportView } from './ReportView';

interface Props {
  id: string;
  onDeleted: () => void;
}

/**
 * One analysis, from "still working" through to its report.
 *
 * The three terminal outcomes get three different screens, deliberately. A failed analysis
 * rendered as an empty report reads as "your climbing was fine".
 */
export function AnalysisView({ id, onDeleted }: Props) {
  const { status, error, loading, notFound } = useAnalysis(id);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  if (loading) {
    return (
      <section className="panel">
        <p className="muted">Loading…</p>
      </section>
    );
  }

  if (notFound) {
    return (
      <section className="panel">
        <h1>Not found</h1>
        <p>
          There is no analysis at this link. It may have been deleted, or it may have passed its
          retention window — analyses and their videos are removed automatically.
        </p>
        <button type="button" className="primary" onClick={onDeleted}>
          Analyze a climb
        </button>
      </section>
    );
  }

  if (!status) {
    return (
      <section className="panel">
        <p className="error" role="alert">
          {error ?? 'Something went wrong.'}
        </p>
      </section>
    );
  }

  const remove = async () => {
    setDeleting(true);
    try {
      await deleteAnalysis(id);
      onDeleted();
    } finally {
      setDeleting(false);
    }
  };

  const deleteControl = confirmingDelete ? (
    <div className="confirm" role="alertdialog" aria-label="confirm deletion">
      <p>Delete this analysis, its video, and any rendered overlay? This cannot be undone.</p>
      <button type="button" className="danger" disabled={deleting} onClick={() => void remove()}>
        {deleting ? 'Deleting…' : 'Delete permanently'}
      </button>
      <button type="button" onClick={() => setConfirmingDelete(false)}>
        Keep it
      </button>
    </div>
  ) : (
    <button type="button" className="quiet" onClick={() => setConfirmingDelete(true)}>
      Delete this analysis
    </button>
  );

  if (status.state === 'pending' || status.state === 'running') {
    return (
      <section className="panel">
        <h1>Analyzing</h1>
        <p className="muted">{status.filename}</p>
        <progress value={status.progress} max={1} aria-label="analysis progress" />
        <p aria-live="polite">
          {status.state === 'pending'
            ? 'Waiting to start…'
            : `Working through the video… ${Math.round(status.progress * 100)}%`}
        </p>
        <p className="hint">
          This takes a minute or two. The link to this page is stable — you can close the tab and
          come back to it.
        </p>
        {error && <p className="warning">Lost contact with the service; still trying…</p>}
        {deleteControl}
      </section>
    );
  }

  if (status.state === 'failed') {
    return (
      <section className="panel">
        <h1>Analysis failed</h1>
        <p className="error" role="alert">
          {status.failure_reason ?? 'The analysis failed.'}
        </p>
        <p className="hint">Nothing was analyzed, so there is no report to show.</p>
        <button type="button" className="primary" onClick={onDeleted}>
          Try another video
        </button>
        {deleteControl}
      </section>
    );
  }

  if (!status.report) {
    // Succeeded without a report should be impossible; saying so is better than rendering an
    // empty report that reads as a clean climb.
    return (
      <section className="panel">
        <p className="error" role="alert">
          The analysis finished but its report is missing.
        </p>
      </section>
    );
  }

  return (
    <ReportView
      id={id}
      report={status.report}
      overlay={status.overlay}
      deleteControl={deleteControl}
    />
  );
}
