import type { ModelInfo } from '../api/types';

const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Accuracy',
  fault_precision: 'Of moves flagged, actually faults',
  fault_recall: 'Of actual faults, caught',
  fault_f1: 'F1',
  roc_auc: 'ROC AUC',
  validation_windows: 'Validation windows',
};

/**
 * What produced this report and what it is worth.
 *
 * Shown with every report rather than on a separate "about" page. The models were trained on
 * a small set of climbers in one gym and labelled by one person; a climber reading a verdict
 * about their technique is entitled to know that at the same moment, not a click away.
 */
export function ModelPanel({
  model,
  scope,
  limitations,
}: {
  model: ModelInfo;
  scope: string;
  limitations: string[];
}) {
  const metrics = Object.entries(model.metrics);

  return (
    <section className="model-panel">
      <h2>About this analysis</h2>
      <p>{scope}</p>

      <dl className="model-meta">
        <div>
          <dt>Model</dt>
          <dd>
            {model.bundle_id} <span className="muted">v{model.version}</span>
          </dd>
        </div>
        {model.trained_on && (
          <div>
            <dt>Trained</dt>
            <dd>{model.trained_on}</dd>
          </div>
        )}
      </dl>

      {metrics.length > 0 ? (
        <>
          <h3>Measured on held-out climbs</h3>
          <dl className="metrics">
            {metrics.map(([key, value]) => (
              <div key={key}>
                <dt>{METRIC_LABELS[key] ?? key.replace(/_/g, ' ')}</dt>
                <dd>{key === 'validation_windows' ? value : formatMetric(value)}</dd>
              </div>
            ))}
          </dl>
        </>
      ) : (
        <p className="muted">This model has not been evaluated on a held-out set.</p>
      )}

      <h3>Limitations</h3>
      <ul className="limitations">
        {limitations.map((limitation) => (
          <li key={limitation}>{limitation}</li>
        ))}
      </ul>
    </section>
  );
}

function formatMetric(value: number): string {
  return value >= 0 && value <= 1 ? `${Math.round(value * 100)}%` : String(value);
}
