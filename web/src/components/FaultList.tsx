import type { Fault } from '../api/types';
import { formatTime } from '../format';

interface Props {
  faults: Fault[];
  activeFault: number | null;
  onSelectFault: (index: number) => void;
}

/**
 * What the analysis found, in the climber's language.
 *
 * The explanation string is rendered as it arrives. It was written server-side from templates
 * that deliberately contain no feature identifiers and no statistical vocabulary, and there is
 * a test on that side asserting so — restating it here in different words would put a second,
 * unverified copy of that copy in the frontend.
 */
export function FaultList({ faults, activeFault, onSelectFault }: Props) {
  if (faults.length === 0) {
    return (
      <section className="findings">
        <h2>Findings</h2>
        <p className="no-faults">No arm-technique faults were detected in this climb.</p>
        <p className="hint">
          That is a statement about arm technique only — it is not a judgement of the climb as a
          whole.
        </p>
      </section>
    );
  }

  return (
    <section className="findings">
      <h2>
        {faults.length} {faults.length === 1 ? 'fault' : 'faults'}
      </h2>
      <ol className="fault-list">
        {faults.map((fault, index) => (
          <li key={`${fault.frame_start}-${fault.frame_end}`}>
            <button
              type="button"
              className={`fault${activeFault === index ? ' fault-active' : ''}`}
              onClick={() => onSelectFault(index)}
              aria-current={activeFault === index}
            >
              <span className="fault-time">
                {formatTime(fault.time_start)}–{formatTime(fault.time_end)}
              </span>
              <span className="fault-explanation">{fault.explanation}</span>
              <span className="fault-meta">
                {fault.confidence !== null && (
                  <span className="confidence">
                    {Math.round(fault.confidence * 100)}% confidence
                  </span>
                )}
                {!fault.attribution.reliable && (
                  <span className="caveat">no clean stretch in this climb to compare against</span>
                )}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
