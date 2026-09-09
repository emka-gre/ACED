import type { QualityFlag } from '../api/types';

/**
 * Conditions that make the analysis less trustworthy.
 *
 * Shown above the findings, not tucked under them. A report built on a video where the
 * climber was barely detected is a weaker claim than one built on a clean recording, and the
 * climber has to see that before reading the faults, not after.
 */
export function QualityFlags({ flags }: { flags: QualityFlag[] }) {
  if (flags.length === 0) return null;

  return (
    <aside className="quality" aria-label="analysis quality warnings">
      <h2>Read this first</h2>
      <ul>
        {flags.map((flag) => (
          <li key={flag.kind}>{flag.message}</li>
        ))}
      </ul>
    </aside>
  );
}
