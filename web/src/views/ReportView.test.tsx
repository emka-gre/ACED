import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ReportView } from './ReportView';
import { fault, report } from '../test/fixtures';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, getAnalysis: vi.fn() };
});

function renderReport(overrides = {}) {
  return render(
    <ReportView
      id="abc123"
      report={report(overrides)}
      overlay="absent"
      deleteControl={<button type="button">Delete this analysis</button>}
    />,
  );
}

describe('ReportView', () => {
  beforeEach(() => {
    // jsdom has no media pipeline; the element's own state is all these tests need.
    vi.spyOn(window.HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
  });

  it('leads with the summary', () => {
    renderReport();

    expect(
      screen.getByRole('heading', { name: /Detected 2 arm-technique faults/ }),
    ).toBeInTheDocument();
  });

  it('lists each fault with its time range and explanation', () => {
    renderReport();

    expect(screen.getByText('0:02–0:04')).toBeInTheDocument();
    expect(
      screen.getByText('Your left arm stayed straight while you reached.'),
    ).toBeInTheDocument();
    expect(screen.getByText('0:10–0:12')).toBeInTheDocument();
  });

  it('shows confidence when the backend supplies one', () => {
    renderReport();
    expect(screen.getAllByText(/82% confidence/)[0]).toBeInTheDocument();
  });

  it('says nothing about confidence when the backend supplies none', () => {
    renderReport({ faults: [fault({ confidence: null })] });
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });

  it('flags a fault whose baseline was unreliable', () => {
    renderReport({
      faults: [
        fault({
          attribution: { reliable: false, dominant: null, top_contributors: [] },
        }),
      ],
    });

    expect(screen.getByText(/no clean stretch in this climb/i)).toBeInTheDocument();
  });

  describe('seeking', () => {
    it('selecting a fault moves playback to it', async () => {
      const user = userEvent.setup();
      const { container } = renderReport();
      const video = container.querySelector('video');
      expect(video).not.toBeNull();

      await user.click(screen.getByText('0:10–0:12'));

      expect(video?.currentTime).toBe(10);
    });

    it('the fault being played is called out by name', async () => {
      const user = userEvent.setup();
      renderReport();

      await user.click(screen.getByText('0:02–0:04'));

      expect(await screen.findByText(/playing fault 1/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /0:02–0:04/ })).toHaveAttribute(
        'aria-current',
        'true',
      );
    });

    it('nothing is called out between faults', () => {
      renderReport();
      expect(screen.queryByText(/playing fault/i)).not.toBeInTheDocument();
    });
  });

  describe('a climb with no faults', () => {
    it('says so explicitly rather than showing an empty list', () => {
      renderReport({
        faults: [],
        summary: {
          duration_seconds: 20,
          window_count: 28,
          faulty_window_count: 0,
          faulty_proportion: 0,
          fault_count: 0,
          message: 'No arm-technique faults were detected.',
        },
      });

      expect(
        screen.getByText('No arm-technique faults were detected in this climb.'),
      ).toBeInTheDocument();
      // An empty screen would leave a climber wondering whether it worked.
      expect(screen.queryByRole('list', { name: /fault/i })).not.toBeInTheDocument();
    });

    it('does not overstate what that means', () => {
      renderReport({ faults: [] });

      // Said in the findings themselves, not only in the model notes further down: "no
      // faults" must not be read as "nothing was wrong with the climb".
      const findings = screen.getByRole('heading', { name: 'Findings' }).parentElement;
      expect(findings).toHaveTextContent(/arm technique only/i);
    });
  });

  describe('what the report says about itself', () => {
    it('names the model and version with every report', () => {
      renderReport();

      expect(screen.getByText('lstm-arm-technique')).toBeInTheDocument();
      expect(screen.getByText('v1.0.0')).toBeInTheDocument();
    });

    it('shows the validation metrics', () => {
      renderReport();

      expect(screen.getByText('88%')).toBeInTheDocument();
      expect(screen.getByText(/of actual faults, caught/i)).toBeInTheDocument();
    });

    it('says plainly when a model has never been evaluated', () => {
      const unevaluated = report();
      unevaluated.model.metrics = {};
      render(<ReportView id="abc" report={unevaluated} overlay="absent" deleteControl={null} />);

      expect(screen.getByText(/has not been evaluated on a held-out set/i)).toBeInTheDocument();
    });

    it('lists the limitations', () => {
      renderReport();

      expect(screen.getByText('Only one climber is analyzed.')).toBeInTheDocument();
      expect(screen.getByText(/sensitive to camera angle/i)).toBeInTheDocument();
    });

    it('puts quality warnings above the findings', () => {
      renderReport({
        quality_flags: [
          {
            kind: 'low_visibility',
            message: 'The climber was hard to see for much of this climb.',
          },
        ],
      });

      const warnings = screen.getByRole('complementary', { name: /quality warnings/i });
      const findings = screen.getByRole('heading', { name: '2 faults' });
      expect(warnings.compareDocumentPosition(findings)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    });
  });
});
