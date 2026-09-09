import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AnalysisView } from './AnalysisView';
import { ApiError } from '../api/client';
import { report, status } from '../test/fixtures';

const getAnalysis = vi.hoisted(() => vi.fn());
const deleteAnalysis = vi.hoisted(() => vi.fn());

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, getAnalysis, deleteAnalysis };
});

describe('AnalysisView', () => {
  beforeEach(() => {
    getAnalysis.mockReset();
    deleteAnalysis.mockReset().mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('polls a running analysis until it succeeds', async () => {
    vi.useFakeTimers();
    getAnalysis
      .mockResolvedValueOnce(status({ state: 'pending', progress: 0 }))
      .mockResolvedValueOnce(status({ state: 'running', progress: 0.4 }))
      .mockResolvedValue(status({ state: 'succeeded', progress: 1 }));

    render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);

    await vi.waitFor(() => expect(screen.getByText(/waiting to start/i)).toBeInTheDocument());

    await vi.advanceTimersByTimeAsync(1500);
    await vi.waitFor(() => expect(screen.getByText(/40%/)).toBeInTheDocument());

    await vi.advanceTimersByTimeAsync(1500);
    await vi.waitFor(() =>
      expect(screen.getByText(/Detected 2 arm-technique faults/)).toBeInTheDocument(),
    );
  });

  it('stops polling once the analysis is terminal', async () => {
    vi.useFakeTimers();
    getAnalysis.mockResolvedValue(status({ state: 'succeeded' }));

    render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);
    await vi.waitFor(() => expect(getAnalysis).toHaveBeenCalledTimes(1));

    await vi.advanceTimersByTimeAsync(10_000);

    // A finished analysis does not change, and a client that keeps asking is a client that
    // hammers the service for the whole time the tab is left open.
    expect(getAnalysis).toHaveBeenCalledTimes(1);
  });

  it('shows the progress it is told, not an invented one', async () => {
    getAnalysis.mockResolvedValue(status({ state: 'running', progress: 0.7 }));

    render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);

    expect(await screen.findByLabelText('analysis progress')).toHaveValue(0.7);
  });

  it('presents a failure as a failure, not as an empty report', async () => {
    // A failed analysis carries no report at all — not a null one.
    const failed = status({ state: 'failed', failure_reason: 'The video could not be read.' });
    delete failed.report;
    getAnalysis.mockResolvedValue(failed);

    render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);

    expect(await screen.findByRole('heading', { name: /analysis failed/i })).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('The video could not be read.');
    // The distinction that matters: nothing here may read as "your climbing was fine".
    expect(screen.queryByText(/no arm-technique faults/i)).not.toBeInTheDocument();
  });

  it('explains a missing analysis rather than showing an error', async () => {
    getAnalysis.mockRejectedValue(new ApiError('no such analysis', 404));

    render(<AnalysisView id="gone" onDeleted={vi.fn()} />);

    expect(await screen.findByRole('heading', { name: /not found/i })).toBeInTheDocument();
    expect(screen.getByText(/retention window/i)).toBeInTheDocument();
  });

  it('keeps trying after a transient network failure', async () => {
    vi.useFakeTimers();
    getAnalysis
      .mockRejectedValueOnce(new Error('network down'))
      .mockResolvedValue(status({ state: 'succeeded' }));

    render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);
    await vi.waitFor(() => expect(getAnalysis).toHaveBeenCalledTimes(1));

    await vi.advanceTimersByTimeAsync(3000);

    // The analysis is still running on the server; giving up would strand the climber.
    await vi.waitFor(() => expect(getAnalysis).toHaveBeenCalledTimes(2));
  });

  describe('deletion', () => {
    it('asks before deleting anything', async () => {
      const user = userEvent.setup();
      getAnalysis.mockResolvedValue(status());
      render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);

      await user.click(await screen.findByRole('button', { name: /delete this analysis/i }));

      expect(screen.getByRole('alertdialog')).toHaveTextContent(/cannot be undone/i);
      expect(deleteAnalysis).not.toHaveBeenCalled();
    });

    it('deletes once confirmed and leaves the analysis', async () => {
      const user = userEvent.setup();
      const onDeleted = vi.fn();
      getAnalysis.mockResolvedValue(status());
      render(<AnalysisView id="abc123" onDeleted={onDeleted} />);

      await user.click(await screen.findByRole('button', { name: /delete this analysis/i }));
      await user.click(screen.getByRole('button', { name: /delete permanently/i }));

      await waitFor(() => expect(deleteAnalysis).toHaveBeenCalledWith('abc123'));
      await waitFor(() => expect(onDeleted).toHaveBeenCalled());
    });

    it('backs out without deleting', async () => {
      const user = userEvent.setup();
      getAnalysis.mockResolvedValue(status());
      render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);

      await user.click(await screen.findByRole('button', { name: /delete this analysis/i }));
      await user.click(screen.getByRole('button', { name: /keep it/i }));

      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
      expect(deleteAnalysis).not.toHaveBeenCalled();
    });

    it('can be deleted while it is still running', async () => {
      const user = userEvent.setup();
      const running = status({ state: 'running', progress: 0.3 });
      delete running.report;
      getAnalysis.mockResolvedValue(running);
      render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);

      await user.click(await screen.findByRole('button', { name: /delete this analysis/i }));

      expect(screen.getByRole('alertdialog')).toBeInTheDocument();
    });
  });

  it('warns about approximate times when the container reported no frame rate', async () => {
    const approximate = report();
    approximate.video.fps_defaulted = true;
    getAnalysis.mockResolvedValue(status({ report: approximate }));

    render(<AnalysisView id="abc123" onDeleted={vi.fn()} />);

    expect(await screen.findByText(/every time shown here is approximate/i)).toBeInTheDocument();
  });
});
