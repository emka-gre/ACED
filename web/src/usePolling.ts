import { useEffect, useRef, useState } from 'react';

import { ApiError, getAnalysis } from './api/client';
import type { AnalysisStatus } from './api/types';

/** How often a running analysis is re-checked. Progress here is coarse — pose extraction
 *  over hundreds of frames — so a faster poll would show the same number more often. */
export const POLL_INTERVAL_MS = 1500;

export interface PollResult {
  status: AnalysisStatus | null;
  error: string | null;
  /** True until the first response arrives, so a fresh page does not flash "not found". */
  loading: boolean;
  notFound: boolean;
}

/**
 * Poll an analysis until it reaches a terminal state, then stop.
 *
 * Polling rather than a socket: it is trivially resumable, which is the whole point when the
 * climber may close the tab and come back to the link an hour later.
 */
export function useAnalysis(id: string | null): PollResult {
  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(true);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!id) return;

    let cancelled = false;
    const controller = new AbortController();
    setLoading(true);
    setNotFound(false);
    setStatus(null);
    setError(null);

    const tick = async () => {
      try {
        const next = await getAnalysis(id, controller.signal);
        if (cancelled) return;
        setStatus(next);
        setError(null);
        setLoading(false);
        if (next.state === 'pending' || next.state === 'running') {
          timer.current = setTimeout(tick, POLL_INTERVAL_MS);
        }
      } catch (caught) {
        if (cancelled) return;
        setLoading(false);
        if (caught instanceof ApiError && caught.status === 404) {
          setNotFound(true);
          return;
        }
        // A transient network failure should not end the poll: the analysis is still
        // running on the server, and giving up would strand the climber on an error.
        setError(caught instanceof Error ? caught.message : 'could not reach the service');
        timer.current = setTimeout(tick, POLL_INTERVAL_MS * 2);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer.current) clearTimeout(timer.current);
    };
  }, [id]);

  return { status, error, loading, notFound };
}
