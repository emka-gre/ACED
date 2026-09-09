import { useCallback, useEffect, useRef, useState } from 'react';

import { getAnalysis, overlayUrl } from './api/client';
import type { ArtifactState } from './api/types';

const POLL_MS = 2000;

/**
 * The skeleton overlay, which does not exist until someone asks for it.
 *
 * Asking is a GET of the overlay URL: the API renders on first request and answers 202 while
 * it works. The request is made with `Range: bytes=0-0` so that a ready overlay costs one byte
 * rather than downloading a whole video just to find out it is there — the `<video>` element
 * fetches it properly a moment later.
 */
export function useOverlay(id: string, initial: ArtifactState) {
  const [state, setState] = useState<ArtifactState>(initial);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => setState(initial), [initial]);

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const poll = useCallback(async () => {
    try {
      const status = await getAnalysis(id);
      setState(status.overlay);
      if (status.overlay === 'preparing') {
        timer.current = setTimeout(() => void poll(), POLL_MS);
      }
    } catch {
      timer.current = setTimeout(() => void poll(), POLL_MS * 2);
    }
  }, [id]);

  const request = useCallback(async () => {
    if (state === 'ready' || state === 'preparing') return;
    setError(null);
    try {
      const response = await fetch(overlayUrl(id), { headers: { Range: 'bytes=0-0' } });
      if (response.status === 202) {
        setState('preparing');
        timer.current = setTimeout(() => void poll(), POLL_MS);
        return;
      }
      if (response.ok || response.status === 206) {
        setState('ready');
        return;
      }
      setState('failed');
      setError('The overlay could not be rendered for this climb.');
    } catch {
      setError('Could not reach the service to render the overlay.');
    }
  }, [id, poll, state]);

  return { state, error, request };
}
