import { useEffect, useState } from 'react';

/**
 * The current analysis id, taken from the URL.
 *
 * Hash routing rather than history routing, and no router dependency. An analysis has to be
 * addressable — the climber closes the tab while a two-minute analysis runs and comes back to
 * the link — and a hash survives being served by anything, including a static host with no
 * rewrite rules and the API itself. A path-based route would need every deployment to know
 * to fall back to index.html.
 */
export function useAnalysisId(): [string | null, (id: string | null) => void] {
  const [id, setId] = useState<string | null>(parseHash(window.location.hash));

  useEffect(() => {
    const onChange = () => setId(parseHash(window.location.hash));
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  const navigate = (next: string | null) => {
    window.location.hash = next ? `/a/${next}` : '';
    setId(next);
  };

  return [id, navigate];
}

export function parseHash(hash: string): string | null {
  const match = /^#\/a\/([^/?#]+)$/.exec(hash);
  return match?.[1] ? decodeURIComponent(match[1]) : null;
}
