/** Seconds as `m:ss`, the way a video player writes a timestamp. */
export function formatTime(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `${minutes}:${remainder.toString().padStart(2, '0')}`;
}

/** A fraction as a whole percentage. */
export function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}
