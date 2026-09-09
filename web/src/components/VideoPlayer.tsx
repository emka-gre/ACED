import { useEffect, useRef, useState } from 'react';
import type { RefObject } from 'react';

import { overlayUrl, videoUrl } from '../api/client';
import type { ArtifactState, Fault } from '../api/types';
import { formatTime } from '../format';
import { useOverlay } from '../useOverlay';

interface Props {
  id: string;
  faults: Fault[];
  duration: number;
  overlay: ArtifactState;
  videoRef: RefObject<HTMLVideoElement | null>;
  activeFault: number | null;
  onTimeUpdate: (seconds: number) => void;
  onSelectFault: (index: number) => void;
}

/**
 * The climb, with its faults marked on the timeline.
 *
 * The timeline is driven by the report's own times rather than the video element's duration:
 * they come from the same container metadata, and using the report's keeps a marker in the
 * same place as the text that describes it even when a container misreports its length.
 */
export function VideoPlayer({
  id,
  faults,
  duration,
  overlay: initialOverlay,
  videoRef,
  activeFault,
  onTimeUpdate,
  onSelectFault,
}: Props) {
  const [showOverlay, setShowOverlay] = useState(false);
  const { state: overlayState, error: overlayError, request } = useOverlay(id, initialOverlay);
  // Where playback was when the source changed, so the toggle does not restart the climb.
  const resumeAt = useRef<{ time: number; playing: boolean } | null>(null);

  const source = showOverlay && overlayState === 'ready' ? overlayUrl(id) : videoUrl(id);

  useEffect(() => {
    const video = videoRef.current;
    const resume = resumeAt.current;
    if (!video || !resume) return;

    const restore = () => {
      video.currentTime = resume.time;
      if (resume.playing) void video.play();
      resumeAt.current = null;
    };
    video.addEventListener('loadedmetadata', restore, { once: true });
    return () => video.removeEventListener('loadedmetadata', restore);
  }, [source, videoRef]);

  const toggleOverlay = async () => {
    const video = videoRef.current;
    if (video) {
      resumeAt.current = { time: video.currentTime, playing: !video.paused };
    }
    if (!showOverlay && overlayState !== 'ready') {
      await request();
    }
    setShowOverlay((previous) => !previous);
  };

  const overlayLabel = () => {
    if (overlayState === 'preparing') return 'Rendering the overlay…';
    if (overlayState === 'failed') return 'Overlay unavailable';
    return showOverlay ? 'Hide skeleton overlay' : 'Show skeleton overlay';
  };

  return (
    <div className="player">
      <video
        ref={videoRef}
        src={source}
        controls
        playsInline
        preload="metadata"
        onTimeUpdate={(event) => onTimeUpdate(event.currentTarget.currentTime)}
      />

      <div className="timeline" aria-hidden="true">
        {faults.map((fault, index) => {
          const left = duration > 0 ? (fault.time_start / duration) * 100 : 0;
          const width = duration > 0 ? ((fault.time_end - fault.time_start) / duration) * 100 : 0;
          return (
            <button
              key={`${fault.frame_start}-${fault.frame_end}`}
              type="button"
              className={`marker${activeFault === index ? ' marker-active' : ''}`}
              style={{ left: `${left}%`, width: `${Math.max(width, 1)}%` }}
              onClick={() => onSelectFault(index)}
              title={`Fault ${index + 1} at ${formatTime(fault.time_start)}`}
            />
          );
        })}
      </div>

      <div className="player-controls">
        <button
          type="button"
          onClick={() => void toggleOverlay()}
          disabled={overlayState === 'preparing' || overlayState === 'failed'}
          aria-pressed={showOverlay}
        >
          {overlayLabel()}
        </button>
        {showOverlay && overlayState === 'ready' && (
          <span className="muted">Colour marks how each frame was classified.</span>
        )}
      </div>

      {overlayError && (
        <p className="warning" role="alert">
          {overlayError}
        </p>
      )}
    </div>
  );
}
