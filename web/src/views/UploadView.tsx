import { useCallback, useRef, useState } from 'react';

import {
  ApiError,
  formatBytes,
  MAX_UPLOAD_BYTES,
  ALLOWED_EXTENSIONS,
  uploadVideo,
  validateFile,
} from '../api/client';
import type { UploadHandle } from '../api/client';

interface Props {
  onStarted: (analysisId: string) => void;
}

type Phase = 'idle' | 'uploading' | 'failed';

/**
 * Choosing a video and getting it to the service.
 *
 * The file is validated before a byte is sent. Uploading 200 MB over gym wifi only to be told
 * the format is unsupported is the worst version of this screen, and the check that prevents
 * it costs nothing. The server still validates — this is a courtesy, not a guarantee.
 */
export function UploadView({ onStarted }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [rejection, setRejection] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState(0);
  const [failure, setFailure] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const upload = useRef<UploadHandle | null>(null);

  const choose = useCallback((chosen: File | null) => {
    setFailure(null);
    if (!chosen) {
      setFile(null);
      setRejection(null);
      return;
    }
    const problem = validateFile(chosen);
    setRejection(problem);
    setFile(problem ? null : chosen);
  }, []);

  const start = useCallback(
    async (chosen: File) => {
      // Cancel any attempt still in flight, or a retry races the upload it replaces and
      // whichever finishes last wins.
      upload.current?.abort();

      setPhase('uploading');
      setProgress(0);
      setFailure(null);

      const handle = uploadVideo(chosen, setProgress);
      upload.current = handle;
      try {
        const created = await handle.promise;
        onStarted(created.id);
      } catch (caught) {
        const message =
          caught instanceof ApiError ? caught.message : 'the upload could not be completed';
        setFailure(message);
        setPhase('failed');
      } finally {
        upload.current = null;
      }
    },
    [onStarted],
  );

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files?.[0] ?? null);
  };

  const uploading = phase === 'uploading';

  return (
    <section className="panel">
      <h1>Analyze a climb</h1>
      <p className="lede">
        Upload a video of a single climber. The analysis looks at arm technique only, and takes a
        minute or two — you can close this page and come back to the link.
      </p>

      <div
        className={`dropzone${dragging ? ' dropzone-active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          id="video-input"
          type="file"
          accept={ALLOWED_EXTENSIONS.join(',')}
          disabled={uploading}
          onChange={(event) => choose(event.target.files?.[0] ?? null)}
        />
        <label htmlFor="video-input">Choose a video</label>
        <p className="hint">
          or drag one here — {ALLOWED_EXTENSIONS.join(', ')}, up to {formatBytes(MAX_UPLOAD_BYTES)}
        </p>
      </div>

      {rejection && (
        <p className="error" role="alert">
          {rejection}
        </p>
      )}

      {file && !uploading && (
        <div className="chosen">
          <p>
            <strong>{file.name}</strong> <span className="muted">{formatBytes(file.size)}</span>
          </p>
          <button type="button" className="primary" onClick={() => void start(file)}>
            {phase === 'failed' ? 'Try again' : 'Analyze'}
          </button>
        </div>
      )}

      {uploading && (
        <div className="uploading">
          <progress value={progress} max={1} aria-label="upload progress" />
          <p className="muted">Uploading… {Math.round(progress * 100)}%</p>
          <button type="button" onClick={() => upload.current?.abort()}>
            Cancel
          </button>
        </div>
      )}

      {failure && (
        <p className="error" role="alert">
          {failure}
        </p>
      )}
    </section>
  );
}
