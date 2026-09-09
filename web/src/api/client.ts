/**
 * The typed client. Every call to the service goes through here.
 *
 * Upload uses XMLHttpRequest rather than fetch, for one reason: fetch cannot report upload
 * progress. A climber uploading a 200 MB video over gym wifi needs to see it moving, and a
 * spinner that sits still for two minutes is indistinguishable from a hang.
 */

import type { AnalysisReport, AnalysisStatus, CreatedAnalysis, Health } from './types';

/**
 * Limits mirrored from `api/config.py`. Checking here as well as on the server is not
 * redundant: it turns "upload 200 MB, wait, get rejected" into an instant answer. The server
 * remains the authority — a client-side check is a courtesy, never a guarantee.
 */
export const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;

export const ALLOWED_EXTENSIONS = ['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'] as const;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** A rejection the client worked out for itself, before anything was sent. */
export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot === -1 ? '' : filename.slice(dot).toLowerCase();
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const megabytes = bytes / (1024 * 1024);
  return megabytes >= 1 ? `${megabytes.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

/**
 * Why this file cannot be uploaded, or null if it can.
 *
 * Returns a message rather than throwing: the caller shows it next to the file input, and a
 * rejected file is an ordinary thing for a person to do, not an exceptional one.
 */
export function validateFile(file: File): string | null {
  const extension = extensionOf(file.name);
  if (!ALLOWED_EXTENSIONS.includes(extension as (typeof ALLOWED_EXTENSIONS)[number])) {
    return `${extension || 'That file'} is not a supported video format. Supported: ${ALLOWED_EXTENSIONS.join(', ')}.`;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `That video is ${formatBytes(file.size)}, over the ${formatBytes(MAX_UPLOAD_BYTES)} limit.`;
  }
  if (file.size === 0) {
    return 'That file is empty.';
  }
  return null;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return body.detail;
  } catch {
    // A non-JSON error body is not worth a second failure; fall through to the status text.
  }
  return response.statusText || `request failed with ${response.status}`;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, signal ? { signal } : {});
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  return (await response.json()) as T;
}

export function getAnalysis(id: string, signal?: AbortSignal): Promise<AnalysisStatus> {
  return getJson<AnalysisStatus>(`/analyses/${encodeURIComponent(id)}`, signal);
}

export function getReport(id: string, signal?: AbortSignal): Promise<AnalysisReport> {
  return getJson<AnalysisReport>(`/analyses/${encodeURIComponent(id)}/report`, signal);
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return getJson<Health>('/health', signal);
}

export async function deleteAnalysis(id: string): Promise<void> {
  const response = await fetch(`/analyses/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!response.ok && response.status !== 404) {
    throw new ApiError(await readError(response), response.status);
  }
}

export function videoUrl(id: string): string {
  return `/analyses/${encodeURIComponent(id)}/video`;
}

export function overlayUrl(id: string): string {
  return `/analyses/${encodeURIComponent(id)}/overlay`;
}

export interface UploadHandle {
  promise: Promise<CreatedAnalysis>;
  abort: () => void;
}

/**
 * Upload a video, reporting progress as it goes.
 *
 * The returned handle carries an `abort` so a retry can cancel the attempt it is replacing —
 * without it a retried upload races the original, and whichever finishes last wins.
 */
export function uploadVideo(file: File, onProgress?: (fraction: number) => void): UploadHandle {
  const request = new XMLHttpRequest();
  const body = new FormData();
  body.append('file', file);

  const promise = new Promise<CreatedAnalysis>((resolve, reject) => {
    request.open('POST', '/analyses');

    request.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(event.loaded / event.total);
      }
    });

    request.addEventListener('load', () => {
      if (request.status >= 200 && request.status < 300) {
        try {
          resolve(JSON.parse(request.responseText) as CreatedAnalysis);
        } catch {
          reject(new ApiError('the server sent a response we could not read', request.status));
        }
        return;
      }
      let detail = `upload failed with ${request.status}`;
      try {
        const parsed = JSON.parse(request.responseText) as { detail?: unknown };
        if (typeof parsed.detail === 'string') detail = parsed.detail;
      } catch {
        // Keep the status-based message.
      }
      reject(new ApiError(detail, request.status));
    });

    request.addEventListener('error', () =>
      reject(new ApiError('the upload could not reach the server', 0)),
    );
    request.addEventListener('abort', () => reject(new ApiError('the upload was cancelled', 0)));

    request.send(body);
  });

  return { promise, abort: () => request.abort() };
}
