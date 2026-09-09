import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { UploadView } from './UploadView';
import { ApiError } from '../api/client';

const uploadVideo = vi.hoisted(() => vi.fn());

vi.mock('../api/client', async (importOriginal) => {
  // The validation helpers are the real ones — they are what this view is meant to apply,
  // and stubbing them would leave the behaviour under test untested.
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, uploadVideo };
});

function videoFile(name = 'climb.mp4', size = 4096): File {
  const file = new File(['data'], name, { type: 'video/mp4' });
  Object.defineProperty(file, 'size', { value: size });
  return file;
}

function dropFile(file: File): void {
  const dropzone = screen.getByText(/or drag one here/i).closest('.dropzone');
  if (!dropzone) throw new Error('no dropzone rendered');
  fireEvent.drop(dropzone, { dataTransfer: { files: [file] } });
}

describe('UploadView', () => {
  beforeEach(() => {
    uploadVideo.mockReset();
  });

  it('refuses an unsupported format without contacting the server', async () => {
    render(<UploadView onStarted={vi.fn()} />);

    // Dropped rather than picked: the file input's `accept` filters the picker, so drag and
    // drop is the path an unsupported file actually arrives by — which is exactly why the
    // check cannot live on the input alone.
    dropFile(new File(['x'], 'notes.txt', { type: 'text/plain' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/not a supported video format/i);
    expect(uploadVideo).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: /analyze/i })).not.toBeInTheDocument();
  });

  it('refuses an oversized file without contacting the server', async () => {
    const user = userEvent.setup();
    render(<UploadView onStarted={vi.fn()} />);

    await user.upload(
      screen.getByLabelText(/choose a video/i),
      videoFile('huge.mp4', 400 * 1024 * 1024),
    );

    expect(await screen.findByRole('alert')).toHaveTextContent(/over the/i);
    expect(uploadVideo).not.toHaveBeenCalled();
  });

  it('uploads an accepted file and hands back the analysis id', async () => {
    const user = userEvent.setup();
    const onStarted = vi.fn();
    uploadVideo.mockReturnValue({
      promise: Promise.resolve({ id: 'abc123', state: 'pending', location: '/analyses/abc123' }),
      abort: vi.fn(),
    });

    render(<UploadView onStarted={onStarted} />);
    await user.upload(screen.getByLabelText(/choose a video/i), videoFile());
    await user.click(await screen.findByRole('button', { name: /analyze/i }));

    await waitFor(() => expect(onStarted).toHaveBeenCalledWith('abc123'));
  });

  it('shows the file it is about to send', async () => {
    const user = userEvent.setup();
    render(<UploadView onStarted={vi.fn()} />);

    await user.upload(screen.getByLabelText(/choose a video/i), videoFile('sunday-project.mp4'));

    expect(await screen.findByText('sunday-project.mp4')).toBeInTheDocument();
  });

  it('reports a server rejection and offers a retry', async () => {
    const user = userEvent.setup();
    // Built per call, not once up front: a rejected promise sitting unawaited between test
    // setup and render is reported as an unhandled rejection even though the component
    // handles it a moment later.
    uploadVideo.mockImplementation(() => ({
      promise: Promise.reject(new ApiError('the video exceeds the 200 MB limit', 413)),
      abort: vi.fn(),
    }));

    render(<UploadView onStarted={vi.fn()} />);
    await user.upload(screen.getByLabelText(/choose a video/i), videoFile());
    await user.click(await screen.findByRole('button', { name: /analyze/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/exceeds the 200 MB limit/);
    expect(await screen.findByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('retries with a fresh request rather than resuming the failed one', async () => {
    const user = userEvent.setup();
    uploadVideo.mockImplementation(() => ({
      promise: Promise.reject(new ApiError('the upload could not reach the server', 0)),
      abort: vi.fn(),
    }));

    render(<UploadView onStarted={vi.fn()} />);
    await user.upload(screen.getByLabelText(/choose a video/i), videoFile());
    await user.click(await screen.findByRole('button', { name: /analyze/i }));
    await user.click(await screen.findByRole('button', { name: /try again/i }));

    expect(uploadVideo).toHaveBeenCalledTimes(2);
  });

  it('cancelling aborts the request in flight', async () => {
    const user = userEvent.setup();
    const abort = vi.fn();
    uploadVideo.mockReturnValue({ promise: new Promise(() => {}), abort });

    render(<UploadView onStarted={vi.fn()} />);
    await user.upload(screen.getByLabelText(/choose a video/i), videoFile());
    await user.click(await screen.findByRole('button', { name: /analyze/i }));
    await user.click(await screen.findByRole('button', { name: /cancel/i }));

    // Without a real abort the upload keeps consuming the connection after the climber has
    // walked away from it.
    expect(abort).toHaveBeenCalled();
  });

  it('reports progress while the upload is in flight', async () => {
    const user = userEvent.setup();
    let report: ((fraction: number) => void) | undefined;
    uploadVideo.mockImplementation((_file: File, onProgress: (f: number) => void) => {
      report = onProgress;
      return { promise: new Promise(() => {}), abort: vi.fn() };
    });

    render(<UploadView onStarted={vi.fn()} />);
    await user.upload(screen.getByLabelText(/choose a video/i), videoFile());
    await user.click(await screen.findByRole('button', { name: /analyze/i }));

    report?.(0.42);
    expect(await screen.findByText(/42%/)).toBeInTheDocument();
    expect(screen.getByLabelText('upload progress')).toHaveValue(0.42);
  });
});
