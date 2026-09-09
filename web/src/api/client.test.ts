import { describe, expect, it } from 'vitest';

import { extensionOf, formatBytes, MAX_UPLOAD_BYTES, validateFile } from './client';

function fileOf(name: string, bytes: number, type = 'video/mp4'): File {
  const file = new File(['x'], name, { type });
  // File size is read-only; a real 200 MB file in a unit test is not worth the memory.
  Object.defineProperty(file, 'size', { value: bytes });
  return file;
}

describe('extensionOf', () => {
  it('lowercases, so a .MOV from a phone is not rejected on case', () => {
    expect(extensionOf('CLIMB.MOV')).toBe('.mov');
  });

  it('handles a name with no extension', () => {
    expect(extensionOf('climb')).toBe('');
  });

  it('takes the last dot, not the first', () => {
    expect(extensionOf('my.climb.v2.mp4')).toBe('.mp4');
  });
});

describe('validateFile', () => {
  it('accepts a supported video within the limit', () => {
    expect(validateFile(fileOf('climb.mp4', 1024))).toBeNull();
  });

  it('accepts every format the server accepts', () => {
    for (const extension of ['.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm']) {
      expect(validateFile(fileOf(`climb${extension}`, 1024))).toBeNull();
    }
  });

  it('names the supported formats when refusing one', () => {
    const message = validateFile(fileOf('climb.gif', 1024));
    expect(message).toContain('.mp4');
    expect(message).toContain('not a supported video format');
  });

  it('refuses a file over the limit, stating its size and the limit', () => {
    const message = validateFile(fileOf('climb.mp4', MAX_UPLOAD_BYTES + 1));
    expect(message).toContain('200.0 MB');
  });

  it('refuses an empty file', () => {
    expect(validateFile(fileOf('climb.mp4', 0))).toBe('That file is empty.');
  });

  it('checks before anything is sent, so a rejection costs no upload', () => {
    // The point of the client-side check: it is synchronous and needs no network.
    expect(validateFile(fileOf('notes.txt', 10))).not.toBeNull();
  });
});

describe('formatBytes', () => {
  it('uses megabytes above a megabyte', () => {
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB');
  });

  it('uses kilobytes below one', () => {
    expect(formatBytes(2048)).toBe('2 KB');
  });
});
