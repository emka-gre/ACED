import { describe, expect, it } from 'vitest';

import { parseHash } from './useHashRoute';

describe('parseHash', () => {
  it('reads the analysis id from an analysis link', () => {
    expect(parseHash('#/a/9f2c')).toBe('9f2c');
  });

  it('is empty at the root, which is the upload screen', () => {
    expect(parseHash('')).toBeNull();
    expect(parseHash('#/')).toBeNull();
  });

  it('does not match a deeper path', () => {
    expect(parseHash('#/a/9f2c/report')).toBeNull();
  });

  it('decodes an escaped id', () => {
    expect(parseHash('#/a/a%20b')).toBe('a b');
  });
});
