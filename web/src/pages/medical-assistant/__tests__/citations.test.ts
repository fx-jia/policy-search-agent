import { citationMarkdown, safeSourceUrl } from '../utils/citations';
import { FirstTurn } from '../../../../test-utils/medical-assistant-fixtures';

describe('turn-local citation mapping', () => {
  it('maps raw zero to display one without reordering evidence', () => {
    expect(citationMarkdown('依据##0$$和[ID:0]。', FirstTurn.citations)).toBe(
      '依据[1](medical-citation:0)和[1](medical-citation:0)。',
    );
  });
  it('marks unknown and negative indices unavailable', () => {
    expect(citationMarkdown('##4$$ [ID:-1]', FirstTurn.citations)).toBe(
      '引用暂不可用 引用暂不可用',
    );
  });
  it.each([
    'javascript:alert(1)',
    '//example.gov.cn',
    'data:text/html,a',
    'file:///etc/passwd',
  ])('rejects unsafe external source %s', (url) => {
    expect(safeSourceUrl(url)).toBeNull();
  });
  it('keeps a real https source', () => {
    expect(safeSourceUrl('https://example.gov.cn/policy')).toBe(
      'https://example.gov.cn/policy',
    );
  });
});
