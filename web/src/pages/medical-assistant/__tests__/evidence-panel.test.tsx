import { render, screen } from '@testing-library/react';
import { FirstTurn } from '../../../../test-utils/medical-assistant-fixtures';
import { EvidencePanel } from '../components/evidence-panel';

jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

it('positions citations inside each panel when desktop and drawer are both mounted', () => {
  const visited: HTMLElement[] = [];
  const original = HTMLElement.prototype.scrollIntoView;
  HTMLElement.prototype.scrollIntoView = function () {
    visited.push(this);
  };
  try {
    render(
      <>
        <EvidencePanel turn={FirstTurn} activeCitationIndex={0} />
        <EvidencePanel turn={FirstTurn} activeCitationIndex={0} />
      </>,
    );
    const articles = screen.getAllByRole('article');
    expect(visited).toHaveLength(2);
    expect(visited[0]).toBe(articles[0]);
    expect(visited[1]).toBe(articles[1]);
  } finally {
    HTMLElement.prototype.scrollIntoView = original;
  }
});
