import { fireEvent, render, screen, within } from '@testing-library/react';
import { AnswerMessage } from '../components/answer-message';
import { EvidencePanel } from '../components/evidence-panel';
import {
  FirstTurn,
  SecondTurn,
} from '../../../../test-utils/medical-assistant-fixtures';

jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      key === 'medicalAssistant.citationUnavailable' ? '引用暂不可用' : key,
  }),
}));
it('binds identical citation indices to the clicked turn', () => {
  const select = jest.fn();
  render(
    <>
      <AnswerMessage turn={FirstTurn} onSelectCitation={select} />
      <AnswerMessage turn={SecondTurn} onSelectCitation={select} />
    </>,
  );
  const citations = screen.getAllByRole('button', { name: '[1]' });
  fireEvent.click(citations[1]);
  expect(select).toHaveBeenLastCalledWith({ request_id: 'turn-2', index: 0 });
  fireEvent.click(citations[0]);
  expect(select).toHaveBeenLastCalledWith({ request_id: 'turn-1', index: 0 });
  expect(screen.queryByText('##0$$')).not.toBeInTheDocument();
});
it('renders invalid citation as unavailable, without a clickable approximate source', () => {
  render(
    <AnswerMessage
      turn={{ ...FirstTurn, answer: '未知来源##9$$' }}
      onSelectCitation={jest.fn()}
    />,
  );
  expect(screen.getByText(/引用暂不可用/)).toBeInTheDocument();
  expect(
    screen.queryByRole('button', { name: '[10]' }),
  ).not.toBeInTheDocument();
});
it('does not execute HTML, fetch images or create unsafe links in Markdown', () => {
  const { container } = render(
    <AnswerMessage
      turn={{
        ...FirstTurn,
        answer:
          '<img src=x onerror="alert(1)">\n\n[恶意](javascript:alert%281%29) ![remote](https://tracker.test/x.png)\n\n|项目|条件|\n|---|---|\n|居民医保|参保|',
      }}
      onSelectCitation={jest.fn()}
    />,
  );
  expect(container.querySelector('img')).toBeNull();
  expect(container.querySelector('[onerror]')).toBeNull();
  expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
  expect(screen.getByRole('table')).toBeInTheDocument();
});
it('keeps unknown timeliness and partial coverage visible even when technical status is ok', () => {
  render(<AnswerMessage turn={FirstTurn} onSelectCitation={jest.fn()} />);
  expect(
    screen.getByText('medicalAssistant.applicabilityUnknown'),
  ).toBeInTheDocument();
  expect(
    screen.getByText('medicalAssistant.coveragePartial'),
  ).toBeInTheDocument();
});
it('labels a restored cancelled query as incomplete', () => {
  render(
    <AnswerMessage
      turn={{
        ...FirstTurn,
        status: 'cancelled',
        answer: '查询未能完成，请查看服务状态。',
      }}
      onSelectCitation={jest.fn()}
    />,
  );
  expect(screen.getByText('medicalAssistant.turnFailed')).toBeInTheDocument();
  expect(
    screen.queryByText('medicalAssistant.assistant'),
  ).not.toBeInTheDocument();
});
it('renders citations with missing title and chunk metadata without inventing values', () => {
  render(
    <EvidencePanel
      turn={{
        ...FirstTurn,
        citations: [{ ...FirstTurn.citations[0], title: null, chunk_id: null }],
      }}
      activeCitationIndex={null}
    />,
  );
  expect(
    screen.getByText('medicalAssistant.untitledPolicy'),
  ).toBeInTheDocument();
  expect(screen.getByText('甲的原文')).toBeInTheDocument();
});
it('preserves source text, same-document chunks and missing metadata without inventing dates', () => {
  const citations = [
    FirstTurn.citations[0],
    {
      ...FirstTurn.citations[0],
      index: 1,
      chunk_id: 'chunk-second',
      title: '很长的政策标题'.repeat(20),
      text: '同文第二块',
      source_url: 'javascript:alert(1)',
    },
  ];
  render(
    <EvidencePanel
      turn={{ ...FirstTurn, citations }}
      activeCitationIndex={1}
    />,
  );
  const panel = screen.getByRole('region', {
    name: 'medicalAssistant.evidence',
  });
  expect(within(panel).getByText('甲的原文')).toBeInTheDocument();
  expect(within(panel).getByText('同文第二块')).toBeInTheDocument();
  expect(within(panel).getAllByRole('link')).toHaveLength(1);
  expect(
    within(panel).queryByText('medicalAssistant.effectiveFrom'),
  ).not.toBeInTheDocument();
});

it('keeps pre-web historical citations isolated by real question id when no command id exists', () => {
  const select = jest.fn();
  render(
    <>
      <AnswerMessage
        turn={{ ...FirstTurn, request_id: null }}
        onSelectCitation={select}
      />
      <AnswerMessage
        turn={{ ...SecondTurn, request_id: null }}
        onSelectCitation={select}
      />
    </>,
  );
  const buttons = screen.getAllByRole('button', { name: '[1]' });
  fireEvent.click(buttons[0]);
  expect(select).toHaveBeenLastCalledWith({ request_id: 'q-1', index: 0 });
  fireEvent.click(buttons[1]);
  expect(select).toHaveBeenLastCalledWith({ request_id: 'q-2', index: 0 });
});
