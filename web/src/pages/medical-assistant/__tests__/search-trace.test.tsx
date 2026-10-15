import { render, screen } from '@testing-library/react';
import { SearchTrace } from '../components/search-trace';
import { FirstTurn } from '../../../../test-utils/medical-assistant-fixtures';
jest.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, data?: Record<string, string | number>) => {
      const translations =
        jest.requireActual('@/locales/zh').default.translation;
      const value =
        key
          .split('.')
          .reduce((entry: any, part) => entry?.[part], translations) ?? key;
      return value.replace(/{{(\w+)}}/g, (_: string, name: string) =>
        String(data?.[name] ?? ''),
      );
    },
  }),
}));
it('shows only actual events and overall duration, without fabricated stage timings', () => {
  render(<SearchTrace turn={FirstTurn} />);
  expect(screen.getByText('首次检索')).toBeInTheDocument();
  expect(screen.queryByText('定向检索政策')).not.toBeInTheDocument();
  expect(screen.getByText('本轮总耗时 1.23 秒')).toBeInTheDocument();
});
it('shows executed tool activities from the projected backend DTO and returned chunk counts', () => {
  const toolActions = ['read_chunk', 'read_neighbors', 'search_policy'];
  render(
    <SearchTrace
      turn={{
        ...FirstTurn,
        trace: toolActions.map((action) => ({
          event: 'tool',
          query: null,
          needs: null,
          action,
          chunk_ids: null,
          new_chunk_ids:
            action === 'read_neighbors' ? ['chunk-b', 'chunk-c'] : [],
        })),
        fallback_used: true,
      }}
    />,
  );
  for (const label of ['补读片段', '读取相邻片段', '定向检索政策'])
    expect(screen.getByText(label)).toBeInTheDocument();
  expect(screen.getByText('新增 2 个片段')).toBeInTheDocument();
  expect(screen.getAllByText('新增 0 个片段')).toHaveLength(2);
  expect(screen.getByText('本轮采用基础检索回答。')).toBeInTheDocument();
});

it('does not describe an unexecuted decision proposal or its query as performed retrieval', () => {
  render(
    <SearchTrace
      turn={{
        ...FirstTurn,
        trace: [
          {
            event: 'decision',
            query: '尚未执行的检索词',
            action: 'search_policy',
            needs: ['办理日期：missing'],
            chunk_ids: null,
            new_chunk_ids: null,
          },
        ],
      }}
    />,
  );
  expect(screen.getByText('证据需求判断')).toBeInTheDocument();
  expect(screen.getByText('证据需求: 办理日期：missing')).toBeInTheDocument();
  expect(
    screen.queryByText(/实际动作|search_policy|尚未执行的检索词/),
  ).not.toBeInTheDocument();
  expect(screen.queryByText('定向检索政策')).not.toBeInTheDocument();
});

it.each([
  ['sufficient', '已有足够证据'],
  ['action_limit', '已达到本轮补充动作上限'],
  ['missing_user_information', '需要用户补充条件'],
  ['decision_timeout', '证据判断超时'],
  ['resolution_error', '问题解析遇到技术问题'],
])('explains the actual %s stop reason in Chinese', (reason, label) => {
  render(<SearchTrace turn={{ ...FirstTurn, stop_reason: reason }} />);
  expect(screen.getByText(`停止原因: ${label}`)).toBeInTheDocument();
});

it('uses a readable fallback without exposing an unknown internal stop code', () => {
  render(
    <SearchTrace turn={{ ...FirstTurn, stop_reason: 'internal_reason_v2' }} />,
  );
  expect(
    screen.getByText('停止原因: 本轮检索已停止，具体原因暂不可展示。'),
  ).toBeInTheDocument();
  expect(screen.queryByText(/internal_reason_v2/)).not.toBeInTheDocument();
});
