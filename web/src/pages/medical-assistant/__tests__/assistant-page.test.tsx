import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { AxiosError } from 'axios';
import medicalAssistantService from '@/services/medical-assistant-service';
import { AssistantPage } from '../assistant-page';
import {
  FirstTurn,
  ReadySession,
} from '../../../../test-utils/medical-assistant-fixtures';

jest.mock('@/services/medical-assistant-service', () => ({
  __esModule: true,
  default: {
    getSession: jest.fn(),
    submitTurn: jest.fn(),
    clearTopic: jest.fn(),
  },
}));
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));
const Transport = jest.mocked(medicalAssistantService);
const response = <T,>(data: T) => ({ data: { code: 0, message: '', data } });
function mount() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <AssistantPage />
    </QueryClientProvider>,
  );
}
async function ask(question: string) {
  const input = await screen.findByRole('textbox');
  await waitFor(() => expect(input).not.toBeDisabled());
  fireEvent.change(input, { target: { value: question } });
  fireEvent.click(
    screen.getByRole('button', { name: 'medicalAssistant.send' }),
  );
}
beforeEach(() => {
  jest.clearAllMocks();
  Transport.getSession.mockResolvedValue(response(ReadySession) as never);
});
it('examples only fill the composer; never initiate generation', async () => {
  mount();
  fireEvent.click(
    await screen.findByRole('button', { name: 'medicalAssistant.example1' }),
  );
  expect(screen.getByRole('textbox')).toHaveValue('medicalAssistant.example1');
  expect(Transport.submitTurn).not.toHaveBeenCalled();
});
it('submits only current user words after clarification, without client-built history', async () => {
  const clarification = {
    ...FirstTurn,
    status: 'needs_clarification',
    answer: '请补充参保类型',
    citations: [],
  };
  Transport.submitTurn
    .mockImplementationOnce(async () => {
      Transport.getSession.mockResolvedValue(
        response({ ...ReadySession, turns: [clarification] }) as never,
      );
      return response({ state: 'completed', turn: clarification }) as never;
    })
    .mockImplementationOnce(async () => {
      Transport.getSession.mockResolvedValue(
        response({
          ...ReadySession,
          turns: [clarification, { ...FirstTurn, request_id: 'turn-2' }],
        }) as never,
      );
      return response({
        state: 'completed',
        turn: { ...FirstTurn, request_id: 'turn-2' },
      }) as never;
    });
  mount();
  await ask('报销条件是什么？');
  await screen.findByText('请补充参保类型');
  await ask('我是上海居民医保');
  await screen.findByText(/按规定报销/);
  const bodies = Transport.submitTurn.mock.calls.map(
    ([config]) => (config as { data: unknown }).data,
  );
  expect(bodies).toEqual([
    { request_id: expect.any(String), question: '报销条件是什么？' },
    { request_id: expect.any(String), question: '我是上海居民医保' },
  ]);
});
it('recovers a disconnected POST through GET, never automatically resubmits', async () => {
  Transport.submitTurn.mockRejectedValueOnce(new Error('network interrupted'));
  Transport.getSession
    .mockResolvedValueOnce(response(ReadySession) as never)
    .mockResolvedValue(
      response({ ...ReadySession, turns: [FirstTurn] }) as never,
    );
  mount();
  await ask(FirstTurn.question);
  await screen.findByText(/按规定报销/);
  expect(Transport.submitTurn).toHaveBeenCalledTimes(1);
  expect(Transport.getSession.mock.calls.length).toBeGreaterThan(1);
});
it.each([400, 403, 409])(
  'shows a definite HTTP %s rejection after state readback',
  async (status) => {
    Transport.submitTurn.mockRejectedValueOnce(
      new AxiosError(
        'Request failed',
        'ERR_BAD_REQUEST',
        undefined,
        undefined,
        {
          status,
          statusText: 'Rejected',
          headers: {},
          config: {} as never,
          data: { code: status, message: 'Request rejected', data: null },
        },
      ),
    );
    mount();
    await ask('原话');
    await screen.findByRole('alert');
    await waitFor(() => expect(screen.getByRole('textbox')).not.toBeDisabled());
    expect(screen.getByRole('alert')).toHaveTextContent(
      'medicalAssistant.requestFailed',
    );
    expect(screen.getByRole('textbox')).toHaveValue('原话');
    expect(Transport.submitTurn).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByText('medicalAssistant.confirming'),
    ).not.toBeInTheDocument();
  },
);
it('keeps composer locked if readback cannot confirm a disconnected command', async () => {
  Transport.submitTurn.mockRejectedValueOnce(new Error('network interrupted'));
  Transport.getSession
    .mockResolvedValueOnce(response(ReadySession) as never)
    .mockRejectedValue(new Error('still offline'));
  mount();
  await ask('原话');
  await screen.findByText('medicalAssistant.confirming');
  expect(screen.getByRole('textbox')).toBeDisabled();
  expect(
    screen.queryByRole('button', { name: /retry|重试/i }),
  ).not.toBeInTheDocument();
});
it('refresh restores completed turns without POST replay', async () => {
  Transport.getSession.mockResolvedValue(
    response({ ...ReadySession, turns: [FirstTurn] }) as never,
  );
  mount();
  await screen.findByText(/按规定报销/);
  expect(Transport.submitTurn).not.toHaveBeenCalled();
});
it('server running state prevents both submission and topic clear', async () => {
  Transport.getSession.mockResolvedValue(
    response({
      ...ReadySession,
      state: 'running',
      can_submit: false,
      current_request_id: 'pending',
    }) as never,
  );
  mount();
  await screen.findByText('medicalAssistant.waiting');
  expect(screen.getByRole('textbox')).toBeDisabled();
  expect(
    screen.getByRole('button', { name: 'medicalAssistant.newTopic' }),
  ).toBeDisabled();
});
it('new topic does not remove a provider-usage block or create a new session', async () => {
  const blocked: import('@/interfaces/database/medical-assistant').AssistantSession =
    {
      ...ReadySession,
      state: 'blocked',
      can_submit: false,
      blocked_reason: 'unresolved_provider_usage',
      turns: [FirstTurn],
    };
  Transport.getSession.mockResolvedValue(response(blocked) as never);
  Transport.clearTopic.mockResolvedValue(
    response({ ...blocked, turns: [] }) as never,
  );
  mount();
  const newTopic = await screen.findByRole('button', {
    name: 'medicalAssistant.newTopic',
  });
  await waitFor(() => expect(newTopic).not.toBeDisabled());
  fireEvent.click(newTopic);
  await waitFor(() =>
    expect(screen.queryByText(/按规定报销/)).not.toBeInTheDocument(),
  );
  expect(screen.getByRole('textbox')).toBeDisabled();
  expect(
    screen.getByText('medicalAssistant.reasons.unresolved_provider_usage'),
  ).toBeInTheDocument();
  expect(Transport.clearTopic).toHaveBeenCalledTimes(1);
});
it('keeps technical failure distinct from a normal policy answer', async () => {
  Transport.getSession.mockResolvedValue(
    response({
      ...ReadySession,
      turns: [
        { ...FirstTurn, status: 'technical_error', answer: '用户可读失败说明' },
      ],
    }) as never,
  );
  mount();
  await screen.findByText('用户可读失败说明');
  expect(screen.getByText('medicalAssistant.turnFailed')).toBeInTheDocument();
});
it('shows unavailable configuration without sample results', async () => {
  Transport.getSession.mockResolvedValue(
    response({
      ...ReadySession,
      state: 'unavailable',
      can_submit: false,
      blocked_reason: 'service_unavailable',
    }) as never,
  );
  mount();
  await screen.findByText('medicalAssistant.states.unavailable');
  expect(screen.getByRole('textbox')).toBeDisabled();
  expect(screen.queryByText(/按规定报销/)).not.toBeInTheDocument();
});

it('switches real historical evidence panels when separate turns have no web request id', async () => {
  const second = {
    ...FirstTurn,
    request_id: null,
    question_id: 'q-older-two',
    answer: '第二轮。##0$$',
    citations: [{ ...FirstTurn.citations[0], text: '第二轮原文' }],
  };
  Transport.getSession.mockResolvedValue(
    response({
      ...ReadySession,
      turns: [{ ...FirstTurn, request_id: null }, second],
    }) as never,
  );
  mount();
  const citations = await screen.findAllByRole('button', { name: '[1]' });
  fireEvent.click(citations[0]);
  expect(screen.getByText('甲的原文')).toBeInTheDocument();
  fireEvent.click(screen.getAllByRole('button', { name: '[1]' })[1]);
  expect(screen.getByText('第二轮原文')).toBeInTheDocument();
  expect(screen.queryByText('甲的原文')).not.toBeInTheDocument();
});
