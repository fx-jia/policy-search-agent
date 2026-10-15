import type {
  AssistantSession,
  AssistantTurn,
} from '@/interfaces/database/medical-assistant';

export const FirstTurn: AssistantTurn = {
  request_id: 'turn-1',
  question_id: 'q-1',
  question: '上海居民医保如何报销？',
  effective_question: '上海居民医保如何报销？',
  status: 'ok',
  answer: '按规定报销。##0$$',
  citations: [
    {
      index: 0,
      chunk_id: 'chunk-a',
      document_id: 'doc-a',
      title: '政策甲',
      text: '甲的原文',
      source_url: 'https://example.gov.cn/policy-a',
      document_number: null,
      published_at: '2026-01-01',
      effective_from: null,
      effective_to: null,
      confirmation_status: null,
      revision_id: null,
    },
  ],
  trace: [
    {
      event: 'initial_search',
      query: '居民医保',
      needs: null,
      action: null,
      chunk_ids: ['chunk-a'],
      new_chunk_ids: null,
    },
  ],
  stop_reason: 'sufficient',
  latency_ms: 1234,
  fallback_used: false,
  source_state: {
    release_id: 'release-a',
    as_of: '2026-10-12',
    scanned_at: null,
    applicability: 'unknown',
    stale: null,
    coverage: { complete: false },
    pending_count: null,
  },
};
export const SecondTurn: AssistantTurn = {
  ...FirstTurn,
  request_id: 'turn-2',
  question_id: 'q-2',
  question: '职工医保呢？',
  answer: '职工医保另有规定。[ID:0]',
  citations: [
    {
      ...FirstTurn.citations[0],
      chunk_id: 'chunk-b',
      title: '政策乙',
      text: '乙的原文',
    },
  ],
};
export const ReadySession: AssistantSession = {
  session_id: 'fixed-session',
  can_submit: true,
  state: 'ready',
  blocked_reason: null,
  current_request_id: null,
  turns: [],
  scope: { city: '上海', label: '上海医保政策' },
};
