export interface AssistantCitation {
  index: number;
  chunk_id: string | null;
  document_id: string | null;
  title: string | null;
  text: string;
  source_url: string | null;
  document_number: string | null;
  published_at: string | null;
  effective_from: string | null;
  effective_to: string | null;
  confirmation_status: string | null;
  revision_id: string | null;
}

export interface AssistantSourceState {
  release_id: string | null;
  as_of: string | null;
  scanned_at: string | null;
  applicability: string | null;
  stale: boolean | null;
  coverage: { complete?: boolean | null; [key: string]: unknown } | null;
  pending_count: number | null;
}

export interface AssistantTraceItem {
  event: string;
  query: string | null;
  needs: string[] | null;
  action: string | null;
  chunk_ids: string[] | null;
  new_chunk_ids: string[] | null;
}

export interface AssistantTurn {
  request_id: string | null;
  question_id: string;
  question: string;
  effective_question: string | null;
  status: string;
  answer: string;
  citations: AssistantCitation[];
  trace: AssistantTraceItem[];
  stop_reason: string | null;
  latency_ms: number | null;
  fallback_used: boolean;
  source_state: AssistantSourceState;
}

export interface AssistantSession {
  session_id: string | null;
  can_submit: boolean;
  state: 'ready' | 'running' | 'blocked' | 'unconfirmed' | 'unavailable';
  blocked_reason: string | null;
  current_request_id: string | null;
  turns: AssistantTurn[];
  scope: {
    city: '上海';
    label: '上海医保政策';
  } & Partial<AssistantSourceState>;
}

export type AssistantSubmission =
  | { state: 'completed'; turn: AssistantTurn }
  | { state: 'running' | 'unconfirmed'; request_id: string };
