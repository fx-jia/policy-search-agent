import type { AssistantTurn } from '@/interfaces/database/medical-assistant';
import { useTranslation } from 'react-i18next';

const TraceEvents = ['initial_search', 'decision', 'fallback'];
const ToolActions = ['read_chunk', 'read_neighbors', 'search_policy'];
const StopReasons = [
  'decision_limit',
  'native_replay',
  'no_evidence',
  'final_call_reserved',
  'search_deadline',
  'controller_fallback',
  'sufficient',
  'missing_user_information',
  'no_more_evidence',
  'insufficient',
  'search_disabled',
  'action_limit',
  'duplicate_action',
  'context_budget',
  'no_progress',
  'search_timeout',
  'decision_timeout',
  'generation_timeout',
  'invalid_decision',
  'technical_error',
  'resolution_timeout',
  'resolution_error',
  'policy_request_cancelled',
  'policy_request_error',
];

export function SearchTrace({ turn }: { turn: AssistantTurn }) {
  const { t } = useTranslation();
  const events = turn.trace.filter(
    (item) =>
      TraceEvents.includes(item.event) ||
      (item.event === 'tool' &&
        item.action != null &&
        ToolActions.includes(item.action)),
  );
  return (
    <details className="mt-6 rounded-xl border border-border-button bg-bg-base p-4">
      <summary className="cursor-pointer text-sm font-semibold">
        {t('medicalAssistant.searchTrace')}
      </summary>
      <div className="mt-4 space-y-3 text-xs leading-6 text-text-secondary">
        {events.length === 0 && <p>{t('medicalAssistant.noTrace')}</p>}
        <ol className="space-y-3">
          {events.map((item, index) => (
            <li
              key={`${item.event}-${index}`}
              className="border-l-2 border-border-button pl-3"
            >
              <p className="font-medium text-text-primary">
                {t(
                  `medicalAssistant.trace.${item.event === 'tool' ? item.action : item.event}`,
                )}
              </p>
              {item.event !== 'decision' && item.query && (
                <p className="break-words">
                  {t('medicalAssistant.traceQuery')}: {item.query}
                </p>
              )}
              {!!item.needs?.length && (
                <p className="break-words">
                  {t('medicalAssistant.traceNeeds')}: {item.needs.join('；')}
                </p>
              )}
              {!!item.chunk_ids?.length && (
                <p>
                  {t('medicalAssistant.traceChunks', {
                    count: item.chunk_ids.length,
                  })}
                </p>
              )}
              {item.new_chunk_ids && (
                <p>
                  {t('medicalAssistant.traceNewChunks', {
                    count: item.new_chunk_ids.length,
                  })}
                </p>
              )}
            </li>
          ))}
        </ol>
        {turn.fallback_used && <p>{t('medicalAssistant.fallback')}</p>}
        {turn.stop_reason && (
          <p>
            {t('medicalAssistant.stopReason')}:{' '}
            {StopReasons.includes(turn.stop_reason)
              ? t(`medicalAssistant.stopReasons.${turn.stop_reason}`)
              : t('medicalAssistant.stopReasonUnknown')}
          </p>
        )}
        {turn.latency_ms != null && (
          <p>
            {t('medicalAssistant.totalLatency', {
              seconds: (turn.latency_ms / 1000).toFixed(2),
            })}
          </p>
        )}
        {turn.effective_question &&
          turn.effective_question !== turn.question && (
            <details>
              <summary className="cursor-pointer">
                {t('medicalAssistant.effectiveQuestion')}
              </summary>
              <p className="mt-2 whitespace-pre-wrap break-words">
                {turn.effective_question}
              </p>
            </details>
          )}
      </div>
    </details>
  );
}
