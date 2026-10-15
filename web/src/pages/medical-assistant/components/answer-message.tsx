import { Button } from '@/components/ui/button';
import type {
  AssistantCitation,
  AssistantSourceState,
  AssistantTurn,
} from '@/interfaces/database/medical-assistant';
import { Check, Copy } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { citationMarkdown, safeSourceUrl } from '../utils/citations';

export interface CitationSelection {
  request_id: string;
  index: number;
}
interface AnswerMessageProps {
  turn: AssistantTurn;
  onSelectCitation: (selection: CitationSelection) => void;
}

function CitationButton({
  turn,
  citation,
  onSelectCitation,
}: {
  turn: AssistantTurn;
  citation: AssistantCitation;
  onSelectCitation: AnswerMessageProps['onSelectCitation'];
}) {
  function handleSelect() {
    onSelectCitation({
      request_id: turn.request_id ?? turn.question_id,
      index: citation.index,
    });
  }
  return (
    <button
      className="mx-0.5 inline-flex rounded border border-accent-primary/30 bg-accent-primary/10 px-1.5 text-sm font-semibold text-accent-primary hover:bg-accent-primary/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent-primary"
      onClick={handleSelect}
    >
      [{citation.index + 1}]
    </button>
  );
}

export function SourceLimitations({
  source,
}: {
  source: AssistantSourceState | Partial<AssistantSourceState>;
}) {
  const { t } = useTranslation();
  return (
    <div className="mt-4 space-y-1 border-t border-border-button pt-3 text-xs leading-relaxed text-text-secondary">
      <p>{t('medicalAssistant.sourceBoundary')}</p>
      {source.as_of && (
        <p>
          {t('medicalAssistant.asOf')}: {source.as_of}
        </p>
      )}
      {!source.applicability || source.applicability === 'unknown' ? (
        <p>{t('medicalAssistant.applicabilityUnknown')}</p>
      ) : (
        <p>
          {t('medicalAssistant.applicability')}: {source.applicability}
        </p>
      )}
      {source.stale === true && (
        <p className="text-state-warning">{t('medicalAssistant.stale')}</p>
      )}
      {source.stale == null && <p>{t('medicalAssistant.freshnessUnknown')}</p>}
      {source.coverage?.complete === false ? (
        <p>{t('medicalAssistant.coveragePartial')}</p>
      ) : (
        source.coverage?.complete !== true && (
          <p>{t('medicalAssistant.coverageUnknown')}</p>
        )
      )}
      {!!source.pending_count && (
        <p>
          {t('medicalAssistant.pendingDocuments', {
            count: source.pending_count,
          })}
        </p>
      )}
    </div>
  );
}

function markdownUrl(url: string) {
  return /^medical-citation:\d+$/.test(url) ? url : (safeSourceUrl(url) ?? '');
}

export function AnswerMessage({ turn, onSelectCitation }: AnswerMessageProps) {
  const { t } = useTranslation();
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>(
    'idle',
  );
  const failed = [
    'timeout',
    'technical_error',
    'generation_error',
    'cancelled',
  ].includes(turn.status);
  async function handleCopy() {
    const plainAnswer = turn.answer.replace(
      /##(-?\d+)\$\$|\[ID:(-?\d+)\]/g,
      (_marker, hashIndex, idIndex) => {
        const index = Number(hashIndex ?? idIndex);
        return turn.citations.some((citation) => citation.index === index)
          ? `[${index + 1}]`
          : t('medicalAssistant.citationUnavailable');
      },
    );
    try {
      await navigator.clipboard.writeText(plainAnswer);
      setCopyState('copied');
    } catch {
      setCopyState('failed');
    }
  }
  return (
    <article
      className="rounded-2xl border border-border-button bg-bg-card p-5 sm:p-6"
      aria-label={t('medicalAssistant.answer')}
    >
      <div className="mb-4 flex items-center justify-between gap-3">
        <span className="text-xs font-semibold tracking-wide text-text-secondary">
          {t(
            failed
              ? 'medicalAssistant.turnFailed'
              : turn.status === 'needs_clarification'
                ? 'medicalAssistant.clarification'
                : 'medicalAssistant.assistant',
          )}
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          aria-label={t('medicalAssistant.copyAnswer')}
        >
          <Copy className="size-3.5" />
          {copyState === 'copied' && <Check className="size-3.5" />}
        </Button>
      </div>
      {copyState === 'failed' && (
        <p role="status" className="mb-2 text-xs text-text-secondary">
          {t('medicalAssistant.copyFailed')}
        </p>
      )}
      <div className="break-words text-[15px] leading-7 text-text-primary [&_p]:mb-3 [&_p:last-child]:mb-0 [&_h1]:mb-3 [&_h1]:text-xl [&_h1]:font-semibold [&_h2]:mb-3 [&_h2]:mt-6 [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:mb-2 [&_h3]:mt-4 [&_h3]:font-semibold [&_ul]:mb-3 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:mb-3 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:mb-1 [&_blockquote]:border-l-2 [&_blockquote]:border-accent-primary [&_blockquote]:pl-4 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-bg-base [&_pre]:p-3 [&_code]:text-sm [&_th]:border [&_th]:border-border-button [&_th]:bg-bg-base [&_th]:p-2 [&_td]:border [&_td]:border-border-button [&_td]:p-2">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          skipHtml
          urlTransform={markdownUrl}
          components={{
            a({ href, children }) {
              const match = /^medical-citation:(\d+)$/.exec(href ?? '');
              if (match) {
                const citation = turn.citations.find(
                  (item) => item.index === Number(match[1]),
                );
                return citation ? (
                  <CitationButton
                    turn={turn}
                    citation={citation}
                    onSelectCitation={onSelectCitation}
                  />
                ) : (
                  <span>{t('medicalAssistant.citationUnavailable')}</span>
                );
              }
              const sourceUrl = safeSourceUrl(href);
              return sourceUrl ? (
                <a
                  href={sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent-primary underline underline-offset-4"
                >
                  {children}
                </a>
              ) : (
                <span>{children}</span>
              );
            },
            img({ alt }) {
              return <span>{alt}</span>;
            },
            table({ children }) {
              return (
                <div className="mb-4 max-w-full overflow-x-auto">
                  <table className="w-full border-collapse text-left text-sm">
                    {children}
                  </table>
                </div>
              );
            },
          }}
        >
          {citationMarkdown(
            turn.answer,
            turn.citations,
            t('medicalAssistant.citationUnavailable'),
          )}
        </ReactMarkdown>
      </div>
      <SourceLimitations source={turn.source_state} />
    </article>
  );
}
