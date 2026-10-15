import type {
  AssistantCitation,
  AssistantTurn,
} from '@/interfaces/database/medical-assistant';
import dayjs from 'dayjs';
import { ExternalLink, FileText } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { safeSourceUrl } from '../utils/citations';

function CitationEvidence({
  citation,
  active,
}: {
  citation: AssistantCitation;
  active: boolean;
}) {
  const { t } = useTranslation();
  const sourceUrl = safeSourceUrl(citation.source_url);
  const metadata = [
    ['documentNumber', citation.document_number],
    ['publishedAt', citation.published_at],
    ['effectiveFrom', citation.effective_from],
    ['effectiveTo', citation.effective_to],
    ['confirmationStatus', citation.confirmation_status],
    ['revision', citation.revision_id],
  ];
  return (
    <article
      data-citation-index={citation.index}
      className={`scroll-mt-4 rounded-xl border p-4 ${active ? 'border-accent-primary bg-accent-primary/5' : 'border-border-button bg-bg-base'}`}
    >
      <h3 className="mb-3 flex items-start gap-2 text-sm font-semibold leading-6">
        <span className="shrink-0 text-accent-primary">
          [{citation.index + 1}]
        </span>
        <span className="break-words">
          {citation.title || t('medicalAssistant.untitledPolicy')}
        </span>
      </h3>
      <dl className="mb-3 space-y-1 text-xs leading-5 text-text-secondary">
        {metadata.map(
          ([key, value]) =>
            value && (
              <div key={key} className="flex gap-2">
                <dt className="shrink-0">{t(`medicalAssistant.${key}`)}</dt>
                <dd className="break-words">
                  {['publishedAt', 'effectiveFrom', 'effectiveTo'].includes(
                    key!,
                  ) && dayjs(value).isValid()
                    ? dayjs(value).format('YYYY-MM-DD')
                    : value}
                </dd>
              </div>
            ),
        )}
      </dl>
      <p className="whitespace-pre-wrap break-words text-sm leading-7">
        {citation.text}
      </p>
      {sourceUrl ? (
        <a
          className="mt-4 inline-flex items-center gap-1 text-xs text-accent-primary underline underline-offset-4"
          href={sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
        >
          {t('medicalAssistant.viewSource')}
          <ExternalLink className="size-3" />
        </a>
      ) : (
        <p className="mt-3 text-xs text-text-secondary">
          {t('medicalAssistant.sourceLinkUnavailable')}
        </p>
      )}
    </article>
  );
}

export function EvidencePanel({
  turn,
  activeCitationIndex,
}: {
  turn: AssistantTurn;
  activeCitationIndex: number | null;
}) {
  const { t } = useTranslation();
  const panelRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (activeCitationIndex == null) return;
    panelRef.current
      ?.querySelector<HTMLElement>(
        `[data-citation-index="${activeCitationIndex}"]`,
      )
      ?.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
  }, [activeCitationIndex, turn.request_id, turn.question_id]);
  return (
    <section
      ref={panelRef}
      aria-label={t('medicalAssistant.evidence')}
      className="space-y-3"
    >
      <h2 className="flex items-center gap-2 text-base font-semibold">
        <FileText className="size-4 text-accent-primary" />
        {t('medicalAssistant.evidence')}
      </h2>
      <p className="text-xs leading-5 text-text-secondary">
        {t('medicalAssistant.evidenceBoundary')}
      </p>
      {turn.citations.length === 0 ? (
        <p className="py-6 text-sm text-text-secondary">
          {t('medicalAssistant.noEvidence')}
        </p>
      ) : (
        turn.citations.map((citation) => (
          <CitationEvidence
            key={`${citation.index}-${citation.chunk_id}`}
            citation={citation}
            active={citation.index === activeCitationIndex}
          />
        ))
      )}
    </section>
  );
}
