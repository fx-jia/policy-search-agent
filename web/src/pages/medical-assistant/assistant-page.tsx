import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
} from '@/components/ui/sheet';
import {
  MedicalAssistantApiError,
  useClearMedicalAssistantTopic,
  useMedicalAssistantSession,
  useSubmitMedicalAssistantTurn,
} from '@/hooks/use-medical-assistant-request';
import {
  FileText,
  Loader2,
  MessageSquarePlus,
  ShieldCheck,
  X,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { v4 as uuid } from 'uuid';
import {
  AnswerMessage,
  SourceLimitations,
  type CitationSelection,
} from './components/answer-message';
import { Composer } from './components/composer';
import { EvidencePanel } from './components/evidence-panel';
import { SearchTrace } from './components/search-trace';

const ExampleKeys = ['example1', 'example2', 'example3'];
const KnownReasons = [
  'unresolved_provider_usage',
  'usage_reconciliation_invalid',
  'session_token_budget_exhausted',
  'command_unconfirmed',
  'source_unavailable',
  'service_unavailable',
];

function ExampleQuestion({
  exampleKey,
  onChoose,
}: {
  exampleKey: string;
  onChoose: (value: string) => void;
}) {
  const { t } = useTranslation();
  function handleChoose() {
    onChoose(t(`medicalAssistant.${exampleKey}`));
  }
  return (
    <Button
      className="h-auto justify-between whitespace-normal rounded-xl p-4 text-left leading-6"
      variant="outline"
      onClick={handleChoose}
    >
      {t(`medicalAssistant.${exampleKey}`)}
      <span aria-hidden="true" className="ml-3 text-accent-primary">
        ↗
      </span>
    </Button>
  );
}

export function AssistantPage() {
  const { t } = useTranslation();
  const [question, setQuestion] = useState('');
  const [processing, setProcessing] = useState(false);
  const processingRef = useRef(false);
  const [confirming, setConfirming] = useState(false);
  const recoveryUpdatedAt = useRef(0);
  const [requestFailed, setRequestFailed] = useState(false);
  const [pendingQuestion, setPendingQuestion] = useState<{
    request_id: string;
    question: string;
  } | null>(null);
  const [selection, setSelection] = useState<CitationSelection | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [mobilePanelOpen, setMobilePanelOpen] = useState(false);
  const session = useMedicalAssistantSession(confirming);
  const submit = useSubmitMedicalAssistantTurn();
  const clear = useClearMedicalAssistantTopic();
  const data = session.data;
  const turns = data?.turns ?? [];
  const waiting = processing || data?.state === 'running';
  const canSubmit =
    data?.can_submit === true &&
    !waiting &&
    !confirming &&
    !session.isError &&
    !clear.isPending;
  const selectedTurn =
    turns.find(
      (turn) => (turn.request_id ?? turn.question_id) === selection?.request_id,
    ) ?? turns.at(-1);
  const visiblePending =
    pendingQuestion &&
    !turns.some((turn) => turn.request_id === pendingQuestion.request_id);

  useEffect(() => {
    if (
      confirming &&
      !session.isError &&
      session.dataUpdatedAt > recoveryUpdatedAt.current
    )
      setConfirming(false);
  }, [confirming, session.isError, session.dataUpdatedAt]);

  async function handleSubmit(originalQuestion: string) {
    if (!canSubmit || processingRef.current || !originalQuestion.trim()) return;
    processingRef.current = true;
    setProcessing(true);
    setRequestFailed(false);
    const body = { request_id: uuid(), question: originalQuestion };
    setPendingQuestion(body);
    let completed = false;
    try {
      const result = await submit.mutateAsync(body);
      completed = result.state === 'completed';
      setQuestion('');
    } catch (error) {
      if (error instanceof MedicalAssistantApiError) setRequestFailed(true);
      else setConfirming(true);
    }
    const recovered = await session.refetch();
    if (recovered.isError || !recovered.data) {
      recoveryUpdatedAt.current = session.dataUpdatedAt;
      setConfirming(true);
    } else {
      setConfirming(false);
      if (
        completed ||
        recovered.data.turns.some((turn) => turn.request_id === body.request_id)
      )
        setQuestion('');
      if (
        recovered.data.state !== 'running' &&
        recovered.data.state !== 'unconfirmed'
      )
        setPendingQuestion(null);
    }
    processingRef.current = false;
    setProcessing(false);
  }

  async function handleNewTopic() {
    if (
      !data ||
      processingRef.current ||
      waiting ||
      confirming ||
      clear.isPending ||
      data.state === 'unconfirmed' ||
      data.state === 'unavailable'
    )
      return;
    processingRef.current = true;
    setRequestFailed(false);
    try {
      await clear.mutateAsync({ request_id: uuid() });
      setQuestion('');
      setPendingQuestion(null);
      setSelection(null);
      setPanelOpen(false);
      setMobilePanelOpen(false);
    } catch {
      setRequestFailed(true);
      const recovered = await session.refetch();
      if (recovered.isError) {
        recoveryUpdatedAt.current = session.dataUpdatedAt;
        setConfirming(true);
      }
    } finally {
      processingRef.current = false;
    }
  }

  function handleSelectCitation(nextSelection: CitationSelection) {
    setSelection(nextSelection);
    setPanelOpen(true);
    if (window.matchMedia?.('(max-width: 1023px)').matches)
      setMobilePanelOpen(true);
  }
  function handleOpenPanel() {
    setPanelOpen(true);
    if (window.matchMedia?.('(max-width: 1023px)').matches)
      setMobilePanelOpen(true);
  }
  function handleClosePanel() {
    setPanelOpen(false);
  }
  function handleMobilePanelChange(open: boolean) {
    setMobilePanelOpen(open);
  }

  const evidenceContent = selectedTurn ? (
    <>
      <EvidencePanel
        turn={selectedTurn}
        activeCitationIndex={selection?.index ?? null}
      />
      <SearchTrace turn={selectedTurn} />
    </>
  ) : (
    <p className="text-sm text-text-secondary">
      {t('medicalAssistant.noEvidenceYet')}
    </p>
  );
  const state = data?.state;
  const reason = data?.blocked_reason;
  return (
    <div className="flex h-dvh min-h-0 flex-col bg-bg-base text-text-primary">
      <header className="shrink-0 border-b border-border-button bg-bg-base px-4 py-4 sm:px-8">
        <div className="mx-auto flex max-w-screen-2xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-accent-primary/10 text-accent-primary">
              <ShieldCheck className="size-5" />
            </div>
            <div>
              <h1 className="text-base font-semibold sm:text-lg">
                {t('medicalAssistant.title')}
              </h1>
              <p className="mt-0.5 text-xs text-text-secondary">
                {t('medicalAssistant.subtitle')}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden rounded-full border border-border-button px-3 py-1 text-xs text-text-secondary sm:inline">
              {data?.scope.label ?? t('medicalAssistant.scopeLabel')}
            </span>
            <Button
              variant="outline"
              onClick={handleNewTopic}
              disabled={
                !data ||
                waiting ||
                confirming ||
                clear.isPending ||
                state === 'unconfirmed' ||
                state === 'unavailable'
              }
            >
              <MessageSquarePlus />
              {t('medicalAssistant.newTopic')}
            </Button>
          </div>
        </div>
      </header>
      <div className="mx-auto flex min-h-0 w-full max-w-screen-2xl flex-1">
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-8">
            <div className="mx-auto max-w-3xl space-y-6">
              {state && state !== 'ready' && state !== 'running' && (
                <div
                  role="status"
                  className="rounded-xl border border-border-button bg-bg-card p-4 text-sm leading-6"
                >
                  <p className="font-semibold">
                    {t(`medicalAssistant.states.${state}`)}
                  </p>
                  <p className="mt-1 text-text-secondary">
                    {KnownReasons.includes(reason ?? '')
                      ? t(`medicalAssistant.reasons.${reason}`)
                      : t('medicalAssistant.serviceRestricted')}
                  </p>
                </div>
              )}
              {session.isError && !confirming && (
                <p
                  role="alert"
                  className="rounded-xl border border-border-button bg-bg-card p-4 text-sm"
                >
                  {t('medicalAssistant.loadFailed')}
                </p>
              )}
              {requestFailed && (
                <p role="alert" className="text-sm text-state-error">
                  {t('medicalAssistant.requestFailed')}
                </p>
              )}
              {turns.length === 0 && !visiblePending && (
                <div className="py-8 sm:py-14">
                  <p className="mb-3 text-xs font-semibold tracking-widest text-accent-primary">
                    {t('medicalAssistant.scopeLabel')}
                  </p>
                  <h2 className="mb-4 text-2xl font-semibold leading-snug sm:text-3xl">
                    {t('medicalAssistant.welcome')}
                  </h2>
                  <p className="mb-8 max-w-xl text-sm leading-7 text-text-secondary">
                    {t('medicalAssistant.introduction')}
                  </p>
                  <div className="grid gap-3">
                    {ExampleKeys.map((key) => (
                      <ExampleQuestion
                        key={key}
                        exampleKey={key}
                        onChoose={setQuestion}
                      />
                    ))}
                  </div>
                  <details className="mt-6 text-xs text-text-secondary">
                    <summary className="cursor-pointer">
                      {t('medicalAssistant.scopeDetails')}
                    </summary>
                    <SourceLimitations source={data?.scope ?? {}} />
                  </details>
                </div>
              )}
              {turns.map((turn) => (
                <section
                  key={turn.request_id ?? turn.question_id}
                  className="space-y-4"
                >
                  <div className="ml-auto max-w-[90%] rounded-2xl bg-accent-primary/10 px-5 py-3 text-sm leading-7">
                    <p className="whitespace-pre-wrap break-words">
                      {turn.question}
                    </p>
                  </div>
                  <AnswerMessage
                    turn={turn}
                    onSelectCitation={handleSelectCitation}
                  />
                </section>
              ))}
              {visiblePending && (
                <p className="ml-auto max-w-[90%] whitespace-pre-wrap break-words rounded-2xl bg-accent-primary/10 px-5 py-3 text-sm leading-7">
                  {pendingQuestion.question}
                </p>
              )}
              {(waiting || session.isPending || confirming) && (
                <p
                  role="status"
                  className="flex items-center gap-2 py-4 text-sm text-text-secondary"
                >
                  <Loader2 className="size-4 animate-spin" />
                  {t(
                    confirming
                      ? 'medicalAssistant.confirming'
                      : 'medicalAssistant.waiting',
                  )}
                </p>
              )}
            </div>
          </div>
          <footer className="shrink-0 border-t border-border-button bg-bg-base px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-4 sm:px-8">
            <div className="mx-auto max-w-3xl">
              <div className="mb-3 flex items-center justify-between">
                <p className="text-xs text-text-secondary">
                  {t('medicalAssistant.composerScope')}
                </p>
                {turns.length > 0 && (
                  <Button size="sm" variant="ghost" onClick={handleOpenPanel}>
                    <FileText />
                    {t('medicalAssistant.evidenceAndTrace')}
                  </Button>
                )}
              </div>
              <Composer
                value={question}
                onChange={setQuestion}
                onSubmit={handleSubmit}
                disabled={!canSubmit}
              />
            </div>
          </footer>
        </main>
        {panelOpen && (
          <aside className="hidden w-[360px] shrink-0 overflow-y-auto border-l border-border-button bg-bg-card p-5 lg:block xl:w-[400px]">
            <div className="mb-4 flex justify-end">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleClosePanel}
                aria-label={t('medicalAssistant.closeEvidence')}
              >
                <X />
              </Button>
            </div>
            {evidenceContent}
          </aside>
        )}
      </div>
      <Sheet open={mobilePanelOpen} onOpenChange={handleMobilePanelChange}>
        <SheetContent
          className="w-[92vw] max-w-[440px] overflow-y-auto"
          closeIcon
        >
          <SheetTitle className="mb-2">
            {t('medicalAssistant.evidenceAndTrace')}
          </SheetTitle>
          <SheetDescription className="mb-6">
            {t('medicalAssistant.evidenceBoundary')}
          </SheetDescription>
          {evidenceContent}
        </SheetContent>
      </Sheet>
    </div>
  );
}
