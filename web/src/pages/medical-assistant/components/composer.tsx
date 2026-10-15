import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ArrowUp } from 'lucide-react';
import { useRef, useState, type ChangeEvent, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question: string) => Promise<void>;
  disabled: boolean;
}

export function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
}: ComposerProps) {
  const { t } = useTranslation();
  const composing = useRef(false);
  const sending = useRef(false);
  const [submitting, setSubmitting] = useState(false);
  function handleChange(event: ChangeEvent<HTMLTextAreaElement>) {
    onChange(event.target.value);
  }
  function handleCompositionStart() {
    composing.current = true;
  }
  function handleCompositionEnd() {
    composing.current = false;
  }
  async function handleSubmit() {
    if (disabled || sending.current || composing.current || !value.trim())
      return;
    sending.current = true;
    setSubmitting(true);
    try {
      await onSubmit(value);
    } finally {
      sending.current = false;
      setSubmitting(false);
    }
  }
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key !== 'Enter' ||
      event.shiftKey ||
      event.nativeEvent.isComposing ||
      composing.current ||
      event.keyCode === 229
    )
      return;
    event.preventDefault();
    void handleSubmit();
  }
  return (
    <div className="rounded-2xl border border-border-button bg-bg-card p-3 shadow-sm">
      <Textarea
        aria-label={t('medicalAssistant.question')}
        placeholder={t('medicalAssistant.placeholder')}
        className="min-h-24 border-0 bg-transparent text-base shadow-none focus-visible:ring-0"
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onCompositionStart={handleCompositionStart}
        onCompositionEnd={handleCompositionEnd}
        disabled={disabled || submitting}
        rows={3}
      />
      <div className="flex items-center justify-between gap-3 pt-2">
        <span className="text-xs text-text-secondary">
          {t('medicalAssistant.keyboardHint')}
        </span>
        <Button
          size="lg"
          disabled={disabled || submitting || !value.trim()}
          onClick={handleSubmit}
        >
          <ArrowUp aria-hidden="true" />
          {t('medicalAssistant.send')}
        </Button>
      </div>
    </div>
  );
}
