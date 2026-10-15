import { BackendVariant } from '@/utils/backend-variant';
import MedicalAssistantAuth from '@/wrappers/medical-assistant-auth';
import { useTranslation } from 'react-i18next';
import { AssistantPage } from './assistant-page';

function UnsupportedBackend() {
  const { t } = useTranslation();
  return (
    <main className="flex min-h-dvh items-center justify-center bg-bg-base p-6 text-text-primary">
      <div className="max-w-lg rounded-2xl border border-border-button bg-bg-card p-8">
        <h1 className="mb-3 text-xl font-semibold">
          {t('medicalAssistant.title')}
        </h1>
        <p className="text-sm leading-7 text-text-secondary">
          {t('medicalAssistant.goUnsupported')}
        </p>
      </div>
    </main>
  );
}

export default function MedicalAssistant() {
  return (
    <MedicalAssistantAuth>
      <BackendVariant go={<UnsupportedBackend />} python={<AssistantPage />} />
    </MedicalAssistantAuth>
  );
}
