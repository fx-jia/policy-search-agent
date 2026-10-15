import type {
  AssistantSession,
  AssistantSubmission,
} from '@/interfaces/database/medical-assistant';
import type {
  ClearMedicalAssistantTopic,
  SubmitMedicalAssistantTurn,
} from '@/interfaces/request/medical-assistant';
import medicalAssistantService from '@/services/medical-assistant-service';
import { useMedicalAssistantIdentity } from '@/wrappers/medical-assistant-auth';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';

export const MedicalAssistantKeys = {
  session: (identity: number) =>
    ['medical-assistant', 'session', identity] as const,
};

export class MedicalAssistantApiError extends Error {
  constructor(public code: number) {
    super('Medical assistant request rejected');
  }
}

function readEnvelope<T>(response: { data: { code: number; data: T } }): T {
  if (response.data.code !== 0)
    throw new MedicalAssistantApiError(response.data.code);
  return response.data.data;
}

async function readRequest<T>(
  request: Promise<{ data: { code: number; data: T } }>,
): Promise<T> {
  try {
    return readEnvelope<T>(await request);
  } catch (error) {
    if (
      isAxiosError(error) &&
      error.response &&
      error.response.status >= 400 &&
      error.response.status < 500
    ) {
      const code = error.response.data?.code;
      throw new MedicalAssistantApiError(
        typeof code === 'number' ? code : error.response.status,
      );
    }
    throw error;
  }
}

const SilentRequest = { skipGlobalErrorNotification: true };

export function useMedicalAssistantSession(confirming = false) {
  const identity = useMedicalAssistantIdentity();
  return useQuery<AssistantSession>({
    queryKey: MedicalAssistantKeys.session(identity),
    queryFn: async () =>
      readRequest<AssistantSession>(
        medicalAssistantService.getSession(SilentRequest, true),
      ),
    retry: false,
    gcTime: 0,
    staleTime: 0,
    refetchOnMount: 'always',
    refetchInterval: (query) =>
      confirming || query.state.data?.state === 'running' ? 4000 : false,
  });
}

export function useSubmitMedicalAssistantTurn() {
  const identity = useMedicalAssistantIdentity();
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (body: SubmitMedicalAssistantTurn) =>
      readRequest<AssistantSubmission>(
        medicalAssistantService.submitTurn(
          { ...SilentRequest, data: body },
          true,
        ),
      ),
    retry: false,
    onSuccess: (submission) => {
      if (submission.state !== 'completed') return;
      client.setQueryData<AssistantSession>(
        MedicalAssistantKeys.session(identity),
        (session) =>
          session
            ? {
                ...session,
                turns: [
                  ...session.turns.filter(
                    (turn) => turn.request_id !== submission.turn.request_id,
                  ),
                  submission.turn,
                ],
              }
            : session,
      );
    },
  });
}

export function useClearMedicalAssistantTopic() {
  const identity = useMedicalAssistantIdentity();
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (body: ClearMedicalAssistantTopic) =>
      readRequest<AssistantSession>(
        medicalAssistantService.clearTopic(
          { ...SilentRequest, data: body },
          true,
        ),
      ),
    retry: false,
    onSuccess: (session) =>
      client.setQueryData(MedicalAssistantKeys.session(identity), session),
  });
}
