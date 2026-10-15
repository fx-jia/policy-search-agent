import { registerNextServer } from '@/utils/register-server';
const Methods = {
  getSession: { url: '/api/v1/medical-assistant/session', method: 'get' },
  submitTurn: { url: '/api/v1/medical-assistant/turn', method: 'post' },
  clearTopic: { url: '/api/v1/medical-assistant/clear', method: 'post' },
} as const;
export default registerNextServer<keyof typeof Methods>(Methods);
