import { getLoginRedirectPath, getLoginReturnPath } from '../login-return';

describe('medical assistant login return', () => {
  it('returns to the assistant with its selected detail after login', () => {
    const query = new URLSearchParams({
      returnTo: '/medical-assistant?view=sources#citation-1',
    }).toString();
    expect(getLoginReturnPath(query)).toBe(
      '/medical-assistant?view=sources#citation-1',
    );
  });

  it.each([
    'https://elsewhere.example/medical-assistant',
    '//elsewhere.example/medical-assistant',
    '/\\elsewhere.example/medical-assistant',
    'javascript:alert(1)',
    '/medical-assistant/../../login',
    '/medical-assistant%2f..%2flogin',
    '/medical-assistant\n',
    '/datasets',
  ])('rejects an unsafe or unrelated return target %s', (returnTo) => {
    expect(
      getLoginReturnPath(new URLSearchParams({ returnTo }).toString()),
    ).toBe('/');
  });

  it('keeps the normal login destination when no assistant was requested', () => {
    expect(getLoginReturnPath('')).toBe('/');
    expect(getLoginRedirectPath('/datasets')).toBe('/login');
  });

  it('preserves the assistant entry when authentication expires', () => {
    expect(getLoginRedirectPath('/medical-assistant', '?view=sources')).toBe(
      '/login?returnTo=%2Fmedical-assistant%3Fview%3Dsources',
    );
  });

  it('keeps the deployment basename outside the router return target', () => {
    expect(
      getLoginRedirectPath('/ragflow/medical-assistant', '', '', '/ragflow/'),
    ).toBe('/ragflow/login?returnTo=%2Fmedical-assistant');
  });
});
