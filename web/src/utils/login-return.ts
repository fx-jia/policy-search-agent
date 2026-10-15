const AssistantPath = '/medical-assistant';

export function getLoginReturnPath(search: string): string {
  const returnTo = new URLSearchParams(search).get('returnTo');
  if (
    !returnTo ||
    !returnTo.startsWith('/') ||
    returnTo.startsWith('//') ||
    /[\s\\]/.test(returnTo) ||
    [...returnTo].some((character) => {
      const code = character.charCodeAt(0);
      return code < 32 || code === 127;
    })
  ) {
    return '/';
  }
  const target = new URL(returnTo, 'https://assistant.invalid');
  if (
    target.origin !== 'https://assistant.invalid' ||
    ![AssistantPath, `${AssistantPath}/`].includes(target.pathname)
  ) {
    return '/';
  }
  return target.pathname + target.search + target.hash;
}

export function getLoginRedirectPath(
  pathname: string,
  search = '',
  hash = '',
  baseUrl = '',
): string {
  const base = baseUrl.replace(/^\/+|\/+$/g, '');
  const prefix = base ? `/${base}` : '';
  const routePath =
    prefix && pathname.startsWith(`${prefix}/`)
      ? pathname.slice(prefix.length)
      : pathname;
  const returnTo = getLoginReturnPath(
    new URLSearchParams({ returnTo: routePath + search + hash }).toString(),
  );
  const loginPath = `${prefix}/login`;
  return returnTo === '/'
    ? loginPath
    : `${loginPath}?${new URLSearchParams({ returnTo })}`;
}
