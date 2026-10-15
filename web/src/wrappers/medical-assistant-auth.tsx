import authorizationUtil from '@/utils/authorization-util';
import { getLoginRedirectPath } from '@/utils/login-return';
import {
  createContext,
  Fragment,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from 'react';
import { Navigate, useLocation } from 'react-router';

const IdentityContext = createContext(0);
let IdentityGeneration = 0;

export function useMedicalAssistantIdentity(): number {
  return useContext(IdentityContext);
}

export default function MedicalAssistantAuth({
  children,
}: {
  children: ReactNode;
}) {
  const location = useLocation();
  const [identity, setIdentity] = useState(() => ({
    token: authorizationUtil.getAuthorization() || '',
    version: ++IdentityGeneration,
  }));

  useEffect(() => {
    function handleIdentityChange() {
      const token = authorizationUtil.getAuthorization() || '';
      setIdentity((previous) =>
        previous.token === token
          ? previous
          : { token, version: ++IdentityGeneration },
      );
    }
    window.addEventListener('storage', handleIdentityChange);
    window.addEventListener('focus', handleIdentityChange);
    return () => {
      window.removeEventListener('storage', handleIdentityChange);
      window.removeEventListener('focus', handleIdentityChange);
    };
  }, []);

  if (!identity.token) {
    return (
      <Navigate
        to={getLoginRedirectPath(
          location.pathname,
          location.search,
          location.hash,
        )}
        replace
      />
    );
  }
  return (
    <IdentityContext.Provider value={identity.version}>
      <Fragment key={identity.version}>{children}</Fragment>
    </IdentityContext.Provider>
  );
}
