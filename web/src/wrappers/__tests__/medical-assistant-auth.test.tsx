import { fireEvent, render, screen } from '@testing-library/react';
import { useEffect } from 'react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router';
import MedicalAssistantAuth, {
  useMedicalAssistantIdentity,
} from '../medical-assistant-auth';

let mockAuthorization = '';
jest.mock('@/utils/authorization-util', () => ({
  __esModule: true,
  default: { getAuthorization: () => mockAuthorization },
}));

let mounts = 0;
let identities: number[] = [];
function SessionView() {
  const identity = useMedicalAssistantIdentity();
  useEffect(() => {
    mounts += 1;
    identities.push(identity);
  }, [identity]);
  return <div>Authorized session</div>;
}

function LoginDestination() {
  const location = useLocation();
  return <div>{location.pathname + location.search}</div>;
}

function renderAssistant() {
  return render(
    <MemoryRouter initialEntries={['/medical-assistant']}>
      <Routes>
        <Route
          path="/medical-assistant"
          element={
            <MedicalAssistantAuth>
              <SessionView />
            </MedicalAssistantAuth>
          }
        />
        <Route path="/login" element={<LoginDestination />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  mockAuthorization = '';
  mounts = 0;
  identities = [];
});

it('does not mount session readers before authentication', () => {
  renderAssistant();
  expect(mounts).toBe(0);
  expect(screen.queryByText('Authorized session')).not.toBeInTheDocument();
  expect(
    screen.getByText('/login?returnTo=%2Fmedical-assistant'),
  ).toBeInTheDocument();
});

it('unmounts protected data when another tab logs out', () => {
  mockAuthorization = 'test-user-a';
  renderAssistant();
  expect(screen.getByText('Authorized session')).toBeInTheDocument();
  mockAuthorization = '';
  fireEvent(window, new Event('storage'));
  expect(screen.queryByText('Authorized session')).not.toBeInTheDocument();
});

it('remounts session readers when the authenticated identity changes', () => {
  mockAuthorization = 'test-user-a';
  renderAssistant();
  expect(mounts).toBe(1);
  mockAuthorization = 'test-user-b';
  fireEvent(window, new Event('storage'));
  expect(mounts).toBe(2);
  expect(identities[1]).not.toBe(identities[0]);
});
