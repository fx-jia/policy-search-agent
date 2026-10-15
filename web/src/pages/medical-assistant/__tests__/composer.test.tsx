import { act, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { Composer } from '../components/composer';

jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));
function Harness({
  onSubmit,
  disabled = false,
}: {
  onSubmit: (question: string) => Promise<void>;
  disabled?: boolean;
}) {
  const [value, setValue] = useState('');
  return (
    <Composer
      value={value}
      onChange={setValue}
      onSubmit={onSubmit}
      disabled={disabled}
    />
  );
}
it('ignores blank, shift-enter and composing enter', async () => {
  const send = jest.fn().mockResolvedValue(undefined);
  render(<Harness onSubmit={send} />);
  const input = screen.getByRole('textbox');
  fireEvent.keyDown(input, { key: 'Enter' });
  fireEvent.change(input, { target: { value: '上海医保' } });
  fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
  fireEvent.compositionStart(input);
  fireEvent.keyDown(input, { key: 'Enter' });
  expect(send).not.toHaveBeenCalled();
  fireEvent.compositionEnd(input);
  await act(async () => {
    fireEvent.keyDown(input, { key: 'Enter' });
  });
  expect(send).toHaveBeenCalledWith('上海医保');
});
it('locks synchronously against duplicate enter and clicks', () => {
  const send = jest.fn(() => new Promise<void>(() => {}));
  render(<Harness onSubmit={send} />);
  const input = screen.getByRole('textbox');
  fireEvent.change(input, { target: { value: '第一轮原话' } });
  fireEvent.keyDown(input, { key: 'Enter' });
  fireEvent.keyDown(input, { key: 'Enter' });
  fireEvent.click(screen.getByRole('button'));
  expect(send).toHaveBeenCalledTimes(1);
});
it('obeys server disabled state', () => {
  const send = jest.fn().mockResolvedValue(undefined);
  render(<Harness onSubmit={send} disabled />);
  expect(screen.getByRole('textbox')).toBeDisabled();
  expect(screen.getByRole('button')).toBeDisabled();
});
