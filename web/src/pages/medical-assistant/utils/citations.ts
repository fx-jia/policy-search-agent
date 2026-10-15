import type { AssistantCitation } from '@/interfaces/database/medical-assistant';

/** The raw index belongs to this turn's evidence; never resolve another turn. */
export function citationMarkdown(
  answer: string,
  citations: AssistantCitation[],
  unavailable = '引用暂不可用',
): string {
  return answer.replace(
    /##(-?\d+)\$\$|\[ID:(-?\d+)\]/g,
    (_marker, hashIndex, idIndex) => {
      const index = Number(hashIndex ?? idIndex);
      return Number.isSafeInteger(index) &&
        index >= 0 &&
        citations.some((citation) => citation.index === index)
        ? `[${index + 1}](medical-citation:${index})`
        : unavailable;
    },
  );
}

export function safeSourceUrl(url: string | null | undefined): string | null {
  if (!url || !/^https?:\/\//i.test(url)) return null;
  try {
    const parsed = new URL(url);
    return ['http:', 'https:'].includes(parsed.protocol) &&
      !parsed.username &&
      !parsed.password
      ? parsed.href
      : null;
  } catch {
    return null;
  }
}
