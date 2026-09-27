"""Deterministic Chinese bigram BM25 for offline pipeline smoke checks only."""

from collections import Counter, defaultdict
import math
import re


def tokenize(text):
    tokens = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:\.\d+)?", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                tokens.append(part)
        else:
            tokens.append(part)
    return tokens


class LocalBM25:
    def __init__(self, records, chunk_chars=600, k1=1.2, b=0.75):
        self.chunks, self.lengths = [], []
        self.postings = defaultdict(list)
        self.k1, self.b = k1, b
        for _, row in records:
            sid, body = str(row["系统ID"]), row["政策正文"]
            for start in range(0, len(body), chunk_chars):
                end = min(start + chunk_chars, len(body))
                chunk = {"id": f"local:{sid}:{start}:{end}", "document_id": sid, "source_document_id": sid, "start": start, "end": end}
                terms = Counter(tokenize(row["文件名称"] + "\n" + body[start:end]))
                idx = len(self.chunks)
                self.chunks.append(chunk)
                self.lengths.append(sum(terms.values()))
                for term, count in terms.items():
                    self.postings[term].append((idx, count))
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)

    def retrieve(self, question, limit):
        scores = defaultdict(float)
        n = len(self.chunks)
        for term in sorted(set(tokenize(question))):
            postings = self.postings.get(term, [])
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for idx, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.average_length)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores, key=lambda idx: (-scores[idx], self.chunks[idx]["id"]))[:limit]
        return [dict(self.chunks[idx], similarity=scores[idx], rank=rank) for rank, idx in enumerate(ranked, 1)]
