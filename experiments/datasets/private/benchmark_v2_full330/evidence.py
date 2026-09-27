"""Bind authored candidates to local immutable evidence and render review artifacts.

No network, embedding, or model calls. Re-run from any working directory.
"""
import collections
import difflib
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RAW = ROOT / "data/documents.jiangsu_and_2city.jsonl"
FROZEN = ROOT / "experiments/datasets/private/ragflow_corpus1000_v1/frozen_corpus.json"
REVIEW = ROOT / "experiments/results/private/native-agent-high-compose-fix-v1-full-review/config.json"
OLD = ROOT / "experiments/datasets/private/expansion_v1/benchmark_150_reviewed_v3.jsonl"


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def norm(value):
    # Formatting-insensitive character alignment only, not semantic equivalence.
    return "".join(c for c in unicodedata.normalize("NFKC", value) if c.isalnum())


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


raw_rows = [json.loads(s) for s in RAW.read_text().splitlines()]
records = {str(r["系统ID"]): r for r in raw_rows}
row_numbers = {str(r["系统ID"]): n for n, r in enumerate(raw_rows, 1)}
frozen = json.loads(FROZEN.read_text())
assert sha(RAW.read_bytes()) == frozen["source_sha256"]
frozen_map = {str(d["source_document_id"]): d for d in frozen["documents"]}
review = json.loads(REVIEW.read_text())
safe_chunks = collections.defaultdict(list)
for cid, entry in review["review"]["entries"].items():
    assert sha(entry["sanitized_content"]) == entry["safe_sha256"], cid
    safe_chunks[str(entry["source_document_id"])].append((cid, entry, norm(entry["sanitized_content"])))
old = [json.loads(s) for s in OLD.read_text().splitlines()]


def bind_evidence(spec, number):
    sid = spec["source_id"]
    assert sid in frozen_map, sid
    r = records[sid]
    body = r["政策正文"]
    if "lines" in spec:
        lines = body.splitlines(keepends=True)
        first, last = spec["lines"]
        assert 0 <= first <= last < len(lines), (sid, first, last)
        start = sum(map(len, lines[:first]))
        end = sum(map(len, lines[:last + 1]))
        while end > start and body[end-1] in "\r\n":
            end -= 1
    else:
        assert body.count(spec["start_anchor"]) == 1, spec
        start = body.index(spec["start_anchor"])
        end = body.index(spec["end_anchor"], start)
        while body[end-1].isspace():
            end -= 1
    text = body[start:end]
    needle = norm(text)
    masks = []
    for cid, entry, hay in safe_chunks[sid]:
        # Exact normalized contiguous matches. Union supports quotes split at chunk boundaries.
        if needle in hay:
            blocks = [[0, len(needle)]]
        else:
            blocks = [[m.a, m.a + m.size] for m in difflib.SequenceMatcher(None, needle, hay, autojunk=False).get_matching_blocks() if m.size >= 8]
        if blocks:
            masks.append((cid, entry, blocks))
    covered = set()
    links = []
    remaining = list(masks)
    while remaining:
        best = max(remaining, key=lambda v: len({i for a,b in v[2] for i in range(a,b)} - covered))
        positions = {i for a,b in best[2] for i in range(a,b)}
        if not positions - covered:
            break
        covered |= positions
        remaining.remove(best)
        links.append({"chunk_id": best[0], "raw_sha256": best[1]["raw_sha256"],
                      "safe_sha256": best[1]["safe_sha256"], "normalized_quote_spans": best[2]})
    missing = []
    for i in range(len(needle)):
        if i not in covered:
            if missing and missing[-1][1] == i:
                missing[-1][1] += 1
            else:
                missing.append([i, i+1])
    return {
        "evidence_id": f"e{number}", "source_document_id": sid,
        "source_title": r["文件名称"], "source_url": r.get("数据来源路径", {}).get("原始文件路径"),
        "region": r.get("适用区域"), "metadata_publication_date": r.get("发文日期"),
        "date_note": "日期为原始数据字段，未等同于实际签发日/生效日；以题干及正文时间为准。",
        "jsonl_line": row_numbers[sid], "field": "政策正文", "start": start, "end": end,
        "text": text, "body_sha256": sha(body), "quote_sha256": sha(text),
        "ragflow_document_id": frozen_map[sid]["document_id"], "dataset_id": frozen_map[sid]["dataset_id"],
        "indexed_safe_chunk_links": links,
        "safe_text_alignment": {"method": "NFKC + alphanumeric; union of exact contiguous matching blocks >=8 chars",
                                "covered_characters": len(covered), "total_characters": len(needle),
                                "coverage": len(covered)/len(needle),
                                "uncovered_spans": [{"start": a, "end": b, "text": needle[a:b]} for a,b in missing]},
    }


