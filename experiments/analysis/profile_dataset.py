"""Profile the original snapshot without copying sensitive values to reports."""

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import statistics
from urllib.parse import urlparse


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def privacy_flags(text):
    # Candidates for local review, not a claim of complete PII detection.
    patterns = {
        "national_id_candidate": r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
        "mobile_candidate": r"(?<!\d)1[3-9]\d{9}(?!\d)",
        "personal_record_context": r"患者姓名|身份证号码|社会保障卡号|就诊记录|病历号|姓名[：:]",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def load_records(path):
    records, errors, blanks = [], [], 0
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                blanks += 1
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("not_object")
                records.append((line_no, row))
            except (ValueError, TypeError):
                errors.append(line_no)
    return records, errors, blanks


def profile(path):
    numbered, errors, blanks = load_records(path)
    records = [row for _, row in numbered]
    fields = sorted(set().union(*(row.keys() for row in records)))
    result = {
        "analysis_date": date.today().isoformat(),
        "file_name": Path(path).name,
        "bytes": Path(path).stat().st_size,
        "sha256": fingerprint(path),
        "records": len(records),
        "invalid_json_lines": errors,
        "blank_lines": blanks,
        "fields": {
            key: {
                "types": dict(Counter(type(row.get(key)).__name__ for row in records)),
                "missing": sum(key not in row for row in records),
                "empty": sum(row.get(key) in (None, "", [], {}) for row in records),
            }
            for key in fields
        },
    }
    bodies = [str(row.get("政策正文") or "") for row in records]
    lengths = [len(body) for body in bodies]
    result["body_chars"] = {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths), "total": sum(lengths)}
    result["distributions"] = {
        key: dict(Counter(str(row.get(key)) for row in records)) for key in ["数据批次", "适用区域", "政策文件大类", "政策文件类型", "效力状态", "是否公开发布", "版本号", "是否最新版本"]
    }
    result["duplicates"] = {}
    for kind, values in {
        "system_id": [str(row.get("系统ID")) for row in records],
        "exact_body": bodies,
        "whitespace_normalized_body": [re.sub(r"\s+", "", body) for body in bodies],
        "title": [row.get("文件名称") for row in records],
        "source_url": [(row.get("数据来源路径") or {}).get("原始文件路径") for row in records],
        "document_number": [row.get("文号") for row in records],
    }.items():
        groups = defaultdict(list)
        for (_, row), value in zip(numbered, values):
            if value:
                groups[value].append(str(row.get("系统ID")))
        duplicates = [ids for ids in groups.values() if len(ids) > 1]
        result["duplicates"][kind] = {"groups": len(duplicates), "extra_records": sum(len(g) - 1 for g in duplicates), "id_groups": duplicates}
    result["dates"] = {}
    for key in ["发文日期", "生效日期", "废止日期"]:
        valid, invalid, future = [], [], []
        for _, row in numbered:
            if not row.get(key):
                continue
            try:
                parsed = date.fromisoformat(row[key])
                valid.append(parsed.isoformat())
                if parsed > date.today():
                    future.append(str(row["系统ID"]))
            except (ValueError, TypeError):
                invalid.append(str(row["系统ID"]))
        result["dates"][key] = {"min": min(valid, default=None), "max": max(valid, default=None), "invalid_ids": invalid, "future_ids": future}
    flags = defaultdict(list)
    domains = Counter()
    nested = defaultdict(Counter)
    topic_count = topic_exact = topic_normalized = 0
    attachment_count = 0
    for _, row in numbered:
        sid, body = str(row["系统ID"]), row.get("政策正文", "")
        for flag in privacy_flags(body):
            flags[flag].append(sid)
        checks = {
            "body_under_100_chars": len(body.strip()) < 100,
            "replacement_character": "\ufffd" in body,
            "html_markup": bool(re.search(r"</?(?:div|table|tr|td|span|p)\b", body)),
            "hash_like_title": bool(re.fullmatch(r"[a-fA-F0-9-]{32,36}", row["文件名称"])),
            "percentage": bool(re.search(r"\d+(?:\.\d+)?\s*[%％]", body)),
            "benefit_numbers": bool(re.search(r"起付|封顶|限额|报销|支付比例", body)),
            "drug_catalog": bool(re.search(r"药品目录|药品名称|双通道", body)),
            "attachment_reference": "附件" in body,
            "table_markup": "<table" in body or bool(re.search(r"\|.*\|", body)),
            "missing_source_url": not (row.get("数据来源路径") or {}).get("原始文件路径"),
        }
        for name, matched in checks.items():
            if matched:
                flags[name].append(sid)
        url = (row.get("数据来源路径") or {}).get("原始文件路径")
        if url:
            domains[urlparse(url).hostname or "not_a_url"] += 1
        for key in ["政策内容话题树", "政策关联文件", "附件列表"]:
            for item in row.get(key, []):
                if isinstance(item, dict):
                    nested[key].update(item.keys())
        attachment_count += len(row.get("附件列表", []))
        for topic in row.get("政策内容话题树", []):
            text = topic.get("知识点原文", "")
            if text:
                topic_count += 1
                topic_exact += text in body
                topic_normalized += re.sub(r"\s+", "", text) in re.sub(r"\s+", "", body)
    result["flags"] = {k: {"count": len(v), "document_ids": v} for k, v in flags.items()}
    result["source_domains"] = dict(domains)
    result["nested_keys"] = {k: dict(v) for k, v in nested.items()}
    result["topics"] = {"nonempty": topic_count, "exact_body_substrings": topic_exact, "whitespace_normalized_substrings": topic_normalized}
    result["attachment_items"] = attachment_count
    attachments = [a for row in records for a in row.get("附件列表", []) if isinstance(a, dict)]
    result["attachment_fields_nonempty"] = {key: sum(bool(a.get(key)) for a in attachments) for key in sorted(set().union(*(a.keys() for a in attachments)))}
    full_record_privacy = defaultdict(list)
    for row in records:
        for flag in privacy_flags(json.dumps(row, ensure_ascii=False)):
            full_record_privacy[flag].append(str(row["系统ID"]))
    result["privacy_full_record"] = {k: {"count": len(v), "document_ids": v} for k, v in full_record_privacy.items()}
    title_groups = defaultdict(list)
    for row in records:
        title_groups[row["文件名称"]].append(row)
    result["same_title_different_dates"] = [[str(row["系统ID"]) for row in group] for group in title_groups.values() if len({row.get("发文日期") for row in group if row.get("发文日期")}) > 1]
    result["effective_before_publication"] = [str(row["系统ID"]) for row in records if row.get("生效日期") and row.get("发文日期") and row["生效日期"] < row["发文日期"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/documents.jiangsu_and_2city.jsonl")
    parser.add_argument("--output", default="experiments/results/dataset_profile.json")
    args = parser.parse_args()
    result = profile(args.source)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": result["records"], "sha256": result["sha256"], "output": str(path)}))


if __name__ == "__main__":
    main()
