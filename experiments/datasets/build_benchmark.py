"""Resolve authored question specifications to immutable source evidence spans."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from experiments.analysis.profile_dataset import fingerprint, load_records, privacy_flags


def validate_sample(sample, records, dataset_version):
    if sample.get("dataset_version") != dataset_version:
        raise ValueError("Dataset fingerprint mismatch")
    if not isinstance(sample.get("question"), str) or not sample["question"].strip():
        raise ValueError("Question must be nonempty")
    relevant = sample.get("relevant_document_ids")
    if not isinstance(relevant, list) or not all(isinstance(sid, str) and sid in records for sid in relevant):
        raise ValueError("Unknown or invalid source document IDs")
    if len(relevant) != len(set(relevant)):
        raise ValueError("Duplicate relevance labels")
    if sample.get("answerable") is True and (not relevant or not sample.get("evidence")):
        raise ValueError("Answerable sample needs labels and evidence")
    if sample.get("answerable") is False and relevant:
        raise ValueError("Unanswerable sample must not have positive labels")
    if type(sample.get("answerable")) is not bool:
        raise ValueError("answerable must be explicitly labelled")
    for evidence in sample.get("evidence", []):
        sid, start, end = evidence["source_document_id"], evidence["start"], evidence["end"]
        if sid not in relevant or type(start) is not int or type(end) is not int:
            raise ValueError("Invalid evidence coordinates")
        body = records[sid]["政策正文"]
        if not 0 <= start < end <= len(body) or body[start:end] != evidence["text"]:
            raise ValueError("Evidence does not match the original body")
        if fingerprint_text(body) != evidence["body_sha256"]:
            raise ValueError("Source body fingerprint mismatch")
    if set(relevant) != {e["source_document_id"] for e in sample.get("evidence", [])}:
        raise ValueError("Every positive label needs evidence")


def fingerprint_text(text):
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def _evidence(records, lines, sid, first, last, role="answer"):
    sid = str(sid)
    row = records[sid]
    body = row["政策正文"]
    paragraphs = body.splitlines(keepends=True)
    if type(first) is not int or type(last) is not int or not 0 <= first <= last < len(paragraphs):
        raise ValueError("Invalid source line range")
    start = sum(map(len, paragraphs[:first]))
    excerpt = "".join(paragraphs[first : last + 1]).rstrip("\r\n")
    if not excerpt.strip() or privacy_flags(excerpt):
        raise ValueError(f"Evidence in document {sid} is empty or requires privacy review")
    return {
        "source_document_id": sid,
        "jsonl_line": lines[sid],
        "field": "政策正文",
        "paragraph_index": first,
        "last_paragraph_index": last,
        "start": start,
        "end": start + len(excerpt),
        "text": excerpt,
        "body_sha256": fingerprint_text(body),
        "source_url": row["数据来源路径"].get("原始文件路径"),
        "role": role,
    }


def _preview(samples, records):
    preview = [
        "# 扩展候选题人工审核",
        "",
        "以下是构造题目时选定的原文证据，不是检索器返回结果。仅依据本地历史快照，不表示当前政策仍然有效。",
        "核对问题的地区、年份、人群、条件和全部子问是否有依据。参考答案采用答案角色证据的原文摘录，不是模型生成答案。",
        "正文坐标是 Python Unicode 字符 [start, end)，不是字节或 Token；阅读版仅折叠空白，原始换行保留在 JSONL 中。",
        "来源标题等元数据是审核依据；评测时须检查实际检索上下文是否提供这些信息，不能按金标注入。",
        "",
    ]
    for sample in samples:
        preview.extend([f"## {sample['question_id']}：{sample['question']}", "", f"类型：{sample['question_type']}；状态：待人工审核。", ""])
        for e in sample["evidence"]:
            row = records[e["source_document_id"]]
            preview.extend(
                [
                    f"来源：{row['文件名称']}",
                    f"地区：{row['适用区域']}；发文日期：{row['发文日期']}；系统 ID：{e['source_document_id']}。",
                    f"证据角色：{e['role']}；JSONL 第 {e['jsonl_line']} 行；正文 [{e['start']}, {e['end']})。",
                    "",
                    "> " + re.sub(r"\s+", " ", e["text"]).strip(),
                    "",
                ]
            )
        preview.extend(["审核结论：待填写（通过 / 部分支持 / 不通过）。", "", "问题或修改建议：", "", "其他可支持答案的文档 ID（如发现）：", ""])
    return "\n".join(preview)


def build(source, specs_path, output, base_benchmark=None):
    numbered, errors, _ = load_records(source)
    if errors:
        raise ValueError("Source has parse errors")
    records = {str(row["系统ID"]): row for _, row in numbered}
    lines = {str(row["系统ID"]): line for line, row in numbered}
    spec_file = json.loads(Path(specs_path).read_text())
    version = fingerprint(source)
    if spec_file["source_sha256"] != version:
        raise ValueError("Question specifications target a different source snapshot")
    base = []
    if base_benchmark:
        if spec_file.get("base_benchmark_sha256") != fingerprint(base_benchmark):
            raise ValueError("Base benchmark fingerprint mismatch")
        base = [json.loads(line) for line in Path(base_benchmark).read_text().splitlines() if line.strip()]
        for sample in base:
            validate_sample(sample, records, version)
            if sample.get("review_status") != "human_approved" or not sample.get("reviewer"):
                raise ValueError("Base benchmark must be human approved")
    elif spec_file.get("base_benchmark_sha256"):
        raise ValueError("These specifications require the approved base benchmark")
    if "expected_new_questions" in spec_file and len(spec_file["questions"]) != spec_file["expected_new_questions"]:
        raise ValueError("Incomplete question specifications")
    seen_ids = {s["question_id"] for s in base}
    seen_questions = {re.sub(r"\s+", "", s["question"]) for s in base}
    if len(seen_ids) != len(base) or len(seen_questions) != len(base):
        raise ValueError("Duplicate base question")
    samples = []
    for number, spec in enumerate(spec_file["questions"], 1):
        ranges = spec.get("evidence_line_ranges")
        if ranges is None:
            ranges = [(sid, paragraph, paragraph) for sid, paragraph in spec["evidence_paragraphs"]]
        evidence = [_evidence(records, lines, *r) for r in ranges]
        if not evidence:
            raise ValueError("Question needs answer evidence")
        evidence.extend(_evidence(records, lines, *r, role="document_context") for r in spec.get("context_line_ranges", []))
        relevant = list(dict.fromkeys(e["source_document_id"] for e in evidence))
        primary = records[relevant[0]]
        multi = len(relevant) > 1
        sample = {
            "question_id": spec.get("question_id", f"med-{number:03}"),
            "question": spec["question"],
            "reference_answer": "\n".join(e["text"] for e in evidence if e["role"] == "answer"),
            "reference_answer_format": "verbatim_answer_evidence",
            "source_document_id": relevant[0] if not multi else None,
            "relevant_document_ids": relevant,
            "relevant_chunk_ids": [],
            "question_type": spec["question_type"],
            "region": primary["适用区域"] if not multi else None,
            "policy_date": primary["发文日期"] if not multi else None,
            "policy_metadata": [
                {
                    "source_document_id": sid,
                    "document_title": records[sid]["文件名称"],
                    "region": records[sid]["适用区域"],
                    "publication_date": records[sid]["发文日期"],
                    "effective_date": records[sid]["生效日期"],
                    "expiry_date": records[sid]["废止日期"],
                    "status_in_snapshot": records[sid]["效力状态"],
                    "version": records[sid]["版本号"],
                }
                for sid in relevant
            ],
            "answerable": True,
            "evidence": evidence,
            "dataset_version": version,
            "split": "candidate" if base_benchmark else "smoke",
            "review_status": "pending_human_review",
            "reviewer": None,
            "qrels_scope": "known_positive_documents_only",
            "creation_method": "assistant_authored_question_and_exact_source_span",
            "context_requirement": "Actual retrieved context/metadata must identify the policy and support every subquestion; never inject gold evidence.",
        }
        if base_benchmark:
            sample["question_family_id"] = sample["question_id"]
            sample["split_group_review"] = "pending_policy_family_and_near_duplicate_audit"
        validate_sample(sample, records, version)
        normalized_question = re.sub(r"\s+", "", sample["question"])
        if sample["question_id"] in seen_ids or normalized_question in seen_questions:
            raise ValueError("Duplicate question ID or question text")
        seen_ids.add(sample["question_id"])
        seen_questions.add(normalized_question)
        if any(sample["question"] in row["政策正文"] for row in records.values()):
            raise ValueError("Exact question already occurs in the corpus; review leakage")
        samples.append(sample)
    target = Path(output)
    combined = base + samples
    documents = {sid for s in combined for sid in s["relevant_document_ids"]}
    summary = {
        "source_sha256": version,
        "specifications_sha256": fingerprint(specs_path),
        "base_sha256": fingerprint(base_benchmark) if base_benchmark else None,
        "builder_sha256": fingerprint(__file__),
        "total_questions": len(combined),
        "new_candidates": len(samples),
        "review_status": dict(Counter(s["review_status"] for s in combined)),
        "positive_source_documents": len(documents),
        "source_regions": dict(Counter(records[sid]["适用区域"] for sid in documents)),
        "source_categories": dict(Counter(records[sid]["政策文件大类"] for sid in documents)),
        "question_types": dict(Counter(s["question_type"] for s in combined)),
        "multi_document_questions": sum(len(s["relevant_document_ids"]) > 1 for s in combined),
        "answerable_questions": sum(s["answerable"] for s in combined),
        "qrels_scope": "known_positive_documents_only",
        "checks": {"evidence_spans": "passed", "source_hashes": "passed", "unique_ids_and_questions": "passed", "new_exact_question_in_corpus": 0},
        "retrieval_evaluation": "Not Run",
        "external_model_api_calls": 0,
        "split_plan": spec_file.get("split_plan"),
    }
    data = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in combined)
    summary["benchmark_sha256"] = fingerprint_text(data)
    artifacts = {target: data, target.with_name("build_summary.json"): json.dumps(summary, ensure_ascii=False, indent=2) + "\n"}
    index = [
        "# Benchmark 审核入口",
        "",
        f"共 {len(combined)} 题，已有 {len(base)} 题人工通过，本次新增 {len(samples)} 题待审核。",
        "",
        "新题尚未划分开发集和测试集。旧题保留 smoke 分组，不能作为未见测试题。",
        "",
        "每题核对全部子问、原文证据和适用条件；发现其他支持文档时记录 ID。原题和未来改写应归为同一题族。",
        "",
    ]
    for start in range(0, len(samples), 10):
        batch = samples[start : start + 10]
        filename = f"review_{batch[0]['question_id']}_{batch[-1]['question_id']}.md"
        artifacts[target.with_name(filename)] = _preview(batch, records)
        index.append(f"- [{batch[0]['question_id']}～{batch[-1]['question_id']}]({filename})")
    artifacts[target.with_name("review_index.md")] = "\n".join(index) + "\n"
    for path in artifacts:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite dataset or review: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in artifacts.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(contents)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/documents.jiangsu_and_2city.jsonl")
    parser.add_argument("--specs", default="experiments/datasets/private/question_specs.json")
    parser.add_argument("--output", default="experiments/datasets/private/benchmark_v0.jsonl")
    parser.add_argument("--base-benchmark", help="Approved frozen questions to preserve and extend")
    args = parser.parse_args()
    build(args.source, args.specs, args.output, args.base_benchmark)


if __name__ == "__main__":
    main()
