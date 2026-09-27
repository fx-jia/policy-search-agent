"""Aggregate explicit assistant annotations against the frozen 150-question rubric.

This module only checks and counts supplied annotations. It does not judge answers.
"""

import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUBRIC = REPO / "experiments/results/private/native-corpus1000-v2-preflight/combined_answer_rubric_v2.json"
DEFAULT_OUTPUT = REPO / "experiments/results/private/native-corpus1000-v2"
FAILURE_CATEGORIES = {
    "none",
    "retrieval_missing",
    "evidence_not_in_context",
    "generation_omission",
    "wrong_scope_or_version",
    "unsupported_generation",
    "benchmark_ambiguity",
    "api_failure",
}
SENSITIVITY_IDS = ("med-018", "med-026")


def _rubric_questions(rubric):
    if not isinstance(rubric, dict) or not isinstance(rubric.get("questions"), list):
        raise ValueError("Rubric must contain questions")
    questions = rubric["questions"]
    ids = [q.get("question_id") for q in questions]
    if len(questions) != 150 or len(set(ids)) != 150 or any(not isinstance(qid, str) for qid in ids):
        raise ValueError("Expected 150 distinct planned questions")
    counts = Counter(q.get("split") for q in questions)
    points = Counter()
    for q in questions:
        required = q.get("required_points")
        if not isinstance(required, list) or not required or any(not isinstance(p, str) or not p.strip() for p in required):
            raise ValueError("Every rubric question needs nonempty required points")
        points[q["split"]] += len(required)
    if counts != {"dev": 51, "test": 99} or points != {"dev": 92, "test": 227}:
        raise ValueError("Frozen split or 319-point plan changed")
    if rubric.get("question_count") != 150 or rubric.get("point_count") != 319:
        raise ValueError("Rubric metadata does not match the frozen plan")
    if set(rubric.get("sensitivity_question_ids", [])) != set(SENSITIVITY_IDS):
        raise ValueError("Frozen sensitivity questions changed")
    return questions


def _nonempty_string_list(value, field):
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must be a list of nonempty strings")


def _validate_annotation(row, question):
    if not isinstance(row, dict):
        raise ValueError("Annotation row must be an object")
    count = len(question["required_points"])
    for field in ("points_pass", "point_prompt_support"):
        values = row.get(field)
        if not isinstance(values, list) or len(values) != count or any(type(value) is not bool for value in values):
            raise ValueError(f"{question['question_id']}: {field} must have {count} booleans")
    reasons = row.get("point_reasons")
    if not isinstance(reasons, list) or len(reasons) != count:
        raise ValueError(f"{question['question_id']}: point_reasons length mismatch")
    _nonempty_string_list(reasons, "point_reasons")
    for field in ("incorrect_extra_claims", "unsupported_extra_claims", "citation_issues"):
        _nonempty_string_list(row.get(field), field)
    if row.get("failure_category") not in FAILURE_CATEGORIES:
        raise ValueError(f"{question['question_id']}: unknown failure_category")
    if not isinstance(row.get("notes"), str):
        raise ValueError(f"{question['question_id']}: notes must be a string")
    for field in ("applicability_uncertain", "mixed_refusal"):
        if field in row and type(row[field]) is not bool:
            raise ValueError(f"{question['question_id']}: {field} must be a boolean")


def _score(question, annotation):
    result = {
        "question_id": question["question_id"],
        "split": question["split"],
        "planned_points": len(question["required_points"]),
        "assessed": annotation is not None,
        "passed_points": None,
        "rubric_content_complete": None,
        "rubric_content_complete_and_grounded": None,
        "prompt_fully_supports_required_points": None,
        "semantic_citation_issue": None,
        "failure_category": None,
        "incorrect_extra_claim_count": None,
        "unsupported_extra_claim_count": None,
        "applicability_uncertain": None,
        "mixed_refusal": None,
    }
    if annotation is None:
        return result
    complete = all(annotation["points_pass"]) and not annotation["incorrect_extra_claims"]
    prompt_supported = all(annotation["point_prompt_support"])
    result.update(
        passed_points=sum(annotation["points_pass"]),
        rubric_content_complete=complete,
        rubric_content_complete_and_grounded=complete and prompt_supported and not annotation["unsupported_extra_claims"],
        prompt_fully_supports_required_points=prompt_supported,
        semantic_citation_issue=bool(annotation["citation_issues"]),
        failure_category=annotation["failure_category"],
        incorrect_extra_claim_count=len(annotation["incorrect_extra_claims"]),
        unsupported_extra_claim_count=len(annotation["unsupported_extra_claims"]),
        applicability_uncertain=annotation.get("applicability_uncertain", False),
        mixed_refusal=annotation.get("mixed_refusal", False),
    )
    return result


def _summarize(scored):
    assessed = [row for row in scored if row["assessed"]]
    planned_questions = len(scored)
    planned_points = sum(row["planned_points"] for row in scored)
    assessed_points = sum(row["planned_points"] for row in assessed)
    passed = sum(row["passed_points"] for row in assessed)
    complete = sum(row["rubric_content_complete"] for row in assessed)
    grounded = sum(row["rubric_content_complete_and_grounded"] for row in assessed)
    return {
        "planned_questions": planned_questions,
        "planned_points": planned_points,
        "assessed_questions": len(assessed),
        "assessed_points": assessed_points,
        "unassessed_questions": planned_questions - len(assessed),
        "partial_run": len(assessed) != planned_questions,
        "passed_points": passed,
        "point_pass_fraction_planned": passed / planned_points,
        "point_pass_fraction_assessed": passed / assessed_points if assessed_points else None,
        "rubric_content_complete_count": complete,
        "rubric_content_complete_fraction_planned": complete / planned_questions,
        "rubric_content_complete_fraction_assessed": complete / len(assessed) if assessed else None,
        "rubric_content_complete_and_grounded_count": grounded,
        "rubric_content_complete_and_grounded_fraction_planned": grounded / planned_questions,
        "rubric_content_complete_and_grounded_fraction_assessed": grounded / len(assessed) if assessed else None,
        "prompt_fully_supports_required_points_count": sum(row["prompt_fully_supports_required_points"] for row in assessed),
        "unsupported_extra_claim_question_count": sum(row["unsupported_extra_claim_count"] > 0 for row in assessed),
        "unsupported_extra_claim_count": sum(row["unsupported_extra_claim_count"] for row in assessed),
        "applicability_uncertain_count": sum(row["applicability_uncertain"] for row in assessed),
        "mixed_refusal_count": sum(row["mixed_refusal"] for row in assessed),
        "semantic_citation_issue_questions": sum(row["semantic_citation_issue"] for row in assessed),
        "failure_category_counts": dict(sorted(Counter(row["failure_category"] for row in assessed).items())),
    }


def aggregate(rubric, annotations, *, require_complete=False):
    """Validate annotations and return planned-denominator summaries and all question scores."""
    questions = _rubric_questions(rubric)
    by_id = {q["question_id"]: q for q in questions}
    reviewed = {}
    for row in annotations:
        if not isinstance(row, dict):
            raise ValueError("Annotation row must be an object")
        qid = row.get("question_id")
        if qid not in by_id:
            raise ValueError(f"Unknown question_id: {qid}")
        if qid in reviewed:
            raise ValueError(f"Duplicate question_id: {qid}")
        _validate_annotation(row, by_id[qid])
        reviewed[qid] = row
    if require_complete and len(reviewed) != len(by_id):
        missing = sorted(set(by_id) - set(reviewed))
        raise ValueError(f"Missing {len(missing)} required annotations: {missing[:5]}")
    scored = [_score(q, reviewed.get(q["question_id"])) for q in questions]
    summary = {
        "dev51": _summarize([row for row in scored if row["split"] == "dev"]),
        "test99": _summarize([row for row in scored if row["split"] == "test"]),
        "all150": _summarize(scored),
        "sensitivity_questions": {qid: next(row for row in scored if row["question_id"] == qid) for qid in SENSITIVITY_IDS},
        "sensitivity_kept_in_main_scores": True,
        "assessment_scope": "assistant semantic annotations; not independent human review",
    }
    return summary, scored


def load_annotations(patterns):
    """Read only annotation files named by caller-supplied paths or globs."""
    if not patterns:
        raise ValueError("At least one annotation path or glob is required")
    rows, paths, seen = [], [], set()
    for pattern in patterns:
        matches = [Path(path).resolve() for path in sorted(glob.glob(pattern))]
        if not matches:
            raise ValueError(f"Annotation pattern matched no files: {pattern}")
        for path in matches:
            if path in seen or not path.is_file():
                raise ValueError(f"Duplicate or non-file annotation path: {path}")
            seen.add(path)
            paths.append(path)
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from error
    return rows, paths


def report(rubric_path, annotation_patterns, output_dir, *, require_complete=False):
    rubric_path = Path(rubric_path)
    rubric = json.loads(rubric_path.read_text())
    rows, paths = load_annotations(annotation_patterns)
    summary, scored = aggregate(rubric, rows, require_complete=require_complete)
    summary["provenance"] = {
        "rubric_path": str(rubric_path),
        "rubric_sha256": hashlib.sha256(rubric_path.read_bytes()).hexdigest(),
        "annotation_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "require_complete": require_complete,
        "provider_calls": 0,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "semantic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "per_question_scores.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n")
    return summary, scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--annotations", "--annotation", nargs="+", action="append", required=True, help="Explicit JSONL paths or glob patterns")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    patterns = [pattern for group in args.annotations for pattern in group]
    summary, _ = report(args.rubric, patterns, args.output_dir, require_complete=args.require_complete)
    print(json.dumps({"assessed_questions": summary["all150"]["assessed_questions"], "planned_questions": 150, "partial_run": summary["all150"]["partial_run"]}))


if __name__ == "__main__":
    main()
