import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.evaluation import native_baseline_report as report


def chunk(cid, title, content):
    return {"chunk_id": cid, "doc_id": cid, "docnm_kwd": title, "content_with_weight": content}


def block(index, value):
    return f"\nID: {index}\n├── Title: {value['docnm_kwd']}\n└── Content:\n{value['content_with_weight']}"


class NativeBaselineReportTests(unittest.TestCase):
    def test_prompt_mapping_distinguishes_duplicate_text_and_partial_chunk(self):
        first = chunk("a", "A", "相同正文")
        omitted = chunk("b", "B", "相同正文")
        partial = chunk("c", "C", "政策金额380元，年度2023")
        system = block(0, first) + block(2, partial)[:-6] + report.CITATION_INSTRUCTION + block(1, omitted)
        rows = report.map_prompt_chunks(system, [first, omitted, partial])
        self.assertEqual([r["status"] for r in rows], ["full", "absent", "partial"])
        self.assertEqual(rows[2]["sent_content"], partial["content_with_weight"][:-6])

    def test_native_citation_forms_and_reordered_references_are_not_false_invalid(self):
        retrieved = [chunk("a", "A", "甲"), chunk("b", "B", "乙")]
        refs = [{"id": "a"}, {"id": "b"}]
        mappings = [{"chunk_id": "a", "status": "full"}, {"chunk_id": "b", "status": "absent"}]
        rows = report.map_citations("甲[ID:0]乙[١]丙##1$$错[ID:8]", refs, retrieved, mappings)
        self.assertEqual([r["native_id"] for r in rows], [0, 1, 1, 8])
        self.assertEqual(rows[0]["chunk_id"], "a")
        self.assertEqual(rows[1]["prompt_status"], "absent")
        self.assertEqual(rows[-1]["status"], "invalid")
        altered = report.map_citations("乙[ID:1]", [{"id": "b"}], retrieved, mappings)
        self.assertEqual(altered[0]["status"], "indeterminate_reference_mapping")

    def test_empty_content_cannot_match_every_prompt(self):
        rows = report.map_prompt_chunks("ordinary system", [chunk("a", "A", "")])
        self.assertEqual(rows[0]["status"], "empty_not_rendered")

    def test_preflight_fixture_build_is_local_and_preserves_planned_denominator(self):
        fixture = report.OUT / "preflight-full.jsonl"
        rows = report.jsonl(fixture)
        row = copy.deepcopy(rows[0])
        row["body"]["data"]["answer"] = "本地夹具[ID:0]"
        config = json.loads((report.OUT / "prepared.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "generation_frozen.json").write_text(json.dumps(config))
            responses = out / "fixture.jsonl"
            responses.write_text(json.dumps(row, ensure_ascii=False) + "\n")
            with patch.object(report, "OUT", out):
                report.build([responses])
            metrics = json.loads((out / "mechanical_metrics.json").read_text())[0]
            summary = json.loads((out / "mechanical_summary.json").read_text())
            self.assertEqual(metrics["native_citation_ids"], [0])
            self.assertEqual(metrics["invalid_citation_ids"], [])
            self.assertEqual(len(metrics["prompt_full_chunk_ids"]), len(row["outbound_chunks"]))
            self.assertEqual(summary["by_split"]["all"]["planned_questions"], 150)
            self.assertEqual(len(summary["unrun_question_ids"]), 149)
            packet = json.loads((out / "review_packets" / f"{row['question_id']}.json").read_text())
            self.assertEqual(packet["actual_provider_request"], row["outbound_request"])


if __name__ == "__main__":
    unittest.main()
