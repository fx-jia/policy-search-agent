"""Fail-closed checks for the native RAGFlow outbound review boundary."""

import hashlib
import unittest

from experiments.evaluation.native_outbound import sanitize_kbinfos


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


RAW = "医疗救助基金全额资助。联系人王某，电话13800138000。"
SAFE = "医疗救助基金全额资助。"
TITLE = "徐州居民医保筹资通知"
SAFE_TITLE = "徐州居民医保通知"


def inputs():
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "docnm_kwd": TITLE,
            "content_with_weight": RAW,
            "similarity": 0.93,
            "vector_similarity": 0.81,
            "url": "",
            "document_metadata": {},
        },
        {
            "id": "c2",
            "doc_id": "d2",
            "docnm_kwd": "另一个通知",
            "content_with_weight": "标准为每年300元。",
            "similarity": 0.51,
        },
    ]
    review = {
        "c1": {
            "source_document_id": "79320",
            "raw_sha256": digest(RAW),
            "safe_sha256": digest(SAFE),
            "sanitized_content": SAFE,
            "raw_title": TITLE,
            "safe_title": SAFE_TITLE,
            "raw_title_sha256": digest(TITLE),
            "safe_title_sha256": digest(SAFE_TITLE),
        },
        "c2": {
            "source_document_id": "79256",
            "raw_sha256": digest("标准为每年300元。"),
            "safe_sha256": digest("标准为每年300元。"),
            "sanitized_content": "标准为每年300元。",
            "raw_title": "另一个通知",
            "safe_title": "另一个通知",
            "raw_title_sha256": digest("另一个通知"),
            "safe_title_sha256": digest("另一个通知"),
        },
    }
    return {"total": 2, "chunks": chunks, "doc_aggs": []}, review, {"documents": {"d1": "79320", "d2": "79256"}}


class NativeOutboundTests(unittest.TestCase):
    def test_precise_replacement_preserves_policy_fact_rank_scores_and_input(self):
        # Catches leaked raw text, reordered chunks, dropped score fields, or in-place mutation.
        original, review, mapping = inputs()
        sanitized, audit = sanitize_kbinfos(original, review, mapping)

        self.assertIsNot(sanitized, original)
        self.assertIsNot(sanitized["chunks"], original["chunks"])
        self.assertEqual([c.get("chunk_id", c.get("id")) for c in sanitized["chunks"]], ["c1", "c2"])
        self.assertEqual(sanitized["chunks"][0]["content_with_weight"], SAFE)
        self.assertEqual(sanitized["chunks"][0]["docnm_kwd"], SAFE_TITLE)
        self.assertIn("医疗救助基金全额资助", sanitized["chunks"][0]["content_with_weight"])
        self.assertNotIn("13800138000", str(sanitized))
        self.assertEqual(sanitized["chunks"][0]["similarity"], 0.93)
        self.assertEqual(sanitized["chunks"][0]["vector_similarity"], 0.81)
        self.assertEqual(sanitized["chunks"][1]["similarity"], 0.51)
        self.assertEqual(original["chunks"][0]["content_with_weight"], RAW)
        self.assertEqual(audit["chunks"][0], {
            "chunk_id": "c1",
            "source_document_id": "79320",
            "raw_sha256": digest(RAW),
            "safe_sha256": digest(SAFE),
            "changed": True,
        })
        self.assertNotIn(RAW, str(audit))

    def test_unreviewed_or_changed_chunks_fail_closed_without_raw_text(self):
        # Catches a sanitizer that trusts a stale or unrelated review record.
        for mutation in ("unknown_chunk", "changed_content", "changed_source", "changed_safe_hash"):
            with self.subTest(mutation=mutation):
                data, review, mapping = inputs()
                if mutation == "unknown_chunk":
                    data["chunks"][0]["chunk_id"] = "new"
                elif mutation == "changed_content":
                    data["chunks"][0]["content_with_weight"] += "新增个人信息"
                elif mutation == "changed_source":
                    data["chunks"][0]["doc_id"] = "d2"
                else:
                    review["c1"]["safe_sha256"] = "0" * 64
                with self.assertRaises(ValueError) as err:
                    sanitize_kbinfos(data, review, mapping)
                self.assertNotIn(RAW, str(err.exception))

    def test_title_and_secondary_prompt_surfaces_fail_closed(self):
        # Catches leakage through _kb_block fields other than content_with_weight.
        for mutation in ("changed_title", "url", "metadata", "content_alias"):
            with self.subTest(mutation=mutation):
                data, review, mapping = inputs()
                if mutation == "changed_title":
                    data["chunks"][0]["docnm_kwd"] = "患者王某"
                elif mutation == "url":
                    data["chunks"][0]["url"] = "https://example.test/patient/1"
                elif mutation == "metadata":
                    data["chunks"][0]["document_metadata"] = {"patient": "王某"}
                else:
                    data["chunks"][0]["content"] = "患者王某"
                with self.assertRaises(ValueError):
                    sanitize_kbinfos(data, review, mapping)


if __name__ == "__main__":
    unittest.main()
