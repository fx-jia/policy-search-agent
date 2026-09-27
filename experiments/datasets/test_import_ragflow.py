import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from experiments.datasets.import_ragflow import API, complete_parse, reconcile


class ImportAPITests(unittest.TestCase):
    def test_only_maps_exact_manifest_names(self):
        entries = [{"upload_name": "1__policy.md", "source_document_id": "1"}]
        self.assertEqual(reconcile(entries, [{"name": "1__policy.md", "id": "rf1"}]), {"rf1": "1"})
        with self.assertRaises(ValueError):
            reconcile(entries, [{"name": "other.md", "id": "rf2"}])
        with self.assertRaises(ValueError):
            reconcile(entries, [{"name": "1__policy.md", "id": "a"}, {"name": "1__policy.md", "id": "b"}])

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-key"})
    def test_pagination_collects_all_documents(self):
        api = API()
        pages = [{"total": 2, "docs": [{"id": "a"}]}, {"total": 2, "docs": [{"id": "b"}]}]
        with patch.object(api, "request", side_effect=pages) as request:
            self.assertEqual(len(api.documents("kb")), 2)
            self.assertIn("page=2", request.call_args.args[1])

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-key"})
    def test_pagination_rejects_duplicate_ids(self):
        api = API()
        with patch.object(api, "request", return_value={"total": 2, "docs": [{"id": "a"}, {"id": "a"}]}):
            with self.assertRaisesRegex(ValueError, "Duplicate IDs"):
                api.documents("kb")

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-key"})
    def test_business_error_does_not_leak_server_message(self):
        api = API()
        response = {"code": 102, "message": "sensitive server details"}
        with patch.object(api.opener, "open", return_value=io.BytesIO(json.dumps(response).encode())):
            with self.assertRaisesRegex(RuntimeError, "^RAGFlow application error 102; inspect local server diagnostics$"):
                api.request("POST", "/datasets", {"name": "pilot"})

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-key"})
    def test_http_error_is_redacted_and_not_automatically_retried(self):
        api = API()
        error = HTTPError("url", 401, "private", {}, io.BytesIO(b"private"))
        with patch.object(api.opener, "open", side_effect=error) as op:
            with self.assertRaisesRegex(RuntimeError, "^HTTP 401$"):
                api.request("POST", "/datasets", {})
            self.assertEqual(op.call_count, 1)

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-key"})
    @patch("experiments.datasets.import_ragflow.time.sleep")
    def test_queue_retries_failed_document_without_reparsing_success(self, sleep):
        api = API()
        entries = [{"upload_name": name, "source_document_id": name} for name in ("a", "b")]
        initial = [{"id": "a", "name": "a", "run": "DONE"}, {"id": "b", "name": "b", "run": "FAIL"}]
        final = [dict(doc, run="DONE") for doc in initial]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parse_requests.jsonl").write_text(json.dumps({"document_ids": ["b", "b"]}) + "\n")
            with patch.object(api, "documents", side_effect=[initial, final]), patch.object(api, "request") as request, patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(complete_parse(api, "kb", root, entries), final)
                self.assertEqual(request.call_args.args[2], {"document_ids": ["b"]})

    @patch.dict("os.environ", {"RAGFLOW_API_KEY": "test-key"})
    def test_queue_stops_at_retry_limit(self):
        api = API()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parse_requests.jsonl").write_text(json.dumps({"document_ids": ["b", "b", "b"]}) + "\n")
            with patch.object(api, "documents", return_value=[{"id": "b", "name": "b", "run": "FAIL"}]), patch.object(api, "request") as request, patch("sys.stdout", new=io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "three parse attempts"):
                    complete_parse(api, "kb", root, [{"upload_name": "b", "source_document_id": "b"}])
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
