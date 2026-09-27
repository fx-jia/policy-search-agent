"""Check the actual text passed across the export worker/tokenizer boundary."""

import ast
import unittest

from experiments.evaluation.export_context_snapshot import SCRIPT


class ContextSnapshotTests(unittest.TestCase):
    def test_worker_passage_uses_real_newlines_for_tokenization(self):
        # Execute the worker's actual expression without its service imports.
        assignments = [node for node in ast.walk(ast.parse(SCRIPT)) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "block" for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        expression = ast.Expression(assignments[0].value)
        block = eval(compile(expression, "export-worker", "eval"), {"sid": "a", "cid": "c", "content": "政策\n正文"})
        self.assertEqual(block, "[source=a; chunk=c]\n政策\n正文\n\n")


if __name__ == "__main__":
    unittest.main()
