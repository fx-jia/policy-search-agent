"""Run the self-contained backup checks, without provider or database calls."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
modules = ["experiments.search_agent." + p.stem for p in sorted((ROOT / "experiments/search_agent").glob("test_*.py"))]
modules += ["experiments.evaluation." + n for n in (
    "test_search_agent_eval", "test_search_agent_report", "test_answer_quality",
    "test_answer_quality_report", "test_fast_rag_eval", "test_fast_rag_report",
    "test_fast_context_eval", "test_evidence_router_eval",
)]
EXCLUDED = "experiments.evaluation.test_fast_context_eval.FrozenRunnerTests.test_only_selection_changes_and_original_runner_stays_untouched"


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


if __name__ == "__main__":
    tests = list(flatten(unittest.defaultTestLoader.loadTestsFromNames(modules)))
    print("Excluded local-artifact audit (requires omitted runtime config): " + EXCLUDED)
    suite = unittest.TestSuite(t for t in tests if t.id() != EXCLUDED)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
