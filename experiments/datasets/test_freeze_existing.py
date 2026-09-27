import unittest

from experiments.datasets.freeze_existing import select_existing


class ExistingCorpusTest(unittest.TestCase):
    def test_selection_keeps_gold_and_is_deterministic(self):
        candidates = {str(i): {} for i in range(30)}
        gold = {"1", "8"}
        one = select_existing(candidates, gold, 10)
        two = select_existing(dict(reversed(list(candidates.items()))), gold, 10)
        self.assertEqual(one, two)
        self.assertEqual(len(one), 10)
        self.assertTrue(gold <= set(one))

    def test_missing_gold_cannot_be_silently_removed(self):
        with self.assertRaisesRegex(ValueError, "evidence"):
            select_existing({"a": {}}, {"b"}, 1)


if __name__ == "__main__":
    unittest.main()
