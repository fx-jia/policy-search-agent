import unittest

from experiments.datasets.split_benchmark import partition


def sample(qid, *sources, family=None):
    return {"question_id": qid, "relevant_document_ids": list(sources), "question_family_id": family}


class SplitTests(unittest.TestCase):
    def test_multisource_and_external_bridge_propagate_exposure(self):
        samples = [sample("q1", "a"), sample("q2", "b", "c"), sample("q3", "d")]
        ids, groups = partition(samples, ["a", "b", "c", "d", "external"], [("a", "external"), ("external", "b")], [], {"q1"}, 1)
        self.assertEqual(ids["q1"], ids["q2"])
        self.assertEqual(groups[ids["q2"]]["split"], "dev")
        self.assertEqual(groups[ids["q3"]]["split"], "test")

    def test_question_family_is_atomic_even_with_different_sources(self):
        samples = [sample("a", "1", family="same"), sample("b", "2", family="same"), sample("c", "3")]
        ids, groups = partition(samples, ["1", "2", "3"], [], [], {"a"}, 1)
        self.assertEqual(ids["a"], ids["b"])
        self.assertEqual(groups[ids["a"]]["split"], "dev")

    def test_unknown_source_or_exposed_question_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown source"):
            partition([sample("a", "1")], ["1"], [("1", "missing")], [], {"a"})
        with self.assertRaisesRegex(ValueError, "exposed"):
            partition([sample("a", "1")], ["1"], [], [], {"missing"})

    def test_group_partition_is_deterministic_and_disjoint(self):
        samples = [sample(str(i), str(i // 3)) for i in range(15)]
        ids, groups = partition(samples, map(str, range(5)), [], [], {"0"}, 5)
        reverse_ids, reverse_groups = partition(list(reversed(samples)), map(str, reversed(range(5))), [], [], {"0"}, 5)
        self.assertEqual((ids, groups), (reverse_ids, reverse_groups))
        dev = {d for g in groups.values() if g["split"] == "dev" for d in g["source_document_ids"]}
        test = {d for g in groups.values() if g["split"] == "test" for d in g["source_document_ids"]}
        self.assertFalse(dev & test)
        self.assertEqual(sum(len(g["question_ids"]) for g in groups.values() if g["split"] == "dev"), 6)


if __name__ == "__main__":
    unittest.main()
