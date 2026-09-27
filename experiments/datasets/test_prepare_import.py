import unittest

from experiments.datasets.prepare_import import exclusion_reasons, export_document, select_documents


class ImportSelectionTests(unittest.TestCase):
    def row(self, **overrides):
        return {"文件名称": "测试政策", "政策正文": "政策测试正文。" * 30, "是否公开发布": "是", "适用区域": "测试地区", "政策文件大类": "测试类别", **overrides}

    def test_excludes_flagged_and_unconfirmed_public_records(self):
        self.assertIn("public_status_not_yes", exclusion_reasons(self.row(是否公开发布=None)))
        self.assertIn("personal_record_context", exclusion_reasons(self.row(政策正文="患者姓名：测试" * 30)))
        self.assertIn("body_under_100_chars", exclusion_reasons(self.row(政策正文="仅有占位符")))

    def test_does_not_silently_drop_flagged_benchmark_sources(self):
        rows = {"1": self.row(), "2": self.row(是否公开发布=None)}
        with self.assertRaisesRegex(ValueError, "benchmark source"):
            select_documents(rows, {"2"}, 1)

    def test_sampling_is_stable_and_includes_gold(self):
        rows = {str(i): self.row(适用区域=f"地区{i % 3}") for i in range(15)}
        chosen, _ = select_documents(rows, {"0", "14"}, 8)
        self.assertEqual(len(chosen), 8)
        self.assertTrue({"0", "14"} <= set(chosen))
        self.assertEqual(chosen, select_documents(dict(reversed(list(rows.items()))), {"14", "0"}, 8)[0])

    def test_all_eligible_does_not_select_using_gold_labels(self):
        rows = {str(i): self.row() for i in range(8)}
        rows["8"] = self.row(是否公开发布=None)
        self.assertEqual(select_documents(rows, {"0"}, None)[0], list(map(str, range(8))))
        self.assertEqual(select_documents(rows, {"7"}, None)[0], list(map(str, range(8))))

    def test_export_preserves_body_and_omits_auxiliary_fields(self):
        row = self.row(附件列表=[{"内容": "不应导出附件内容"}], 待遇精选数据="不应导出精选数据")
        exported = export_document(row)
        self.assertEqual(exported, "# 测试政策\n\n" + row["政策正文"] + "\n")
        self.assertNotIn("不应导出", exported)

    def test_invalid_sizes_fail_instead_of_looping(self):
        for size in (0, 3):
            with self.assertRaises(ValueError):
                select_documents({"1": self.row(), "2": self.row()}, {"1"}, size)


if __name__ == "__main__":
    unittest.main()
