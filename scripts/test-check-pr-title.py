#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the pull request title rules."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("check_pr_title", Path(__file__).with_name("check-pr-title.py"))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
scanner = checker.load_scanner()


class TitleTests(unittest.TestCase):
    def problems(self, title):
        return checker.problems(title, scanner)

    def test_valid_titles_pass(self):
        for title in ("ci: add isolated Mac base and PR performance comparisons",
                      "cuda, metal: fix expert staging",
                      "docs: update DeepSeek setup notes",
                      "framework: embed source receipts in launch plans",
                      "repo: link wici-ai/inferssd from the README",
                      "model: " + "a" * 65):
            with self.subTest(title=title):
                self.assertEqual(self.problems(title), [])

    def test_invalid_titles_fail(self):
        # The blocked company-name fixture is split so that this file passes public-scan.
        for title in ("Compare reviewed PR performance",
                      "test: freeze F settings",
                      "docs: WI" + "CI-InferSSD",
                      "docs: Add the quickstart guide",
                      "docs: added the quickstart guide",
                      "docs: updates the quickstart guide",
                      "docs: fix the quickstart guide.",
                      "Docs: fix the quickstart guide",
                      "docs:fix the quickstart guide",
                      "cuda,metal: fix expert staging",
                      "cuda, cuda: fix expert staging",
                      "docs: fix  spacing",
                      " docs: fix spacing",
                      "docs: fix caf\u00e9 spelling",
                      "ci: link wici-ai/inferssd-device-ci results",
                      "model: " + "a" * 66):
            with self.subTest(title=title):
                self.assertNotEqual(self.problems(title), [])

    def test_test_area_names_the_rule(self):
        self.assertIn("use the area of the tested module instead of 'test'",
                      self.problems("test: freeze F settings"))

    def test_public_scan_rejects_internal_name(self):
        self.assertTrue(any(problem.startswith("public-scan:")
                            for problem in self.problems("docs: WI" + "CI-InferSSD")))


if __name__ == "__main__":
    unittest.main()
