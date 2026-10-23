#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exercise JSON decoding without weakening public-content checks."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("public_scan", Path(__file__).with_name("public-scan.py"))
scanner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner)


class JsonLineTests(unittest.TestCase):
    def check_findings(self, text, expected, filename="fixture.json"):
        with contextlib.redirect_stdout(io.StringIO()):
            result = scanner.inspect(scanner.added_line_text(filename, text), "fixture:1")
        self.assertEqual(result, expected)

    def test_escaped_whitespace_before_path_marker(self):
        for separator in ("\t", "\n", "\r"):
            with self.subTest(separator=repr(separator)):
                self.check_findings(json.dumps(separator + "@case/input"), 0)

    def test_literal_login_is_still_blocked(self):
        self.check_findings(json.dumps("t" + "@case/input"), 1)

    def test_login_after_escaped_whitespace_is_still_blocked(self):
        self.check_findings(json.dumps("\t" + "user" + "@host"), 1)

    def test_unicode_escaped_login_is_blocked(self):
        self.check_findings('"\\u0075ser\\u0040host"', 1)

    def test_unicode_escaped_private_path_is_blocked(self):
        self.check_findings('"\\u002fhome\\u002ffixture\\u002finput"', 1)

    def test_double_escaped_tab_is_not_whitespace(self):
        self.check_findings(json.dumps("\\t" + "@case/input"), 1)

    def test_invalid_escape_is_preserved(self):
        text = '"\\q user' + '@host"'
        self.assertEqual(scanner.added_line_text("fixture.json", text), text)
        self.check_findings(text, 1)

    def test_unterminated_literal_is_preserved(self):
        text = '"user' + '@host'
        self.assertEqual(scanner.added_line_text("fixture.json", text), text)
        self.check_findings(text, 1)

    def test_trailing_escape_is_preserved(self):
        text = '"user' + '@host\\'
        self.assertEqual(scanner.added_line_text("fixture.json", text), text)
        self.check_findings(text, 1)

    def test_non_json_is_not_decoded(self):
        text = json.dumps("\t" + "@case/input")
        self.assertEqual(scanner.added_line_text("fixture.txt", text), text)
        self.check_findings(text, 1, filename="fixture.txt")

    def test_non_string_fragments_are_scanned(self):
        self.check_findings('"safe": user' + '@host', 1)

    def test_multiple_literals_and_fragments_keep_order(self):
        text = '{"key": "tab\\tquote\\\"slash\\\\", "next": 3}'
        self.assertEqual(scanner.added_line_text("fixture.json", text),
                         '{"key": "tab\tquote"slash\\", "next": 3}')

    def test_invalid_literal_does_not_hide_later_literal(self):
        text = '["\\q", "user' + '@host"]'
        self.check_findings(text, 1)

    def test_empty_and_plain_fragments_are_unchanged(self):
        for text in ("", '""', "42,", "  ]", '"a": 1,'):
            with self.subTest(text=text):
                self.assertEqual(scanner.added_line_text("fixture.json", text), text)


if __name__ == "__main__":
    unittest.main()
