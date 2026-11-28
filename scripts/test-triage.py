#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the issue triage, and keep it in step with the issue forms."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("triage", Path(__file__).with_name("triage.py"))
triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triage)
sys.path.insert(0, str(ROOT / "framework" / "src"))
from inferssd.app.catalog import MODELS  # noqa: E402

TYPES = {"bug", "app", "model-request", "enhancement", "perf", "question"}


def scalar(text):
    return json.loads(text) if text.startswith('"') else text


def forms():
    """Each issue form's labels and questions; reads the subset of YAML the forms use."""
    found = {}
    for path in sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml")):
        if path.name == "config.yml":
            continue
        labels, items, item, in_options = [], [], None, False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if line.startswith("labels:"):
                labels = json.loads(line.split(":", 1)[1])
            elif indent == 2 and stripped.startswith("- type:"):
                item = {"type": stripped.split(":", 1)[1].strip(), "options": []}
                items.append(item)
                in_options = False
            elif item is not None and indent == 6 and stripped.startswith("label:"):
                item["label"] = scalar(stripped.split(":", 1)[1].strip())
            elif item is not None and indent == 6 and stripped == "options:":
                in_options = True
            elif in_options and indent == 8 and stripped.startswith("- "):
                if item["type"] == "dropdown":
                    item["options"].append(scalar(stripped[2:]))
            elif in_options and indent <= 6:
                in_options = False
        found[path.name] = (labels, items)
    return found


def body(**answers):
    return "\r\n\r\n".join(f"### {heading}\r\n\r\n{answer}" for heading, answer in answers.items())


def issue_event(action, text, labels=(), assignees=(), old=None):
    event = {"action": action, "issue": {"number": 7, "body": text, "labels": [{"name": n} for n in labels],
                                         "assignees": list(assignees)}}
    if old is not None:
        event["changes"] = {"body": {"from": old}}
    return event


class FormTests(unittest.TestCase):
    def test_every_form_sets_one_type_label(self):
        for name, (labels, _) in forms().items():
            with self.subTest(form=name):
                self.assertEqual(len(labels), 1)
                self.assertIn(labels[0], TYPES)

    def test_labelled_questions_offer_only_mapped_options(self):
        seen = {heading: set() for heading in triage.FIELDS}
        for name, (_, items) in forms().items():
            for item in items:
                if item["type"] == "dropdown" and item["label"] in triage.FIELDS:
                    with self.subTest(form=name, question=item["label"]):
                        self.assertLessEqual(set(item["options"]), set(triage.FIELDS[item["label"]]))
                    seen[item["label"]].update(item["options"])
        for heading, options in triage.FIELDS.items():
            with self.subTest(question=heading):
                self.assertEqual(seen[heading], set(options), "a mapped option no form offers")

    def test_model_question_lists_the_catalog(self):
        catalog = {f"{model.repo} ({model.quantization})" for model in MODELS}
        for name, (_, items) in forms().items():
            for item in items:
                if item["type"] == "dropdown" and item["label"] == "Model":
                    with self.subTest(form=name):
                        listed = [option for option in item["options"] if not option.startswith("Other")]
                        self.assertEqual(set(listed), catalog)
                        self.assertTrue(item["options"][-1].startswith("Other"))

    def test_owner_rules_name_known_labels(self):
        known = TYPES | {label for options in triage.FIELDS.values() for label in options.values() if label}
        for label, _ in triage.ISSUE_OWNERS:
            self.assertIn(label, known)


class IssueTests(unittest.TestCase):
    def test_answers_become_labels(self):
        text = body(**{"Operating system": "Windows", "Backend": "CUDA, CPU only", "Model": "Other",
                       "Model storage": "Several drives or a RAID volume", "Hardware": "_No response_"})
        self.assertEqual(triage.issue_labels(text), {"windows", "cuda", "cpu", "multi-drive"})
        self.assertEqual(triage.issue_labels(body(Backend="Not sure", **{"Model storage": "One drive"})), set())
        self.assertEqual(triage.issue_labels(body(Kind="Request")), {"enhancement"})
        self.assertEqual(triage.issue_labels("free text without a form"), set())
        self.assertEqual(triage.issue_labels(None), set())

    def test_opened_issue_is_labelled_and_assigned_by_rule_order(self):
        text = body(**{"Operating system": "Linux", "Model storage": "Several drives or a RAID volume"})
        self.assertEqual(triage.plan("issues", issue_event("opened", text, labels=["perf"])), [
            ("POST", "/issues/7/labels", {"labels": ["linux", "multi-drive"]}),
            ("POST", "/issues/7/assignees", {"assignees": ["ustc-sunny"]})])
        self.assertEqual(triage.plan("issues", issue_event("opened", body(Kind="Problem"), labels=["app"]))[-1],
                         ("POST", "/issues/7/assignees", {"assignees": ["DearJohnn"]}))
        self.assertEqual(triage.plan("issues", issue_event("opened", "", labels=["question"])),
                         [("POST", "/issues/7/assignees", {"assignees": ["Chikati"]})])
        self.assertEqual(triage.plan("issues", issue_event("opened", "", labels=["bug"], assignees=[{"login": "x"}])),
                         [])

    def test_edited_answers_replace_only_their_own_labels(self):
        old = body(**{"Operating system": "Linux", "Backend": "CUDA"})
        new = body(**{"Operating system": "Windows", "Backend": "CUDA"})
        event = issue_event("edited", new, labels=["bug", "linux", "cuda", "upstream"], old=old)
        self.assertEqual(triage.plan("issues", event), [
            ("DELETE", "/issues/7/labels/linux", None),
            ("POST", "/issues/7/labels", {"labels": ["windows"]})])
        # A label the old answers named but a maintainer already removed stays removed.
        event = issue_event("edited", new, labels=["bug", "cuda", "windows"], old=old)
        self.assertEqual(triage.plan("issues", event), [])
        self.assertEqual(triage.plan("issues", {"action": "edited", "issue": event["issue"], "changes": {}}), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
