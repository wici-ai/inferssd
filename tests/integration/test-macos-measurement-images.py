import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("images", Path(__file__).resolve().parents[2] / "bench/macos/measurement_images.py")
images = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(images)


class ImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "bundle with spaces"
        self.bundle.mkdir()
        self.main = self.bundle / "llama-server"
        self.main.write_bytes(b"main")
        self.core = [self.main]
        for family in sorted(images.CORE_FAMILIES):
            path = self.bundle / (family + ".0.1.2.dylib")
            path.write_bytes(family.encode())
            self.core.append(path)
        self.backends = []
        for family in sorted(images.BACKEND_FAMILIES):
            path = self.bundle / (family + ".0.dylib")
            path.write_bytes(family.encode())
            self.backends.append(path)
        self.prefix = self.root / "snapshot"

    def output(self, paths=None):
        result = b"p42\0\n"
        for path in paths or self.core:
            st = path.stat()
            result += b"ftxt\0D" + hex(st.st_dev).encode() + b"\0i" + str(st.st_ino).encode()
            result += b"\0n" + os.fsencode(path) + b"\0\n"
        return result

    def run_snapshot(self, data=None, **kwargs):
        result = subprocess.CompletedProcess([], 0, data if data is not None else self.output(), b"warning retained")
        with patch.object(images.subprocess, "run", return_value=result):
            return images.snapshot_images(42, self.main, self.prefix, **kwargs)

    def assert_failure(self, text, data=None, **kwargs):
        with self.assertRaisesRegex(images.ImageIdentityError, text):
            self.run_snapshot(data, **kwargs)
        self.assertFalse(json.loads(Path(str(self.prefix) + ".images.json").read_text())["success"])

    def test_core_snapshot_retains_raw_and_hashes(self):
        result = self.run_snapshot()
        self.assertTrue(result["success"])
        self.assertEqual(len(result["images"]), 4)
        self.assertEqual(Path(str(self.prefix) + ".lsof.stdout").read_bytes(), self.output())
        self.assertEqual(Path(str(self.prefix) + ".lsof.stderr").read_bytes(), b"warning retained")
        self.assertTrue(all(len(item["sha256"]) == 64 for item in result["images"]))

    def test_late_backends_allowed_and_recorded(self):
        before = self.run_snapshot()
        after = self.run_snapshot(self.output(self.core + self.backends), require_backends=True)
        delta = images.verify_image_continuity(before, after)
        self.assertEqual(len(delta["added"]), 2)

    def test_end_requires_backends(self):
        self.assert_failure("required project images", require_backends=True)

    def test_missing_core(self):
        self.assert_failure("required project images", self.output(self.core[:-1]))

    def test_missing_main(self):
        self.assert_failure("main executable", self.output(self.core[1:]))

    def test_wrong_pid(self):
        self.assert_failure("unexpected process", self.output().replace(b"p42", b"p43"))

    def test_empty_lsof(self):
        self.assert_failure("no process images", b"")

    def test_missing_inode(self):
        data = self.output()
        key = b"i" + str(self.main.stat().st_ino).encode() + b"\0"
        self.assert_failure("device/inode", data.replace(key, b"", 1))

    def test_wrong_device(self):
        key = b"D" + hex(self.main.stat().st_dev).encode() + b"\0"
        self.assert_failure("device/inode differs", self.output().replace(key, b"D0xdeadbeef\0", 1))

    def test_loaded_inode_replaced(self):
        data = self.output()
        replacement = self.bundle / "new-main"
        replacement.write_bytes(b"new-main")
        replacement.replace(self.main)
        self.assert_failure("device/inode differs", data)

    def test_foreign_optional_project_library(self):
        foreign = self.root / "libmtmd.0.dylib"
        foreign.write_bytes(b"foreign")
        self.assert_failure("outside selected build", self.output(self.core + [foreign]))

    def test_symlink_escape(self):
        foreign = self.root / "libmtmd.0.dylib"
        foreign.write_bytes(b"foreign")
        link = self.bundle / foreign.name
        link.symlink_to(foreign)
        self.assert_failure("outside selected build", self.output(self.core + [link]))

    def test_deleted_project_path_not_ignored(self):
        self.assert_failure("FileNotFoundError", self.output() + b"ftxt\0D0x1\0i1\0n/tmp/libkimi.dylib (deleted)\0\n")

    def test_duplicate_project_family(self):
        duplicate = self.bundle / "libllama.99.dylib"
        duplicate.write_bytes(b"second-llama")
        self.assert_failure("multiple loaded images", self.output(self.core + [duplicate]))

    def test_contents_changed_during_hash(self):
        original_read = os.read
        changed = False

        def mutate(fd, count):
            nonlocal changed
            value = original_read(fd, count)
            if not changed:
                self.main.write_bytes(b"changed-while-hashing")
                changed = True
            return value

        with patch.object(images.os, "read", side_effect=mutate):
            self.assert_failure("contents changed while hashing")

    def test_timeout_retains_partial_raw(self):
        exc = subprocess.TimeoutExpired(["lsof"], 15, output=b"partial", stderr=b"error")
        with patch.object(images.subprocess, "run", side_effect=exc):
            with self.assertRaisesRegex(images.ImageIdentityError, "timed out"):
                images.snapshot_images(42, self.main, self.prefix)
        receipt = json.loads(Path(str(self.prefix) + ".images.json").read_text())
        self.assertTrue(receipt["timeout"])
        self.assertFalse(receipt["success"])
        self.assertEqual(Path(str(self.prefix) + ".lsof.stdout").read_bytes(), b"partial")

    def test_nonzero_rc_retains_failure(self):
        result = subprocess.CompletedProcess([], 1, b"", b"permission denied")
        with patch.object(images.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(images.ImageIdentityError, "snapshot failed"):
                images.snapshot_images(42, self.main, self.prefix)
        self.assertEqual(json.loads(Path(str(self.prefix) + ".images.json").read_text())["returncode"], 1)

    def test_missing_lsof_retains_failure(self):
        with patch.object(images.subprocess, "run", side_effect=FileNotFoundError("missing lsof")):
            with self.assertRaisesRegex(images.ImageIdentityError, "FileNotFoundError"):
                images.snapshot_images(42, self.main, self.prefix)
        self.assertFalse(json.loads(Path(str(self.prefix) + ".images.json").read_text())["success"])

    def test_continuity_rejects_changed_contents(self):
        before = self.run_snapshot()
        self.core[1].write_bytes(b"new-content")
        after = self.run_snapshot()
        with self.assertRaisesRegex(images.ImageIdentityError, "changed or disappeared"):
            images.verify_image_continuity(before, after)

    def test_continuity_rejects_unloaded_image(self):
        before = self.run_snapshot(self.output(self.core + self.backends))
        after = self.run_snapshot()
        with self.assertRaisesRegex(images.ImageIdentityError, "changed or disappeared"):
            images.verify_image_continuity(before, after)

    def test_bad_pid_receipt(self):
        with self.assertRaisesRegex(images.ImageIdentityError, "positive integer"):
            images.snapshot_images(-1, self.main, self.prefix)
        self.assertFalse(json.loads(Path(str(self.prefix) + ".images.json").read_text())["success"])


if __name__ == "__main__":
    unittest.main()
