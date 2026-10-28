import importlib.util
from pathlib import Path
import shlex

import pytest

spec = importlib.util.spec_from_file_location("mac_operator_entry", Path(__file__).parents[1] / "operator_entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
DIGEST = "a" * 64
COMMAND = shlex.join([*entry.PREFIX, "--expected-sha256", DIGEST])


def test_ssh_entry_only_admits_digest_and_does_not_forward_ssh_environment():
    program, argv = entry.command(entry.SSH_ENTRY, [], 501, 501, COMMAND)
    assert program == "/usr/bin/sudo"
    assert argv == [program, "-n", entry.ROOT_ENTRY, "--expected-sha256", DIGEST]
    assert set(entry.ENVIRONMENT) == {"PATH", "LANG", "LC_ALL"}


def test_root_entry_uses_frozen_interpreter_and_fixed_stager():
    program, argv = entry.command(entry.ROOT_ENTRY, ["--expected-sha256", DIGEST], 0, 0, None)
    assert program == entry.PYTHON
    assert argv == [*entry.PREFIX, "--expected-sha256", DIGEST]


@pytest.mark.parametrize("original", [
    None, "", "echo ok", COMMAND + "; /bin/sh", COMMAND + " /tmp/payload.py",
    COMMAND.replace("-I -B", "-c"), COMMAND.replace(entry.STAGER, "/tmp/payload.py"),
    COMMAND.replace(DIGEST, "A" * 64), COMMAND.replace(DIGEST, "a" * 63),
    COMMAND + "\n/bin/sh", COMMAND + " --ensure", "x" * 2049,
])
def test_ssh_entry_rejects_shell_commands_and_nonexact_arguments(original):
    with pytest.raises(ValueError):
        entry.command(entry.SSH_ENTRY, [], 501, 501, original)


@pytest.mark.parametrize("arguments", [[], [DIGEST], ["--expected-sha256", DIGEST, "extra"],
                                        ["--expected-sha256", DIGEST + "\n"],
                                        ["--expected-sha256", "$(id)"]])
def test_root_entry_revalidates_digest_instead_of_trusting_ssh_wrapper(arguments):
    with pytest.raises(ValueError):
        entry.command(entry.ROOT_ENTRY, arguments, 0, 0, None)


@pytest.mark.parametrize("path,arguments,uid,euid,original", [
    (entry.SSH_ENTRY, [], 62001, 62001, COMMAND),
    (entry.SSH_ENTRY, [], 0, 0, COMMAND),
    (entry.SSH_ENTRY, ["extra"], 501, 501, COMMAND),
    (entry.ROOT_ENTRY, ["--expected-sha256", DIGEST], 501, 0, COMMAND),
    ("/tmp/operator-root-entry", ["--expected-sha256", DIGEST], 0, 0, None),
])
def test_entry_rejects_other_identities_and_paths(path, arguments, uid, euid, original):
    with pytest.raises(ValueError):
        entry.command(path, arguments, uid, euid, original)
