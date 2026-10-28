#!/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 -IB
"""Fixed installed SSH and sudo entry points; accept only one archive digest.

Install identical reviewed bytes as operator-ssh-entry and operator-root-entry.
The SSH key forces the first pathname; sudo grants only the second pathname.
Neither entry accepts a script, environment, job path or arbitrary command.
"""

import os
from pathlib import Path
import re
import shlex
import sys

PYTHON = "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9"
BASE = "/Library/InferSSD-CI"
STAGER = BASE + "/controller/current/bench/ci/operator_mac_stage.py"
SSH_ENTRY = BASE + "/control/operator-ssh-entry"
ROOT_ENTRY = BASE + "/control/operator-root-entry"
PREFIX = [PYTHON, "-I", "-B", STAGER]
ENVIRONMENT = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C", "LC_ALL": "C"}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def digest_arguments(arguments):
    if (len(arguments) != 2 or arguments[0] != "--expected-sha256"
            or not isinstance(arguments[1], str) or HASH.fullmatch(arguments[1]) is None):
        raise ValueError("only --expected-sha256 and one lowercase SHA256 are allowed")
    return arguments[1]


def command(path, arguments, uid, euid, original):
    if path == SSH_ENTRY:
        if arguments or uid != 501 or euid != 501 or not isinstance(original, str) or len(original) > 2048:
            raise ValueError("fixed personal SSH identity and command are required")
        argv = shlex.split(original, posix=True)
        if argv[:len(PREFIX)] != PREFIX:
            raise ValueError("only the fixed reviewed stager command is allowed")
        value = digest_arguments(argv[len(PREFIX):])
        return "/usr/bin/sudo", ["/usr/bin/sudo", "-n", ROOT_ENTRY, "--expected-sha256", value]
    if path == ROOT_ENTRY:
        if uid != 0 or euid != 0:
            raise ValueError("fixed root entry requires root")
        value = digest_arguments(arguments)
        return PYTHON, [*PREFIX, "--expected-sha256", value]
    raise ValueError("entry was not invoked through an installed fixed pathname")


def main():
    executable, argv = command(str(Path(sys.argv[0])), sys.argv[1:], os.getuid(), os.geteuid(),
                               os.environ.get("SSH_ORIGINAL_COMMAND"))
    os.chdir("/")
    os.umask(0o077)
    os.execve(executable, argv, ENVIRONMENT)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print("restricted InferSSD stage entry refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
