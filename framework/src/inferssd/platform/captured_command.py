# SPDX-License-Identifier: Apache-2.0
"""The one way the framework runs a command and reads its output."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Optional


def run_captured(
    argv: Sequence[str], *, timeout: float, env: Optional[Mapping[str, str]] = None
) -> subprocess.CompletedProcess[str]:
    """Run ARGV in ENV (default: this process's) and capture its output for parsing.

    The output is decoded as UTF-8 with bad bytes replaced, and the command runs in
    the C locale. Tools format text for the user's locale: with German settings,
    ``sysctl vm.swapusage`` reports ``2048,00M``. Parsers read one format, whatever
    the user's settings, so the locale is fixed here rather than in each parser.
    """
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**(os.environ if env is None else env), "LC_ALL": "C"},
        timeout=timeout,
        check=False,
    )
