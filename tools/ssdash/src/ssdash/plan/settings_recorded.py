"""The original settings temp-write/replace output adapter."""

from __future__ import annotations

import os


def write_settings(path: str, text: str) -> None:
    # Preserve a failed write/close/replace's temporary file. Cleanup or retries
    # would change the command's behavior and require a separate decision.
    with open(f"{path}.tmp", "w") as stream:
        stream.write(text)
    os.replace(f"{path}.tmp", path)
