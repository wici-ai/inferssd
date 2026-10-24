#!/usr/bin/python3
"""Trusted, read-only start barrier; exec preserves the externally pinned PID."""

import os
import json
from pathlib import Path
import sys
import time


def main():
    token = sys.argv[1]
    release = Path('/inputs/launch/release')
    deadline = time.monotonic() + 15
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('root supervisor did not release stage')
        time.sleep(.01)
    if release.read_text() != token:
        raise RuntimeError('stage release identity differs')
    os.environ.update(json.loads(Path('/inputs/launch/environment.json').read_text()))
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == '__main__':
    main()
