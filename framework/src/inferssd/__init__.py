"""Import-safe framework components; importing this package performs no probes."""

from __future__ import annotations

from importlib import metadata

try:
    # The installed package's own version, which an engine's recorded one replaces (version.py).
    __version__ = metadata.version("inferssd")
except metadata.PackageNotFoundError:  # a source tree on sys.path that was never installed
    __version__ = "0+unknown"
