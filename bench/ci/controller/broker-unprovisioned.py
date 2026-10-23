#!/usr/bin/env python3
"""Deployment placeholder, deliberately not an isolation implementation."""

import sys

sys.stderr.write(
    "No audited per-stage OS isolation backend is provisioned. Physical execution refused.\n"
)
raise SystemExit(78)
