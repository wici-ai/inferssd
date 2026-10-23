"""Legacy recorded-arm adapters; no live tail cache or process ownership."""

from __future__ import annotations

from typing import Optional, cast

from inferssd.plan.settings_types import Ledger
from inferssd.telemetry.files import LOCAL_FILES, lines, readable
from inferssd.telemetry.legacy import ArmParser, _kv
from inferssd.telemetry.types import ArmFiles, ArmResult


def read_arm(
    run_dir: str,
    skip: int = 128,
    parser: Optional[ArmParser] = None,
    *,
    files: ArmFiles = LOCAL_FILES,
) -> ArmResult:
    if parser is None:
        parser = ArmParser()
        log = (
            f"{run_dir}/server.log.gz"
            if readable(f"{run_dir}/server.log.gz", files)
            else f"{run_dir}/server.log"
        )
        for line in lines(log, files):
            parser.feed(line)
    return parser.result(run_dir, skip, files=files)


def ledger(run_dir: str, *, files: ArmFiles = LOCAL_FILES) -> Optional[Ledger]:
    log = (
        f"{run_dir}/server.log.gz"
        if readable(f"{run_dir}/server.log.gz", files)
        else f"{run_dir}/server.log"
    )
    rec = None
    if readable(log, files):
        for line in lines(log, files):
            # Unlike ArmParser.feed, this intentionally requires the exact line
            # start; a timestamp/prefix changes the old standalone-ledger result.
            if line.startswith("[kimi_host_ledger] reason=destroy"):
                rec = {k: _kv(line, k) for k in ("unified", "margin", "margin_updates")}
    return cast(Optional[Ledger], rec)
