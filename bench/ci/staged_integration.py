#!/usr/bin/env python3
"""Build one reviewed source and complete one short model request in a CI job."""

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import digest, load, revision, save
from ci.stage_session import StageSession
from ci.staged_build import run_build
from ci.staged_smoke import collect_log, smoke


def run(args):
    config = load(args.config)
    source_revision = revision(config["revision"])
    if digest(args.config) != args.config_sha256:
        raise ValueError("integration configuration differs from reviewed identity")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "integration-config.json").write_bytes(args.config.read_bytes())
    report = dict(schema_version=1, purpose="integration-smoke", status="failed",
                  revision=source_revision, model=config["smoke"]["model"],
                  config_sha256=args.config_sha256, benchmark_samples=0,
                  performance_acceptance=False)
    session = StageSession(args.socket, args.job_id, controller_uid=os.getuid())
    try:
        opened = session.open()
        save(output / "job-opened.json", opened)
        if opened.get("manifest_sha256") != config["manifest_sha256"]:
            raise ValueError("broker manifest differs from reviewed integration job")
        # Alias generation reads headers only and remains inside the same
        # candidate sandbox; imported candidate helpers never run as root.
        alias = config.get("alias")
        if alias:
            stage = session.start("candidate-alias", "build", alias["argv"],
                                  alias.get("environment", {}), alias["cwd"], alias["timeout_s"])
            status = session.wait(stage, deadline=time.monotonic() + alias["timeout_s"])
            target = output / "alias"
            target.mkdir()
            collect_log(session, stage, target)
            save(target / "stage.json", status)
            if status.get("returncode") != 0 or status.get("cleanup_confirmed") is not True:
                raise RuntimeError("source alias preparation failed")
        build = run_build(session, "candidate", config["build"], config["paths"],
                          output / "build", source_revision)
        native = dict(config["smoke"], revision=source_revision)
        if native["argv"][0] != build["binary"]["runtime_path"]:
            raise ValueError("native smoke executable differs from the sealed build")
        result = smoke(session, native, output / "native")
        if result["status"] != "passed":
            raise RuntimeError("native smoke failed; see retained raw evidence")
        report.update(status="passed", binary_sha256=build["binary"]["sha256"],
                      actual_server_version=build["actual_server_version"],
                      completion_requests=result["completion_requests"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            if session.closed:
                raise RuntimeError("broker disconnected; positive cleanup receipt unavailable")
            cleanup = session.finish()
            save(output / "job-finished.json", cleanup)
            report["cleanup_confirmed"] = cleanup["cleanup_confirmed"]
        except Exception as error:
            report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
        session.close()
        save(output / "integration.json", report)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
