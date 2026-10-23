#!/usr/bin/python3
"""Independent root guardian retains the original flock until exact-unit cleanup."""

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import UNIT, protected
from executor import CLEAN_ENV, cleanup, quarantine, show


def main(args):
    if os.geteuid() != 0 or not UNIT.fullmatch(args.unit):
        raise PermissionError("root-owned guardian only")
    receipt = {"schema_version": 1, "unit": args.unit, "cleanup_confirmed": False}
    process = None
    try:
        os.fstat(args.lock_fd)
        if args.command:
            command = json.loads(protected(args.command).read_text())
            if not isinstance(command, list) or command[0] != '/usr/bin/systemd-run' or '--unit=' + args.unit not in command:
                raise ValueError('invalid root-owned stage launch command')
            # This independent supervisor owns BOTH creation and destruction.
            # A broker death cannot race an unowned queued systemd launch.
            process = subprocess.Popen(command, env=CLEAN_ENV, stdin=subprocess.DEVNULL,
                                       close_fds=True, start_new_session=True)
        os.write(args.ack_fd, b"R")
        os.close(args.ack_fd)
        # No candidate inherits this descriptor or the original lease.
        if process is None:
            while os.read(args.read_fd, 1):
                pass
        else:
            disconnected, observed = False, False
            deadline = time.monotonic() + 15
            while process.poll() is None:
                observed |= show(args.unit).get('LoadState') == 'loaded'
                ready, _, _ = select.select([args.read_fd], [], [], .05)
                if ready and not os.read(args.read_fd, 1):
                    disconnected = True
                if disconnected and observed:
                    break
                if not observed and time.monotonic() > deadline:
                    raise RuntimeError('owned systemd launch did not resolve')
            receipt['returncode'] = process.poll()
            receipt['controller_disconnected'] = disconnected
        receipt["cleanup"] = cleanup(args.unit)
        if process is not None:
            process.wait(timeout=10)
            # Query after the owning systemd-run client has terminated too;
            # there is no surviving launcher that can create a later unit.
            receipt['cleanup'] = cleanup(args.unit)
            receipt['returncode'] = process.returncode
        receipt["cleanup_confirmed"] = True
    except BaseException as error:
        receipt["error"] = type(error).__name__ + ": " + str(error)
        quarantine(args.stop, "guardian could not confirm " + args.unit)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
            try:
                cleanup(args.unit)
            except Exception:
                quarantine(args.stop, 'supervisor final cleanup unconfirmed: ' + args.unit)
        with Path(args.receipt).open("x") as output:
            json.dump(receipt, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        # Closing this final duplicate releases the flock, only after cleanup
        # or a durable STOP that blocks every subsequent broker admission.
        os.close(args.lock_fd)
    return (0 if receipt.get('controller_disconnected') else receipt.get('returncode', 0) or 0) if receipt["cleanup_confirmed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("lock-fd", "read-fd", "ack-fd"):
        parser.add_argument("--" + field, type=int, required=True)
    for field in ("unit", "stop", "receipt"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument('--command')
    raise SystemExit(main(parser.parse_args()))
