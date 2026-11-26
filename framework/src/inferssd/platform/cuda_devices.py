# SPDX-License-Identifier: Apache-2.0
"""Read visible device UUIDs from the CUDA driver in the server's environment."""

from __future__ import annotations

import ctypes
import subprocess
import sys
import uuid
from collections.abc import Mapping
from typing import Optional


def visible(environment: Mapping[str, str]) -> Optional[tuple[str, ...]]:
    from inferssd.platform.captured_command import run_captured

    # CUDA reads visibility at driver initialization. A fresh process sees the
    # exact environment the server will receive even if its parent used CUDA.
    try:
        run = run_captured([sys.executable, "-I", __file__], timeout=10, env=environment)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return tuple(run.stdout.splitlines()) if run.returncode == 0 else None


def driver_uuids() -> list[str]:
    driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")

    def check(status: int) -> None:
        if status:
            raise OSError(f"CUDA driver returned {status}")

    status = driver.cuInit(0)
    if status == 100:  # CUDA_ERROR_NO_DEVICE
        return []
    check(status)
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)))
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    ids = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), ordinal))
        raw = (ctypes.c_ubyte * 16)()
        check(get_uuid(ctypes.byref(raw), device))
        ids.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
    return ids


if __name__ == "__main__":
    for device_id in driver_uuids():
        print(device_id)
