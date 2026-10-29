"""Trusted coordinator transport for one leased job and isolated worker stages.

The socket stays outside candidate processes. Closing it asks the independent
broker guardian to clean up; only the broker's positive receipt proves cleanup.
"""

import base64
from dataclasses import dataclass
import json
import re
import socket
import time


MAX_RESPONSE = 1024 * 1024
# Freezing seals every file of a build tree (hash and fsync), and a baseline job's
# build is then copied into the broker's build cache; a CUDA build on a slow
# disk takes minutes, not the 30 s of an ordinary request.
FREEZE_TIMEOUT_S = 900


class BrokerError(RuntimeError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BrokerError("duplicate broker response key")
        result[key] = value
    return result


@dataclass(frozen=True)
class Stage:
    stage_id: str
    handle: str
    worker_pid: int
    worker_uid: int
    birth_identity: object
    scope_identity: object


class StageSession:
    def __init__(self, address, job_id, *, controller_uid, connect=socket.socket):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("job id must be a fresh 32-character hexadecimal id")
        self.job_id, self.controller_uid = job_id, controller_uid
        self.connection = connect(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(30)
        self.connection.connect(str(address))
        self.buffer = bytearray()
        self.closed = False
        self.finished = False
        self.stages = {}

    def request(self, operation, *, timeout=30, **fields):
        if self.closed or self.finished:
            raise BrokerError("broker session is closed")
        request = dict(schema_version=1, operation=operation,
                       job_id=self.job_id, **fields)
        encoded = json.dumps(request, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_RESPONSE:
            raise BrokerError("oversized broker request")
        try:
            self.connection.settimeout(timeout)
            self.connection.sendall(encoded)
            while b"\n" not in self.buffer:
                chunk = self.connection.recv(min(65536, MAX_RESPONSE + 1 - len(self.buffer)))
                if not chunk:
                    raise BrokerError("broker disconnected; cleanup is unconfirmed")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_RESPONSE:
                    raise BrokerError("oversized broker response")
            line, _, remaining = self.buffer.partition(b"\n")
            self.buffer = bytearray(remaining)
            response = json.loads(line, object_pairs_hook=unique_object,
                                  parse_constant=lambda _: (_ for _ in ()).throw(
                                      BrokerError("nonfinite broker value")))
            if not isinstance(response, dict):
                raise BrokerError("broker response must be an object")
            if response.get("error") or response.get("ok") is False:
                raise BrokerError("broker refused " + operation + ": " + str(response.get("error", "refused")))
            if response.get("job_id", self.job_id) != self.job_id:
                raise BrokerError("broker returned another job identity")
            return response
        except BaseException as error:
            # A late response must never be mistaken for the next operation.
            self.close()
            context = "broker " + operation + " stage=" + str(fields.get("stage_id", "<job>"))
            if isinstance(error, (TimeoutError, socket.timeout)):
                # Preserve the timeout type on Python 3.9 too, where add_note
                # does not exist and JSON reports only serialize str(error).
                raise type(error)(context + ": " + str(error)) from error
            if hasattr(error, "add_note"):
                error.add_note(context)
            raise

    def open(self):
        return self.request("job.open")

    def start(self, stage_id, kind, argv, environment, cwd, timeout_s):
        if stage_id in self.stages:
            raise ValueError("each stage id is used once per job")
        response = self.request("stage.start", stage_id=stage_id, kind=kind,
                                argv=list(argv), environment=dict(environment),
                                cwd=cwd, timeout_s=timeout_s)
        try:
            stage = Stage(stage_id=stage_id, **{k: response[k] for k in (
                "handle", "worker_pid", "worker_uid", "birth_identity", "scope_identity")})
            if (not isinstance(stage.handle, str) or not stage.handle
                    or type(stage.worker_pid) is not int or stage.worker_pid <= 1
                    or type(stage.worker_uid) is not int
                    or stage.worker_uid in (0, self.controller_uid)
                    or not stage.birth_identity or not stage.scope_identity):
                raise ValueError("worker identity is incomplete or shares controller privileges")
            self.stages[stage_id] = stage
            return stage
        except BaseException:
            self.close()
            raise

    def call(self, stage, operation, **fields):
        if self.stages.get(stage.stage_id) != stage:
            raise ValueError("stage does not belong to this session")
        response = self.request(operation, stage_id=stage.stage_id,
                                handle=stage.handle, **fields)
        for key, expected in (("stage_id", stage.stage_id), ("handle", stage.handle)):
            if response.get(key, expected) != expected:
                self.close()
                raise BrokerError("broker stage identity changed")
        return response

    def output(self, stage, stream, offset=0, limit=65536):
        if stream not in ("stdout", "stderr") or type(offset) is not int or offset < 0:
            raise ValueError("invalid stage output cursor")
        if type(limit) is not int or not 1 <= limit <= 65536:
            raise ValueError("invalid output chunk bound")
        value = self.call(stage, "stage.output", stream=stream, offset=offset, limit=limit)
        data = base64.b64decode(value["data_b64"], validate=True)
        if (value.get("offset") != offset or value.get("next_offset") != offset + len(data)
                or len(data) > limit or type(value.get("eof")) is not bool):
            self.close()
            raise BrokerError("invalid output byte offsets or chunk size")
        return data, value["next_offset"], value["eof"]

    def wait(self, stage, *, deadline, interrupted=lambda: False):
        while True:
            if interrupted() or time.monotonic() >= deadline:
                cleanup = self.call(stage, "stage.stop", timeout=50)
                if cleanup.get("cleanup_confirmed") is not True:
                    raise BrokerError("stage interrupted and cleanup is unconfirmed")
                raise InterruptedError("stage interrupted or deadline exceeded")
            value = self.call(stage, "stage.wait", wait_s=min(10, max(1, int(deadline - time.monotonic()))))
            if value.get("running") is False:
                if type(value.get("returncode")) is not int or value.get("cleanup_confirmed") is not True:
                    raise BrokerError("completed stage lacks exit/cleanup confirmation")
                return value
            if value.get("running") is not True:
                raise BrokerError("stage status lacks running state")

    def freeze(self, stage, **fields):
        return self.call(stage, "job.freeze", timeout=FREEZE_TIMEOUT_S, **fields)

    def finish(self):
        result = self.request("job.finish", timeout=60)
        if result.get("cleanup_confirmed") is not True:
            self.close()
            raise BrokerError("job cleanup is unconfirmed")
        self.finished = True
        self.close()
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.connection.close()

    def abort(self):
        """Interrupt an in-flight RPC; disconnect is never a cleanup receipt."""
        if not self.closed:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and not self.closed:
            self.finish()
        else:
            self.close()
