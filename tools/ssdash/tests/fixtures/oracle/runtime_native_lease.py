"""Pass the existing flock open-file-description to the finite-scope worker.

No second lock acquisition and no lock inode replacement. Parent and worker must
both close their copies before the cooperating device lock can be released.
"""
import array
import fcntl
import os
from pathlib import Path
import socket
import struct
import tempfile
import sys


def peer_identity(connection):
    if sys.platform == 'darwin':
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        getpeereid = libc.getpeereid
        getpeereid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        getpeereid.restype = ctypes.c_int
        uid, gid = ctypes.c_uint(), ctypes.c_uint()
        if getpeereid(connection.fileno(), ctypes.byref(uid), ctypes.byref(gid)):
            raise OSError(ctypes.get_errno(), 'getpeereid failed')
        # Darwin SOL_LOCAL / LOCAL_PEERPID: kernel supplied peer process ID.
        pid = connection.getsockopt(0, 2)
        return pid, uid.value, gid.value
    return struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))


class LeaseServer:
    def __init__(self, lock_fd, unit=None):
        self.lock_fd = lock_fd
        self.unit = unit
        self.directory = tempfile.TemporaryDirectory(prefix='ssdash-native-lease-', dir='/tmp')
        self.path = str(Path(self.directory.name) / 'lease.sock')
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.path)
        self.listener.listen(1)
        self.listener.setblocking(False)
        self.connection = None
        self.peer = None
        self.expected_pid = None
    def poll(self):
        if self.connection is not None:
            return
        try:
            connection, _ = self.listener.accept()
        except BlockingIOError:
            return
        try:
            pid, uid, gid = peer_identity(connection)
            if uid != os.getuid():
                raise RuntimeError('lock lease requested by foreign uid')
            group = None
            if self.unit is not None:
                group = Path(f'/proc/{pid}/cgroup').read_text()
                if not group.strip().endswith('/' + self.unit):
                    raise RuntimeError('lock lease peer is not in the exact owned scope')
            elif pid != self.expected_pid:
                raise RuntimeError('lock lease peer is not the exact owned worker')
            # Parent was already holding flock before creating the scope.
            # SCM_RIGHTS duplicates that open-file-description in the worker.
            sent = connection.sendmsg([b'L'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [self.lock_fd]))])
            if sent != 1:
                raise RuntimeError('lock lease transfer incomplete')
            self.connection = connection
            self.peer = {'pid': pid, 'uid': uid, 'gid': gid, 'cgroup': group}
        except BaseException:
            connection.close()
            raise
    def close(self):
        if self.connection:
            self.connection.close()
        self.listener.close()
        self.directory.cleanup()


class Lease:
    def __init__(self, path, lock_path=None, expected_identity=None):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.fd = None
        try:
            self.connection.settimeout(10)
            self.connection.connect(path)
            message, control, flags, _ = self.connection.recvmsg(1, socket.CMSG_SPACE(array.array('i').itemsize))
            descriptors = array.array('i')
            for level, kind, data in control:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    descriptors.frombytes(data[:len(data) - len(data) % descriptors.itemsize])
            if message != b'L' or flags & socket.MSG_CTRUNC or len(descriptors) != 1:
                for fd in descriptors:
                    os.close(fd)
                raise RuntimeError('invalid lock lease transfer')
            self.fd = descriptors[0]
            got = os.fstat(self.fd)
            if expected_identity is None:
                expected = Path(lock_path).stat()
                expected_identity = (expected.st_dev, expected.st_ino)
            if (got.st_dev, got.st_ino) != expected_identity:
                raise RuntimeError('lock lease inode differs')
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.connection.setblocking(False)
        except BaseException:
            self.close()
            raise
    def check(self):
        try:
            data = self.connection.recv(1, socket.MSG_PEEK)
        except BlockingIOError:
            return
        if data == b'':
            raise InterruptedError('controller connection lost; cleanup while retaining lock lease')
        raise RuntimeError('unexpected controller lease message')
    def close(self):
        self.connection.close()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
