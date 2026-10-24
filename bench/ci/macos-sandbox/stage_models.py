#!/usr/bin/python3
"""Root-only APFS model clone staging. Never opens the personal home to workers."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import stat

import supervisor as core


def require(condition, why):
    if not condition:
        raise ValueError(why)


def stage(manifest, name):
    config = core.config()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', name), 'invalid model set name')
    # A trusted operator supplies the exact source paths and previous complete
    # hashes. A candidate PR cannot select host files for privileged copying.
    for item in (manifest, *manifest.parents):
        info = item.lstat()
        require(info.st_uid == 0 and not info.st_mode & 0o022 and not stat.S_ISLNK(info.st_mode), 'manifest must be root protected')
    rows = json.loads(manifest.read_text())['files']
    require(isinstance(rows, list) and 0 < len(rows) < 256, 'invalid file inventory')
    destination = core.BASE / 'models' / name
    require(not destination.exists(), 'model set already exists; no overwrite')
    destination.mkdir(mode=0o700)
    library = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    clone = library.fclonefileat
    clone.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
    clone.restype = ctypes.c_int
    receipt = []
    directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        names = set()
        for row in rows:
            source = Path(row['path'])
            filename = row['name']
            require(filename == Path(filename).name and filename not in ('', '.', '..') and filename not in names, 'invalid/duplicate destination name')
            names.add(filename)
            require(filename.endswith(('.gguf', '.json', '.jinja')), 'not a reviewed model input type')
            require(re.fullmatch('[0-9a-f]{64}', row['sha256']), 'complete expected hash required')
            require(Path(config['personal_home']) in source.parents and not any(p.is_symlink() for p in (source, *source.parents)), 'source not under the reviewed personal model home or has symlink')
            fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                before = os.fstat(fd)
                require(stat.S_ISREG(before.st_mode) and before.st_uid == 501 and before.st_size == row['size'], 'model identity/size changed')
                # APFS clone gets a distinct inode. Failure stays failed rather
                # than copying a terabyte unexpectedly or hardlinking a private file.
                if clone(fd, directory_fd, filename.encode(), 0):
                    raise OSError(ctypes.get_errno(), 'APFS clone failed; no hardlink/full-copy fallback')
                target = destination / filename
                os.chown(target, 0, 0)
                target.chmod(0o444)
                require(core.sha(target) == row['sha256'], 'cloned complete model hash differs')
                after = os.fstat(fd)
                require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'source model changed while cloning')
                copied = target.stat()
                require((copied.st_dev, copied.st_ino) != (before.st_dev, before.st_ino) and copied.st_nlink == 1, 'model clone unexpectedly shares an inode')
                receipt.append({**row, 'staged_path': str(target), 'source_dev_inode': [before.st_dev, before.st_ino], 'staged_dev_inode': [copied.st_dev, copied.st_ino], 'readonly': True})
            finally:
                os.close(fd)
        destination.chmod(0o555)
        core.save(core.BASE / 'evidence' / ('model-staging-' + name + '.json'), {'files': receipt, 'method': 'APFS clone, complete hash verified'})
        return {'path': str(destination), 'files': len(receipt), 'readonly': True}
    finally:
        os.close(directory_fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--execute-reviewed-staging', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.manifest, args.name)))
