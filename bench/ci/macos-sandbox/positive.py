#!/usr/bin/python3
import subprocess
import sys

mode, source, binary, compiler, sdk = sys.argv[1:6]
if mode == 'compile':
    for tool in sys.argv[6:]:
        subprocess.run([tool, '--version'], check=True, timeout=10)
    argv = [compiler, '--driver-mode=g++', '-isysroot', sdk, '-std=c++17', '-Wall', '-Wextra', '-Werror', source, '-o', binary]
elif mode == 'metal':
    argv = [compiler, '--driver-mode=g++', '-isysroot', sdk, '-std=c++17', '-fobjc-arc', source, '-framework', 'Foundation', '-framework', 'Metal', '-o', binary]
else:
    raise SystemExit(2)
subprocess.run(argv, check=True, timeout=45)
subprocess.run([binary], check=True, timeout=20)
