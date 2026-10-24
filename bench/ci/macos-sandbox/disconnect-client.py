#!/usr/bin/python3
"""Immutable runner-side disconnect probe: no credential/environment transport."""
import socket,sys,time
with socket.socket(socket.AF_UNIX)as s:
    s.connect(sys.argv[1]);s.sendall(b'fixed-disconnect-probe\n')
    if s.recv(2)!=b'OK':raise SystemExit(2)
    time.sleep(2)
