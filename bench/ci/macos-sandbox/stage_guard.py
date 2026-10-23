#!/usr/bin/python3 -I
"""Independent privileged per-stage guardian, the sole parent of worker processes."""
import json,os,pathlib,signal,socket,sys
BASE=pathlib.Path('/Library/InferSSD-CI')
sys.path.insert(0,str(BASE/'control'))
import supervisor as s
intent=[]
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,lambda n,f:intent.append(n))
def main():
    c=s.config();request=pathlib.Path(sys.argv[1]);fd=int(sys.argv[2])
    if request.parent!=BASE/'private' or request.is_symlink()or request.stat().st_uid!=0:raise RuntimeError('untrusted root request')
    data=json.loads(request.read_text())
    peer=socket.socket(fileno=fd)
    # Only the root controller owns the opposite endpoint. Any controller death
    # becomes EOF. A malicious worker never inherits either endpoint.
    if not data['kind']in('negative','compile','metal','disconnect'):raise RuntimeError('unknown stage')
    original=s.disconnected
    s.disconnected=lambda p:bool(intent)or original(p)
    result,where=s.run(c,data['kind'],data.get('probe_config'),peer)
    print(json.dumps({'result':str(where/'result.json')}),flush=True)
if __name__=='__main__':main()
