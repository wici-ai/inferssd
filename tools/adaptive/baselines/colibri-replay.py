#!/usr/bin/env python3
"""Two real-text turns using the native, flushed DATA protocol. No product edits."""
import argparse,json,pathlib,subprocess,time,threading
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--cap',type=int,required=True);p.add_argument('--script',required=True);p.add_argument('--out',required=True);a=p.parse_args()
assert a.cap>=1;out=pathlib.Path(a.out);out.mkdir(exist_ok=True)
turns=json.loads(pathlib.Path(a.script).read_text());assert len(turns)==2 and all(isinstance(t['text'],str) for t in turns)
history='<｜begin▁of▁sentence｜>'
rows=[]
stop=threading.Event()
def monitor():
    try:
        cg=next(x.split(':',2)[2] for x in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0:'))
        root=pathlib.Path('/sys/fs/cgroup')/cg.lstrip('/')
        while not stop.is_set():
            sample={'us':time.monotonic_ns()/1000,'cgroup':cg}
            for name in ['memory.max','memory.current','memory.peak','memory.events','cpuset.cpus.effective','cpu.stat']:
                try:sample[name]=(root/name).read_text()
                except OSError:pass
            with (out/'cgroup.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            stop.wait(.1)
    except OSError:pass
threading.Thread(target=monitor,daemon=True).start()
launch=time.monotonic_ns()/1000
with (out/'engine.stderr').open('wb') as err,(out/'protocol.jsonl').open('w') as wire:
    proc=subprocess.Popen([a.binary,str(a.cap)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    def line():
        data=proc.stdout.readline()
        if not data:raise RuntimeError(f'engine ended before frame, rc={proc.poll()}')
        wire.write(json.dumps(dict(us=time.monotonic_ns()/1000,line=data.decode(errors='replace').rstrip()))+'\n');wire.flush();return data.decode(errors='replace').strip()
    try:
        while 'READY' not in line():pass
        # Consume startup telemetry before writing a long prompt: otherwise the
        # bounded stdout/stdin pipes can wait on one another.
        while not line().startswith('EMAP '):pass
        (out/'startup.json').write_text(json.dumps({'ready_s':(time.monotonic_ns()/1000-launch)/1e6})+'\n')
        for turn,item in enumerate(turns):
            history+='<｜User｜>'+item['text']+'<｜Assistant｜></think>'
            payload=history.encode();start=time.monotonic_ns()/1000
            proc.stdin.write(f'SUBMIT {turn+1} 0 {len(payload)} 256 0 1\n'.encode()+payload+b'\n');proc.stdin.flush()
            (out/f'turn-{turn}-submitted.json').write_text(json.dumps(dict(turn=turn,start_us=start,payload_bytes=len(payload),max_tokens=256))+'\n')
            stamps=[];pieces=[];prompt_n=None;done=None
            while done is None:
                header=line();parts=header.split()
                if not parts:continue
                if parts[0]=='ERROR':raise RuntimeError(header)
                if parts[0]=='ACCEPT':
                    prompt_n=int(parts[2]);(out/f'turn-{turn}-accepted.json').write_text(json.dumps(dict(turn=turn,prompt_n=prompt_n,us=time.monotonic_ns()/1000))+'\n')
                if parts[0]=='DATA':
                    n=int(parts[2]);piece=proc.stdout.read(n);assert len(piece)==n;assert proc.stdout.read(1)==b'\n'
                    stamps.append(time.monotonic_ns()/1000);pieces.append(piece)
                    with (out/f'turn-{turn}-data.jsonl').open('a') as f:f.write(json.dumps(dict(frame=len(stamps),us=stamps[-1],text=piece.decode(errors='replace')))+'\n')
                if parts[0]=='DONE':done=parts
            emitted=int(done[3]);count_matches=len(stamps)==emitted;end=time.monotonic_ns()/1000
            while not line().startswith('EMAP '):pass
            text=b''.join(pieces).decode(errors='replace');(out/f'turn-{turn}.txt').write_text(text)
            row=dict(turn=turn,prompt_n=prompt_n,cache_n=0,max_tokens=256,emitted=emitted,data_frames=len(stamps),data_count_matches_emitted=count_matches,
                     ttft_s=(stamps[0]-start)/1e6 if stamps else None,wall_s=(end-start)/1e6,
                     engine_total_including_prefill_tps=float(done[4]),engine_hit_percent=float(done[5]),engine_rss_GB=float(done[6]),length_limited=int(done[8]),
                     decode_tps_129_256=128e6/(stamps[255]-stamps[127]) if count_matches and len(stamps)>=256 else None,
                     decode_after_first_token_tps=(emitted-1)*1e6/(stamps[-1]-stamps[0]) if count_matches and emitted>1 else None,
                     data_us=stamps,metric_note='DATA token ordinals valid only if frame count equals DONE emitted; no ignore-EOS option; DONE rate includes prefill')
            rows.append(row);(out/'turns.json').write_text(json.dumps(rows,indent=2)+'\n')
            history+=text+'<｜end▁of▁sentence｜>'
        proc.stdin.close();proc.wait(timeout=60)
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
        (out/'engine-exit.json').write_text(json.dumps(dict(rc=proc.returncode,wall_s=(time.monotonic_ns()/1000-launch)/1e6))+'\n')
        time.sleep(.15)
        stop.set()
assert len(rows)==2
print(json.dumps(rows,indent=2))
