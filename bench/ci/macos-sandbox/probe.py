#!/usr/bin/python3
"""Immutable negative probe executed only as the sandboxed execution UID."""
import errno,json,os,pathlib,signal,socket,subprocess,sys,time

def denied(call):
    try:call()
    except OSError as e:
        if e.errno in (errno.EACCES,errno.EPERM):return {'denied':True,'errno':e.errno}
        raise
    raise RuntimeError('an access that must be denied succeeded')
def read(p):
    if pathlib.Path(p).is_dir():os.listdir(p)
    else:
        with open(p,'rb')as f:f.read(1)
def write(p):
    fd=os.open(p,os.O_WRONLY|os.O_APPEND);os.write(fd,b'UNAUTHORIZED');os.close(fd)
def connect_inet(port):
    with socket.socket()as s:s.settimeout(2);s.connect(('127.0.0.1',port))
def connect_external():
    # Seatbelt must deny connect before any connection; this probe sends no payload.
    with socket.socket()as s:s.settimeout(2);s.connect(('1.1.1.1',443))
def connect_unix(path):
    with socket.socket(socket.AF_UNIX)as s:s.settimeout(2);s.connect(path)
def main():
    c=json.loads(pathlib.Path(sys.argv[1]).read_text());out={'nonce':c['nonce'],'uid':os.getuid(),'euid':os.geteuid(),'groups':os.getgroups(),'negative':{}}
    require=lambda v: v or (_ for _ in ()).throw(RuntimeError('identity mismatch'))
    require(os.getuid()==c['worker_uid'] and os.getuid()!=c['runner_uid'] and os.geteuid()!=0)
    require(c['worker_gid']in os.getgroups()and not({0,20,80,c['runner_uid']}&set(os.getgroups())))
    n=out['negative']
    for name,path in [('personal_home',c['personal_home']),('runner_canary',c['runner_canary']),('private_canary',c['private_canary']),('model_write',c['input'])]:
        n[name]=denied((lambda p=path:write(p))if name=='model_write'else(lambda p=path:read(p)))
    for name,path in [('control_write',c['control_canary']),('evidence_write',c['evidence_canary'])]:n[name]=denied(lambda p=path:write(p))
    n['loopback_connect']=denied(lambda:connect_inet(c['tcp_port']))
    n['external_connect']=denied(connect_external)
    n['host_socket']=denied(lambda:connect_unix(c['unix_socket']))
    n['root_setuid']=denied(lambda:os.setuid(0))
    n['runner_signal']=denied(lambda:os.kill(c['runner_pid'],signal.SIGTERM))
    for name,argv in [('sudo',['/usr/bin/sudo','-n','/usr/bin/true']),('launchctl',['/bin/launchctl','print','system']),('osascript',['/usr/bin/osascript','-e','return 1'])]:
        n[name]=denied(lambda a=argv:subprocess.run(a,check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
    link=pathlib.Path(c['work'])/('outside-link-'+str(os.getpid()));link.symlink_to(c['runner_canary']);n['symlink_escape']=denied(lambda:read(link))
    n['hardlink_escape']=denied(lambda:os.link(c['runner_canary'],pathlib.Path(c['work'])/('outside-hardlink-'+str(os.getpid()))))
    # Sandboxes are inherited across fork+exec. The fixed child repeats actual
    # protected reads/connect; no candidate or arbitrary assertion is trusted.
    if '--child'not in sys.argv:
        r=subprocess.run([sys.executable,'-I',__file__,sys.argv[1],'--child'],check=True,capture_output=True,text=True,timeout=20)
        child=json.loads(r.stdout);require(child['uid']==c['worker_uid']);out['inherited_child']=child
        pid=os.fork()
        if pid==0:
            os.setsid()
            while True:time.sleep(1)
        out['detached_child_pid']=pid
    else:
        out['child_mode']=True
    out['status']='passed';print(json.dumps(out),flush=True)
if __name__=='__main__':main()
