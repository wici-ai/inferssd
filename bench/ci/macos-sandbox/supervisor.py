#!/usr/bin/python3
"""Root-owned, bounded per-stage Mac isolation core. No arbitrary job command API."""
import fcntl,hashlib,json,os,pathlib,resource,select,signal,socket,stat,subprocess,sys,time,uuid
BASE=pathlib.Path('/Library/InferSSD-CI')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    b=json.dumps(d,sort_keys=True,indent=2).encode()+b'\n';fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:f.write(b)
def config():
    if os.getuid()!=0 or sys.platform!='darwin':raise RuntimeError('root Darwin required')
    for p in (BASE,BASE/'control',BASE/'control/config.json',BASE/'public',BASE/'private',BASE/'evidence',BASE/'jobs'):
        for q in (p,*p.parents):
            st=q.lstat()
            if stat.S_ISLNK(st.st_mode)or st.st_uid!=0 or st.st_mode&0o022:raise RuntimeError('unprotected installed control ancestor')
    c=json.loads((BASE/'control/config.json').read_text())
    for name,h in c['installed_files'].items():
        p=BASE/name;s=p.lstat()
        if not stat.S_ISREG(s.st_mode)or s.st_uid!=0 or s.st_mode&0o022 or sha(p)!=h:raise RuntimeError('root-installed control file changed')
    for p,h in c['tools']['files'].items():
        s=pathlib.Path(p).stat()
        if s.st_uid in (c['worker_uid'],c['runner_uid'])or s.st_mode&0o022 or sha(p)!=h:raise RuntimeError('frozen compile tool changed or is writable by a CI account')
    return c
def members(uid):
    result=[]
    raw=subprocess.check_output(['/bin/ps','-axo','uid=,pid=,rss='],text=True)
    for line in raw.splitlines():
        f=line.split()
        if len(f)!=3:raise RuntimeError('process inventory is not complete')
        if int(f[0])==uid:result.append({'pid':int(f[1]),'rss_bytes':int(f[2])*1024})
    return result
def cleanup(uid,reap=None):
    try:return _cleanup(uid,reap)
    except BaseException as error:
        # Unknown process inventory/domain cleanup is never promoted to empty.
        # The guardian's quarantine loop retains its original device lease.
        return {'confirmed':False,'signals':[],'remaining':None,'error':repr(error)}
def _cleanup(uid,reap=None):
    actions=[]
    def active():
        if reap is not None:reap()
        return members(uid)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for p in active():
            try:os.kill(p['pid'],sig);actions.append({'pid':p['pid'],'signal':int(sig)})
            except ProcessLookupError:pass
        end=time.monotonic()+5
        while active()and time.monotonic()<end:time.sleep(.05)
        if not active():
            retired=None
            if uid==62002:
                # macOS can restart per-user XPC helpers after their last
                # request. Retire only this dedicated worker user domain after
                # owned processes were signalled; never the runner/personal UID.
                r=subprocess.run(['/bin/launchctl','bootout','user/'+str(uid)],capture_output=True,text=True,timeout=5)
                retired={'returncode':r.returncode,'stderr':r.stderr}
            remaining=active()
            if not remaining:return {'confirmed':True,'signals':actions,'remaining':[],'worker_user_domain_retired':retired}
    return {'confirmed':False,'signals':actions,'remaining':active()}

def drain_logs(process,outputs,counts,limit=64<<20):
    streams=[(process.stdout,'stdout'),(process.stderr,'stderr')]
    for _ in range(32):
        active=[stream for stream,_ in streams if stream is not None and not stream.closed]
        if not active:return
        readable,_,_=select.select(active,[],[],0)
        if not readable:return
        for stream,name in streams:
            if stream not in readable:continue
            data=os.read(stream.fileno(),65536)
            if not data:stream.close();continue
            counts[name]=counts.get(name,0)+len(data)
            if counts[name]>limit:raise RuntimeError('worker log byte bound exceeded')
            outputs[name].write(data);outputs[name].flush()

def drop(uid,gid,work,limits=None):
    os.setsid();os.umask(0o077)
    limits=limits or {'files':128,'file_bytes':64<<20,'cpu_seconds':120,'processes':64}
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    for kind,key in ((resource.RLIMIT_NOFILE,'files'),(resource.RLIMIT_FSIZE,'file_bytes'),(resource.RLIMIT_CPU,'cpu_seconds'),(resource.RLIMIT_NPROC,'processes')):
        resource.setrlimit(kind,(limits[key],limits[key]))
    os.setgroups([gid]);os.setgid(gid);os.setuid(uid);os.chdir(work)
    if os.getuid()!=uid or os.geteuid()!=uid:os._exit(126)
def free_memory():
    raw=subprocess.check_output(['/usr/bin/vm_stat'],text=True)
    import re
    page=int(re.search(r'page size of (\d+) bytes',raw).group(1))
    fields=dict((k.strip(),int(v))for k,v in re.findall(r'^([^:\n]+):\s*(\d+)\.',raw,re.M))
    return page*(fields['Pages free']+fields['Pages inactive'])
def disconnected(peer):
    if peer is None:return False
    readable,_,_=select.select([peer],[],[],0)
    return bool(readable)and peer.recv(1,socket.MSG_PEEK)==b''
def lease(c):
    p=pathlib.Path(c['lock']);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);s=os.fstat(fd)
    if [s.st_dev,s.st_ino]!=c['lock_dev_inode']or not stat.S_ISREG(s.st_mode):os.close(fd);raise RuntimeError('original lock inode differs')
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:os.close(fd);raise
    return fd
def run(c,kind,probe_config=None,peer=None):
    # No caller argv/path is accepted. These immutable operations exercise the
    # installation; production requires the separately reviewed CI driver adapter.
    if kind not in ('negative','compile','metal','disconnect'):raise ValueError('stage is not in fixed installation allowlist')
    if members(c['worker_uid']):raise RuntimeError('execution UID already has live processes')
    if pathlib.Path(c['stop']).exists():raise RuntimeError('STOP is present')
    lockfd=lease(c);job=BASE/'jobs'/uuid.uuid4().hex;work=job/'work';inputs=job/'input';out=BASE/'evidence'/job.name
    job.mkdir(mode=0o711);work.mkdir(mode=0o700);inputs.mkdir(mode=0o755);out.mkdir(mode=0o700)
    os.chown(work,c['worker_uid'],c['worker_gid'])
    for name in ('tmp','home'):
        d=work/name;d.mkdir(mode=0o700);os.chown(d,c['worker_uid'],c['worker_gid'])
    profile=BASE/'public/worker.sb';argv=['/usr/bin/sandbox-exec','-f',str(profile)]
    for key,value in {'CLT':c['tools']['clt'],'INPUT':str(inputs),'PUBLIC':str(BASE/'public'),'WORK':str(work),'FROZEN':str(inputs),'MODELS':str(inputs)}.items():argv+=['-D',key+'='+value]
    if kind=='negative':
        pc=dict(probe_config,work=str(work),input=str(inputs/'readonly-input'))
        (inputs/'readonly-input').write_text('immutable model/source policy canary\n');(inputs/'readonly-input').chmod(0o444)
        save(inputs/'probe.json',pc);(inputs/'probe.json').chmod(0o444)
        command=[c['tools']['python'],'-I',str(BASE/'public/probe.py'),str(inputs/'probe.json')]
    elif kind=='compile':
        (inputs/'positive.cpp').write_text('#include <cstdio>\nint main(){std::puts("COMPILE_SANDBOX_PASS value=73");return 0;}\n');(inputs/'positive.cpp').chmod(0o444)
        command=[c['tools']['python'],'-I',str(BASE/'public/positive.py'),'compile',str(inputs/'positive.cpp'),str(work/'positive'),c['tools']['compiler'],c['tools']['sdk'],c['tools']['cmake'],c['tools']['ninja']]
    elif kind=='metal':
        command=[c['tools']['python'],'-I',str(BASE/'public/positive.py'),'metal',str(BASE/'public/metal-probe.mm'),str(work/'metal'),c['tools']['compiler'],c['tools']['sdk']]
    else:command=[c['tools']['python'],'-I',str(BASE/'public/disconnect.py')]
    argv+=command
    env={'PATH':'/usr/bin:/bin','HOME':str(work/'home'),'TMPDIR':str(work/'tmp'),'LANG':'C','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'}
    child=None;result={'stage':kind,'job':str(job),'uid':c['worker_uid'],'argv':argv,'status':'failed','samples':[]};start=time.monotonic();end=start+90
    try:
        with (out/'stdout.log').open('wb')as stdout,(out/'stderr.log').open('wb')as stderr:
            child=subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,preexec_fn=lambda:drop(c['worker_uid'],c['worker_gid'],str(work)))
            result['pid']=child.pid;outputs={'stdout':stdout,'stderr':stderr};counts={}
            while child.poll()is None:
                drain_logs(child,outputs,counts)
                active=members(c['worker_uid']);available=free_memory();rss=sum(p['rss_bytes']for p in active)
                result['samples'].append({'t':time.monotonic(),'pids':active,'available':available,'rss':rss})
                if pathlib.Path(c['stop']).exists()or time.monotonic()>end or available<3<<30 or rss>8<<30 or len(active)>64:raise RuntimeError('STOP/deadline/soft reserve/process budget refused')
                if disconnected(peer):result['controller_disconnected']=True;raise RuntimeError('controller disconnected')
                time.sleep(.1)
            drain_logs(child,outputs,counts)
            result['returncode']=child.returncode
            if child.returncode:raise RuntimeError('sandbox stage failed')
            result['status']='passed'
    except BaseException as e:result['error']=repr(e)
    finally:
        clean=cleanup(c['worker_uid'],reap=child.poll if child is not None else None);result['cleanup']=clean
        if child is not None:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:clean['confirmed']=False
        if child is not None:
            for stream in (child.stdout,child.stderr):
                if stream is not None and not stream.closed:stream.close()
        result['elapsed_s']=time.monotonic()-start
        if not clean['confirmed']:
            pathlib.Path(c['stop']).touch(exist_ok=True);result['status']='quarantined';save(out/'result.json',result)
            # Never release the lease while execution UID membership is unknown.
            while not cleanup(c['worker_uid'])['confirmed']:time.sleep(1)
            save(out/'quarantine-release.json',{'worker_uid_empty':True,'failed_status_retained':True})
        save(out/'result.json',result)if not(out/'result.json').exists()else None
        os.close(lockfd)
    return result,out

def guarded(c,kind,probe_config=None,peer=None):
    """A distinct root guardian retains the lease if its controller disappears.
    The guardian alone launches candidates. The worker receives only stdio,
    never this socket, the lease, runner credentials, or trusted RPC handles.
    """
    token=uuid.uuid4().hex;request=BASE/'private'/('request-'+token+'.json')
    save(request,{'kind':kind,'probe_config':probe_config});parent,child=socket.socketpair()
    process=None
    try:
        process=subprocess.Popen([c['tools']['python'],'-I',str(BASE/'control/stage_guard.py'),str(request),str(child.fileno())],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(child.fileno(),),close_fds=True,start_new_session=True)
        child.close();deadline=time.monotonic()+120;closed=False
        while process.poll()is None:
            if not closed and (disconnected(peer)or time.monotonic()>deadline):parent.close();closed=True
            if time.monotonic()>deadline+30:
                pathlib.Path(c['stop']).touch(exist_ok=True)
                raise RuntimeError('root guardian did not confirm cleanup; lease remains in guardian; inspect its PID '+str(process.pid))
            time.sleep(.1)
        stdout,stderr=process.communicate(timeout=5)
        if process.returncode:raise RuntimeError('root stage guardian failed: '+stderr.decode(errors='replace')[-4096:])
        reply=json.loads(stdout);result=json.loads(pathlib.Path(reply['result']).read_text())
        if result['cleanup']['confirmed']is not True:raise RuntimeError('root guardian cleanup not confirmed')
        return result,pathlib.Path(reply['result']).parent
    finally:
        parent.close();child.close()
        # Never terminate a root guardian that may still hold the device lease.
        # On this controller's exit its socket closes, triggering owned cleanup.
