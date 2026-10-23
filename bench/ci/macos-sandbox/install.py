#!/usr/bin/python3
"""Reviewed root-only installation. No runner registration, network, password or enablement."""
import argparse,grp,hashlib,json,os,pathlib,pwd,shutil,stat,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/Library/InferSSD-CI')
ACCOUNTS=(('_inferssd_runner',62001),('_inferssd_worker',62002))
def cmd(*a):return subprocess.check_output(a,text=True).strip()
def require(ok,why):
    if not ok:raise RuntimeError(why)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def protected(p,tool=False):
    for q in (p,*p.parents):
        s=q.lstat()
        if tool and q==pathlib.Path('/Applications'):
            require(s.st_uid==0 and s.st_gid==80 and stat.S_IMODE(s.st_mode)==0o775 and not stat.S_ISLNK(s.st_mode),'unexpected Applications ownership/mode')
            require(len(cmd('/bin/ls','-lde',str(q)).splitlines())==1,'Applications ACL requires separate review')
        else:require(not stat.S_ISLNK(s.st_mode)and s.st_uid==0 and not(s.st_mode&0o022),'unprotected install ancestor: '+str(q))
def user(name,ident,home,resume=False):
    # Existing collisions are never adopted, reset or deleted.
    try:pwd.getpwnam(name)
    except KeyError:pass
    else:
        require(resume,'account already exists; inspect prior installation rather than mutate it')
        p=pwd.getpwnam(name)
        require(p.pw_uid==ident and p.pw_gid==ident and p.pw_dir==str(home)and p.pw_shell=='/usr/bin/false'and p.pw_passwd=='*','existing account differs from reviewed disabled identity')
        require(grp.getgrnam(name).gr_gid==ident and grp.getgrgid(ident).gr_name==name,'existing group differs')
        require(not({0,20,80}&set(map(int,cmd('/usr/bin/id','-G',name).split()))),'service account has privileged membership')
        return
    for value in (lambda:pwd.getpwuid(ident),lambda:grp.getgrgid(ident),lambda:grp.getgrnam(name)):
        try:value()
        except KeyError:pass
        else:raise RuntimeError('fixed UID/GID collision; no account changes authorized')
    cmd('/usr/bin/dscl','.','-create','/Groups/'+name)
    for k,v in [('PrimaryGroupID',str(ident)),('Password','*')]:cmd('/usr/bin/dscl','.','-create','/Groups/'+name,k,v)
    cmd('/usr/bin/dscl','.','-create','/Users/'+name)
    for k,v in [('UniqueID',str(ident)),('PrimaryGroupID',str(ident)),('NFSHomeDirectory',str(home)),('UserShell','/usr/bin/false'),('IsHidden','1'),('AuthenticationAuthority',';DisabledUser;'),('Password','*')]:cmd('/usr/bin/dscl','.','-create','/Users/'+name,k,v)
    p=pwd.getpwnam(name);require(p.pw_uid==ident and p.pw_gid==ident,'created account identity differs')
    require(not({0,20,80}&set(map(int,cmd('/usr/bin/id','-G',name).split()))),'service account has privileged membership')
    require(p.pw_passwd=='*'and p.pw_shell=='/usr/bin/false','account is not disabled')
def formula_root(p):
    p=p.resolve();parts=p.parts
    require(parts[:4]==('/','opt','homebrew','Cellar')and len(parts)>=7,'tool is not an exact Homebrew Cellar binary')
    return pathlib.Path(*parts[:6])
def tools():
    cmake=pathlib.Path('/opt/homebrew/bin/cmake').resolve();ninja=pathlib.Path('/opt/homebrew/bin/ninja').resolve()
    roots=sorted({str(formula_root(cmake)),str(formula_root(ninja))});files={};pending=[cmake,ninja]
    while pending:
        p=pending.pop().resolve()
        if str(p)in files:continue
        s=p.stat();require(s.st_uid not in (62001,62002)and not(s.st_mode&0o022),'tool writable by service account/group/world')
        files[str(p)]=sha(p)
        for line in cmd('/usr/bin/otool','-L',str(p)).splitlines()[1:]:
            dependency=line.strip().split(' (',1)[0]
            if dependency.startswith('/opt/homebrew/'):
                q=pathlib.Path(dependency).resolve();pending.append(q)
            elif not dependency.startswith(('/usr/lib/','/System/')):raise RuntimeError('unresolved non-system tool dependency: '+dependency)
    clt=pathlib.Path(cmd('/usr/bin/xcode-select','-p')).resolve();protected(clt,tool=True)
    python=pathlib.Path(sys.executable).resolve();require(clt in python.parents,'Python interpreter is outside frozen Xcode tree');files[str(python)]=sha(python)
    compiler=pathlib.Path(cmd('/usr/bin/xcrun','--find','clang++')).resolve();sdk=pathlib.Path(cmd('/usr/bin/xcrun','--show-sdk-path')).resolve()
    require(clt in compiler.parents and clt in sdk.parents,'compiler/SDK outside frozen developer tree');files[str(compiler)]=sha(compiler)
    return {'compiler':str(compiler),'sdk':str(sdk),'python':str(python),'cmake':str(cmake),'ninja':str(ninja),'read_roots':roots,'files':files,'clt':str(clt),'clt_ancestry':[{'path':str(p),'uid':p.stat().st_uid,'gid':p.stat().st_gid,'mode':oct(stat.S_IMODE(p.stat().st_mode))}for p in (clt,*clt.parents)]}
def arguments(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute-reviewed-install',action='store_true');ap.add_argument('--resume-reviewed-install',action='store_true');ap.add_argument('--resume-checkpoint-sha256')
    ap.add_argument('--reviewed-hostname',required=True)
    ap.add_argument('--personal-home',required=True,type=pathlib.Path);ap.add_argument('--personal-uid',required=True,type=int)
    ap.add_argument('--lock-path',required=True,type=pathlib.Path)
    ap.add_argument('--lock-device',required=True,type=int);ap.add_argument('--lock-inode',required=True,type=int)
    return ap.parse_args(argv)
def main():
    a=arguments()
    require(a.execute_reviewed_install and os.getuid()==0 and sys.platform=='darwin','explicit reviewed root Darwin install required')
    require(cmd('/bin/hostname')==a.reviewed_hostname,'wrong target device')
    require(not BASE.exists()or a.resume_reviewed_install,'installation directory already exists; preserve it and review continuation')
    resume_accounts=[]
    if a.resume_reviewed_install:
        checkpoint=BASE/'private/install-checkpoint.json';protected(checkpoint)
        require(sha(checkpoint)==a.resume_checkpoint_sha256,'resume checkpoint hash differs')
        data=json.loads(checkpoint.read_text())
        require(data['phase']=='empty-layout-partial-accounts'and data['package_manifest_sha256']==sha(HERE/'MANIFEST.json'),'wrong reviewed resume phase/package')
        expected={'control','public','private','evidence','jobs','inputs','models','runner','private/job-manifests','private/install-checkpoint.json'}
        require({str(p.relative_to(BASE))for p in BASE.rglob('*')}==expected,'resume installation contains unreviewed paths')
        for row in data['paths']:
            p=pathlib.Path(row['path']);st=p.lstat()
            require(stat.S_ISDIR(st.st_mode)and not p.is_symlink()and[st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode)]==row['identity'],'resume directory identity differs')
        resume_accounts=data['created_accounts']
        require(set(resume_accounts)<=set(n for n,_ in ACCOUNTS),'unknown checkpoint account')
    require(pathlib.Path('/usr/bin/sandbox-exec').is_file(),'sandbox-exec missing; no unsafe fallback')
    personal=a.personal_home
    require(personal.is_absolute()and '..'not in personal.parts and not personal.is_symlink()and a.personal_uid>0,'explicit reviewed personal home required')
    ps=personal.stat();require(ps.st_uid==a.personal_uid and stat.S_IMODE(ps.st_mode)==0o750,'personal home identity/mode changed')
    lock=a.lock_path
    require(lock.is_absolute()and '..'not in lock.parts and personal in lock.parents and a.lock_device>=0 and a.lock_inode>0,'explicit original lock identity required')
    ls=lock.lstat();require(stat.S_ISREG(ls.st_mode)and [ls.st_dev,ls.st_ino]==[a.lock_device,a.lock_inode],'original lock changed')
    stop=lock.parent/'STOP';require(not stop.exists(),'STOP present')
    # Preflight both identities before creating either, to avoid a partial collision.
    for name,i in ACCOUNTS:
        if name in resume_accounts:continue
        require(not any(p.pw_name==name or p.pw_uid==i for p in pwd.getpwall()),'account collision')
        require(not any(g.gr_name==name or g.gr_gid==i for g in grp.getgrall()),'group collision')
    for executable in ('/usr/bin/dscl','/usr/bin/id','/usr/bin/sandbox-exec','/usr/bin/otool'):
        require(pathlib.Path(executable).is_file()and os.access(executable,os.X_OK),'required installer tool missing: '+executable)
    t=tools()
    if not a.resume_reviewed_install:BASE.mkdir(mode=0o755)
    protected(BASE)
    for name,mode in [('control',0o755),('public',0o755),('private',0o700),('evidence',0o700),('jobs',0o711),('inputs',0o755),('models',0o755),('runner',0o750)]:
        p=BASE/name
        if not a.resume_reviewed_install:p.mkdir(mode=mode);p.chmod(mode)
    if not a.resume_reviewed_install:(BASE/'private/job-manifests').mkdir(mode=0o700)
    # Service account creation is deliberately separate from runner registration.
    user(*ACCOUNTS[0],BASE/'runner',resume=ACCOUNTS[0][0]in resume_accounts);user(*ACCOUNTS[1],pathlib.Path('/var/empty'),resume=ACCOUNTS[1][0]in resume_accounts)
    os.chown(BASE/'runner',62001,62001)
    for name in ('supervisor.py','verify.py','stage_guard.py','executor.py','client.py','stage_models.py','verify_broker.py'):
        q=BASE/'control'/name;shutil.copyfile(HERE/name,q);q.chmod(0o755)
    for name in ('probe.py','positive.py','disconnect.py','disconnect-client.py','metal-probe.mm'):
        q=BASE/'public'/name;shutil.copyfile(HERE/name,q);q.chmod(0o444)
    profile=(HERE/'worker.sb').read_text()
    roots=' '.join('(subpath '+json.dumps(p)+')'for p in t['read_roots'])
    # Dependent libraries get literal access only, not their entire formula tree.
    libs=' '.join('(literal '+json.dumps(p)+')'for p in t['files']if not any(pathlib.Path(p).is_relative_to(pathlib.Path(r))for r in t['read_roots']))
    profile=profile.replace(';TOOL_READ_ROOTS;',roots+' '+libs)
    ancestors=sorted({str(a)for root in [t['clt'],*t['read_roots']]for a in pathlib.Path(root).parents})
    profile=profile.replace(';TOOL_METADATA_ANCESTORS;',' '.join('(literal '+json.dumps(a)+')'for a in ancestors))
    (BASE/'public/worker.sb').write_text(profile);(BASE/'public/worker.sb').chmod(0o444)
    server_profile=profile+'\n(allow network-bind network-inbound (local ip \"localhost:18080\"))\n'
    (BASE/'public/server.sb').write_text(server_profile);(BASE/'public/server.sb').chmod(0o444)
    for p in (BASE/'control/canary',BASE/'private/canary',BASE/'evidence/canary',BASE/'runner/isolation-canary'):
        p.write_bytes(os.urandom(32));p.chmod(0o600)
    os.chown(BASE/'runner/isolation-canary',62001,62001)
    installed={str(p.relative_to(BASE)):sha(p)for d in ('control','public')for p in (BASE/d).iterdir()if p.is_file()}
    c={'schema_version':1,'device':'mac-m5-max','platform':'darwin','production_enabled':False,'runner_name':ACCOUNTS[0][0],'runner_uid':62001,'runner_gid':62001,'worker_name':ACCOUNTS[1][0],'worker_uid':62002,'worker_gid':62002,'personal_home':str(personal),'lock':str(lock),'lock_dev_inode':[ls.st_dev,ls.st_ino],'stop':str(stop),'tools':t,'installed_files':installed,'executor':{'enabled':False,'max_stage_seconds':7200,'memory_budget_bytes':40<<30,'host_reserve_bytes':3<<30,'max_processes':128,'max_artifact_bytes':20<<30,'server_port':18080},'pending':['native compile/Metal and isolation proof','production model clone staging and negative write proof','per-stage native protocol/server loopback positive-negative proof','runner registration and enablement after all gates']}
    path=BASE/'control/config.json';path.write_text(json.dumps(c,sort_keys=True,indent=2)+'\n');path.chmod(0o644)
    # No LaunchAgent/LaunchDaemon/Actions service is created or started here.
    print(json.dumps({'status':'installed-disabled-pending-real-verification','base':str(BASE),'runner_uid':62001,'worker_uid':62002,'memberships':{name:cmd('/usr/bin/id','-G',name)for name,_ in ACCOUNTS},'clt_ancestry':t['clt_ancestry'],'config_sha256':sha(path)}))
if __name__=='__main__':main()
