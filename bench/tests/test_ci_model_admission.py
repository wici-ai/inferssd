"""Local admission/transport contracts; no GGUF payload, device or timing run."""

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import model_admission as m
from ci.contract import digest, load, save
from ci.stage_session import Stage


def encode(value):
    return (json.dumps(value, indent=2) + "\n").encode()


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    rows = [dict(device=5, inode=11 + i, bytes=64, mtime_ns=100, ctime_ns=101,
                 uid=1001, gid=1001, mode="0o664", source_path=f"/operator/muse/{i}.gguf",
                 host_path=f"/model-views/muse/{i}.gguf", runtime_path=f"/models/muse/{i}.gguf",
                 sha256=str(i + 3) * 64, historical_sha256=str(i + 3) * 64) for i in range(2)]
    admission = dict(status="full-payload-verified-awaiting-root-readonly-exposure", job_id="1" * 32,
                     models={"muse": dict(shards=rows)}, source_files=2, original_lock=[30, 42])
    proof = dict(status="readonly-host-exposure-pass", root_uid=0, job_id="1" * 32,
                 source_uid_preserved=1001, source_mode_preserved="0o664", payload_copy_or_chown=False,
                 verified_host_paths=[r["host_path"] for r in rows],
                 source_and_view_identities=[dict(source=r["source_path"], host=r["host_path"],
                    identity={k:r[k] for k in (*m.STAT,*m.OWNER)}, view={k:r[k] for k in (*m.STAT,*m.OWNER)},
                    mount_options="ro,nosuid,nodev,noexec,relatime") for r in rows],
                 worker_negative_proof="required separately by operator before and after native stages")
    raw = {}
    config = dict(device="dgx-spark", paths=dict(platform="linux"),
                  model_admission_path="/root-meta/admission.json", model_mount_proof_path="/root-meta/proof.json",
                  prior_model_job_id="1" * 32, manifest_sha256="9" * 64)
    def seal():
        raw[config["model_admission_path"]] = encode(admission)
        config["model_admission_sha256"] = hashlib.sha256(raw[config["model_admission_path"]]).hexdigest()
        proof["admission_sha256"] = config["model_admission_sha256"]
        raw[config["model_mount_proof_path"]] = encode(proof)
        config["model_mount_proof_sha256"] = hashlib.sha256(raw[config["model_mount_proof_path"]]).hexdigest()
    seal()
    shard_identity = [dict(host_path=r["host_path"], runtime_path=r["runtime_path"], sha256=r["sha256"],
                           stat={k:r[k] for k in m.STAT}) for r in rows]
    config["sides"] = {side:dict(paths=dict(cwd="/scratch/"+side, source="/source/"+side),
        models={"muse":dict(model_identity=dict(name="muse", shards=copy.deepcopy(shard_identity)))})
        for side in ("base", "candidate")}
    manifest = dict(job_id="2" * 32, models={"muse":"/model-views/muse"},
                    sources={"base":"/sealed/base", "candidate":"/sealed/candidate"})
    manifest["stages"] = {}
    for when in ("before", "after"):
        spec=m.probe_spec(config,manifest,when)
        manifest["stages"][spec["stage_id"]]=dict(kind=spec["kind"],side="candidate",
            executables=["/usr/bin/python3"],cwd=spec["cwd"],gpu=False)
    def sealed_file(path, **kwargs):
        assert str(path) in raw and kwargs==dict(contents=True,limit=1<<20)
        content=raw[str(path)]
        return dict(sha256=hashlib.sha256(content).hexdigest()),content
    monkeypatch.setattr(m,"sealed_file",sealed_file)
    stats={r["host_path"]:{k:r[k] for k in (*m.STAT,*m.OWNER)} for r in rows}
    monkeypatch.setattr(m,"view_identity",lambda p:copy.deepcopy(stats[p]))
    mounts=[]
    def run(argv, **kwargs):
        assert argv[:3]==["/usr/bin/findmnt","-rn","--mountpoint"]
        assert argv[3] in stats and argv[4:]==["-o","VFS-OPTIONS"]
        mounts.append(argv[3]);return SimpleNamespace(stdout="ro,nosuid,nodev,noexec\n")
    monkeypatch.setattr(m.subprocess,"run",run)
    return SimpleNamespace(config=config,manifest=manifest,admission=admission,proof=proof,seal=seal,
        stats=stats,raw=raw,mounts=mounts,root=tmp_path,monkeypatch=monkeypatch)


def test_existing_root_and_mac_paths_keep_the_original_file_contract(monkeypatch):
    calls=[]
    monkeypatch.setattr(m,"file_identity",lambda p:calls.append(p) or {"inode":1})
    data=dict(model_identity=dict(shards=[dict(host_path="/var/lib/inferssd-ci/models/model.gguf",stat={"inode":1})]))
    for platform in ("linux","darwin"):
        assert m.checked_model({"paths":{"platform":platform}}, {}, "muse", data)==data["model_identity"]
    assert calls==["/var/lib/inferssd-ci/models/model.gguf"]*2
    monkeypatch.setattr(m,"file_identity",lambda p:(_ for _ in ()).throw(ValueError("protected rejects nonroot")))
    with pytest.raises(ValueError,match="protected"):
        m.checked_model({}, {}, "muse", data)


def test_admitted_nonroot_is_stat_and_exact_ro_metadata_only(admitted):
    a=admitted
    admission,_,_=m.documents(a.config)
    values=[m.checked_model(a.config,a.manifest,"muse",a.config["sides"][side]["models"]["muse"],admission)
            for side in ("base","candidate")]
    assert values[0]==values[1]
    assert values[0]["readonly_admission"]==m.binding(a.config)
    assert "worker_pid" not in json.dumps(values[0])
    assert len(a.mounts)==4
    m.prepare(a.config,a.manifest,a.root)
    assert load(a.root/"model-admission/inputs-before.json")["muse"]==values[0]


@pytest.mark.parametrize("field,value",[("prior_model_job_id","wrong"),("model_admission_sha256","bad"),
    ("model_mount_proof_path","relative"),("device","cloud-5090")])
def test_exception_declaration_is_complete_and_spark_only(admitted,field,value):
    a=admitted;a.config[field]=value
    with pytest.raises(ValueError):m.enabled(a.config)


def test_partial_admission_cannot_fall_back_to_the_nonroot_path(admitted):
    del admitted.config["model_mount_proof_sha256"]
    with pytest.raises(ValueError):m.enabled(admitted.config)


@pytest.mark.parametrize("change",["sha","job","rootuid","status","source_uid","copied","paths","source_identity","view_identity","mount","hash","duplicate"])
def test_root_documents_reject_unbound_or_incomplete_proofs(admitted,change):
    a=admitted
    if change=="sha":a.config["model_mount_proof_sha256"]="0"*64
    elif change=="job":a.proof["job_id"]="2"*32
    elif change=="rootuid":a.proof["root_uid"]=False
    elif change=="status":a.proof["status"]="passed"
    elif change=="source_uid":a.proof["source_uid_preserved"]=995
    elif change=="copied":a.proof["payload_copy_or_chown"]=True
    elif change=="paths":a.proof["verified_host_paths"].pop()
    elif change=="source_identity":a.proof["source_and_view_identities"][0]["identity"]["ctime_ns"]+=1
    elif change=="view_identity":a.proof["source_and_view_identities"][0]["view"]["uid"]=0
    elif change=="mount":a.proof["source_and_view_identities"][0]["mount_options"]="rw,noro"
    elif change=="hash":a.admission["models"]["muse"]["shards"][0]["historical_sha256"]="0"*64
    else:a.proof["source_and_view_identities"].append(a.proof["source_and_view_identities"][0])
    if change!="sha":a.seal()
    with pytest.raises((ValueError,KeyError)):m.documents(a.config)


@pytest.mark.parametrize("change",["missing","extra","mapping","runtime","sha","stat","bool_stat","changed_inode","changed_ctime","changed_owner","missing_view","rw_mount"])
def test_selected_current_views_cannot_drift_or_escape(admitted,change):
    a=admitted;data=a.config["sides"]["candidate"]["models"]["muse"]
    if change=="missing":data["model_identity"]["shards"].pop()
    elif change=="extra":data["model_identity"]["shards"].append(data["model_identity"]["shards"][0])
    elif change=="mapping":a.manifest["models"]["muse"]="/other"
    elif change=="runtime":data["model_identity"]["shards"][0]["runtime_path"]="/models/muse/../0.gguf"
    elif change=="sha":data["model_identity"]["shards"][0]["sha256"]="0"*64
    elif change=="stat":data["model_identity"]["shards"][0]["stat"]["mtime_ns"]+=1
    elif change=="bool_stat":data["model_identity"]["shards"][0]["stat"]["device"]=True
    elif change=="changed_inode":a.stats["/model-views/muse/0.gguf"]["inode"]+=1
    elif change=="changed_ctime":a.stats["/model-views/muse/0.gguf"]["ctime_ns"]+=1
    elif change=="changed_owner":a.stats["/model-views/muse/0.gguf"]["uid"]=0
    elif change=="missing_view":a.monkeypatch.setattr(m,"view_identity",lambda p:(_ for _ in ()).throw(FileNotFoundError(p)))
    else:a.monkeypatch.setattr(m.subprocess,"run",lambda *args,**kwargs:SimpleNamespace(stdout="rw,noro\n"))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):m.checked_model(a.config,a.manifest,"muse",data)


def populate_probe(a,when,*,controller_uid=996):
    folder=a.root/"model-admission"/when;folder.mkdir()
    i=0 if when=="before" else 1
    start=dict(stage_id="model-readonly-"+when,handle="handle-"+when,worker_pid=60+i,
               worker_uid=995,birth_identity="boot:"+str(123+i),scope_identity="/scope-"+when)
    spec=m.probe_spec(a.config,a.manifest,when)
    unit=Path(start["scope_identity"]).name
    receipt=dict(start,unit=unit,passed=True,lease=dict(device=30,inode=42),argv=spec["argv"],returncode=0,
                 cleanup_confirmed=True,cleanup=dict(unit=unit,processes_gone=True),
                 guardian=dict(unit=unit,cleanup=dict(unit=unit,processes_gone=True),returncode=0,cleanup_confirmed=True))
    stage=dict(start,running=False,returncode=0,cleanup_confirmed=True,receipt=receipt)
    save(folder/"started.json",start);save(folder/"stage.json",stage)
    save(folder/"stdout.log",m.expected_counts(a.config,a.manifest))
    (folder/"stderr.log").write_bytes(b"")
    save(folder/"probe.json",dict(job_id=a.manifest["job_id"],phase=when,controller_uid=controller_uid,
        manifest_sha256=a.config["manifest_sha256"],started_ns=10+i*10,finished_ns=11+i*10,
        stage_sha256=digest(folder/"stage.json"),started_sha256=digest(folder/"started.json"),stdout_sha256=digest(folder/"stdout.log")))
    return start,stage


def completed(a):
    m.prepare(a.config,a.manifest,a.root)
    selected=m.selected_inputs(a.config,a.manifest,a.admission,live=True)
    save(a.root/"model-admission/inputs-after.json",selected)
    stages={}
    for when in ("before","after"):
        start,stage=populate_probe(a,when);stages[start["stage_id"]]=stage["receipt"]
    base=dict(job_id=a.manifest["job_id"],manifest_sha256=a.config["manifest_sha256"],lease=dict(device=30,inode=42))
    save(a.root/"job-opened.json",dict(base,opened=True))
    save(a.root/"job-finished.json",dict(base,cleanup_confirmed=True,errors=[],stages=stages))
    return m.finalize(a.root,a.config,a.manifest)


def test_final_gate_binds_current_probes_not_the_old_required_string(admitted):
    a=admitted;sha=completed(a)
    assert a.config["prior_model_job_id"]!=a.manifest["job_id"]
    m.retained(a.root,a.config,a.manifest,dict(model_admission_sha256=sha))
    gate=load(a.root/"model-admission/gate.json")
    assert gate["job_id"]==a.manifest["job_id"] and gate["binding"]["prior_model_job_id"]=="1"*32
    assert gate["phases"]["before"]["controller_uid"]==996


@pytest.mark.parametrize("change",["missing_before","missing_after","job","handle","birth","scope","lease","cleanup","uid","counts","after_input","reused_scope","finished_stage","gate_sha"])
def test_aggregate_rebuilds_real_probe_binding_and_rejects_old_or_missing_evidence(admitted,change):
    a=admitted;sha=completed(a);root=a.root/"model-admission"
    if change.startswith("missing_"):(root/change.split("_",1)[1]/"stage.json").unlink()
    elif change=="after_input":
        value=load(root/"inputs-after.json");value["muse"]["shards"].pop();(root/"inputs-after.json").write_bytes(encode(value))
    elif change=="gate_sha":sha="0"*64
    elif change=="finished_stage":
        value=load(a.root/"job-finished.json");value["stages"].pop("model-readonly-after");(a.root/"job-finished.json").write_bytes(encode(value))
    else:
        folder=root/"after"
        if change=="job":
            value=load(folder/"probe.json");value["job_id"]="1"*32;(folder/"probe.json").write_bytes(encode(value))
        elif change=="counts":
            (folder/"stdout.log").write_bytes(encode(dict(marker="MODEL_READONLY_PASS",model_files=22,source_files=2,model_parents=2)))
        else:
            value=load(folder/"stage.json")
            if change=="lease":value["receipt"]["lease"]["inode"]+=1
            elif change=="cleanup":value["receipt"]["cleanup"]["processes_gone"]=False
            elif change=="uid":value["worker_uid"]=1001
            elif change=="reused_scope":value["scope_identity"]="/scope-before"
            else:value[{"handle":"handle","birth":"birth_identity","scope":"scope_identity"}[change]]="old"
            (folder/"stage.json").write_bytes(encode(value))
        # Update the per-probe artifact digest: semantic rejection cannot rely only on a stale hash.
        proof=load(folder/"probe.json")
        for key,name in (("stage","stage.json"),("started","started.json"),("stdout","stdout.log")):
            proof[key+"_sha256"]=digest(folder/name)
        (folder/"probe.json").write_bytes(encode(proof))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):m.retained(a.root,a.config,a.manifest,dict(model_admission_sha256=sha))


def test_probe_keeps_existing_stage_phase_and_exact_selected_counts(admitted):
    a=admitted
    for when,kind,cwd in [("before","build","/scratch/candidate"),("after","test","/scratch")]:
        spec=m.probe_spec(a.config,a.manifest,when)
        assert spec["kind"]==kind and spec["cwd"]==cwd and spec["timeout_s"]==120
        compile(spec["argv"][-1],"<fixed-readonly-probe>","exec")
        assert "os.O_RDWR|os.O_TMPFILE" in spec["argv"][-1]
    assert m.expected_counts(a.config,a.manifest)==dict(marker="MODEL_READONLY_PASS",model_files=2,source_files=2,model_parents=1)


def test_parent_and_leaf_checks_do_not_require_root_owned_payload(monkeypatch):
    class P:
        def __init__(self,value):self.value=value
        @property
        def parents(self):return [SimpleNamespace(lstat=lambda:SimpleNamespace(st_mode=stat.S_IFDIR|0o755,st_uid=0))]
        def lstat(self):return SimpleNamespace(st_mode=stat.S_IFREG|0o664,st_uid=1001,st_gid=1001,
            st_dev=5,st_ino=11,st_size=64,st_mtime_ns=100,st_ctime_ns=101)
    monkeypatch.setattr(m,"Path",P)
    assert m.view_identity("/models/view.gguf")["uid"]==1001


@pytest.mark.parametrize("when",["before","after"])
def test_missing_probe_authorization_refuses_before_any_session(admitted,when):
    a=admitted;del a.manifest["stages"]["model-readonly-"+when]
    with pytest.raises(ValueError,match="authorize"):
        m.prepare(a.config,a.manifest,a.root)
    assert not (a.root/"model-admission").exists()


@pytest.mark.parametrize("fault",[None,"write_allowed","unclean"])
def test_probe_uses_current_session_and_keeps_all_actual_raw_on_refusal(admitted,fault):
    a=admitted;m.prepare(a.config,a.manifest,a.root)
    class Session:
        job_id=a.manifest["job_id"]
        controller_uid=996
        def start(self,**spec):
            assert spec==m.probe_spec(a.config,a.manifest,"before")
            self.stage=Stage(spec["stage_id"],"current-handle",66,995,"boot:7","/scope-current")
            return self.stage
        def wait(self,stage,*,deadline):
            assert stage==self.stage
            row=asdict(stage)
            unit=Path(row["scope_identity"]).name
            receipt=dict(row,unit=unit,passed=fault is None,argv=m.probe_spec(a.config,a.manifest,"before")["argv"],
                lease=dict(device=30,inode=42),returncode=1 if fault=="write_allowed" else 0,
                cleanup_confirmed=fault!="unclean",cleanup=dict(unit=unit,processes_gone=fault!="unclean"),
                guardian=dict(unit=unit,cleanup=dict(unit=unit,processes_gone=fault!="unclean"),
                              returncode=1 if fault=="write_allowed" else 0,cleanup_confirmed=fault!="unclean"))
            return dict(row,returncode=receipt["returncode"],running=False,
                        cleanup_confirmed=receipt["cleanup_confirmed"],receipt=receipt)
    def collect(session,stage,folder):
        save(folder/"stdout.log",m.expected_counts(a.config,a.manifest))
        (folder/"stderr.log").write_text("write unexpectedly allowed" if fault=="write_allowed" else "")
    a.monkeypatch.setattr(m,"collect_log",collect)
    if fault:
        with pytest.raises(ValueError):m.probe(Session(),a.config,a.manifest,"before",a.root)
        assert (a.root/"model-admission/before/stage.json").exists()
        assert not (a.root/"model-admission/before/probe.json").exists()
    else:
        m.probe(Session(),a.config,a.manifest,"before",a.root)
        assert load(a.root/"model-admission/before/probe.json")["job_id"]==a.manifest["job_id"]


@pytest.mark.parametrize("fault",[None,"before","after"])
def test_selected_executor_wires_probes_and_still_finishes_failed_jobs(tmp_path,monkeypatch,fault):
    # Test call ordering only; real proof parsing/transport binding is exercised above.
    from ci import selected_integration as execution
    from test_ci_selected_preflight import setup
    args,config,manifest,needs,events,sessions=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(m,"enabled",lambda cfg:True)
    monkeypatch.setattr(m,"probe_spec",lambda cfg,mf,when:dict(stage_id="model-readonly-"+when,
        kind="build" if when=="before" else "test",argv=["/usr/bin/python3"],cwd="/scratch",timeout_s=120))
    monkeypatch.setattr(m,"prepare",lambda *args:events.append(("model-prepare",)))
    def probe(session,cfg,mf,when,output):
        events.append(("model-probe",when))
        if fault==when:raise ValueError("retained readonly probe refusal")
    monkeypatch.setattr(m,"probe",probe)
    def finalize(*args):
        events.append(("model-finalize",))
        if fault:raise ValueError("missing successful current before/after proof")
        return "b"*64
    monkeypatch.setattr(m,"finalize",finalize)
    assert execution.run(args)==(0 if fault is None else 1)
    opened=events.index(("open",));before=events.index(("model-probe","before"))
    assert events.index(("model-prepare",))<opened<before
    assert events.index(("finish",))<events.index(("model-finalize",))
    if fault=="before":assert not any(e[0] in ("smoke","timed")for e in events)
    else:
        assert before<next(i for i,e in enumerate(events)if e[0]=="smoke")
        assert events.index(("model-probe","after"))>max(i for i,e in enumerate(events)if e[0]=="timed")
    assert load(args.output/"integration.json")["cleanup_confirmed"] is True


def test_probe_counts_are_actual_integer_counts_not_boolean_equivalence(admitted):
    a=admitted;m.prepare(a.config,a.manifest,a.root)
    start,status=populate_probe(a,"before")
    counts=m.expected_counts(a.config,a.manifest);counts["model_parents"]=True
    with pytest.raises(ValueError,match="cover"):
        m.validate_probe(a.config,a.manifest,"before",start,status,encode(counts),
                         dict(device=30,inode=42),996)


def test_selection_allowlist_keeps_before_build_and_after_frozen_test(admitted):
    from ci.selection_contract import linux_stages
    a=admitted
    for side,spec in a.config["sides"].items():
        spec["paths"]["frozen"]="/build/"+side
        spec["build"]=dict(cmake="/usr/bin/cmake",binary_relative="bin/llama-server")
        spec["models"]["muse"].update(smoke=dict(cwd="/scratch"),benchmarks={})
    stages=linux_stages(a.config,dict(models=["muse"],numerical=[]),("base","candidate"))
    for name in ("model-readonly-before","model-readonly-after"):
        assert stages[name]==a.manifest["stages"][name]
    assert stages["model-readonly-before"]["kind"]=="build"
    assert stages["model-readonly-after"]["kind"]=="test"


@pytest.mark.parametrize("raw",[b'{"job_id":1,"job_id":2}',b'{"value":NaN}'])
def test_proof_json_rejects_ambiguous_or_nonfinite_values(raw):
    with pytest.raises(ValueError):m.parsed(raw)


@pytest.mark.parametrize("keys",[("unit",),("cleanup","unit"),("guardian","unit"),("guardian","cleanup","unit")])
@pytest.mark.parametrize("fault",["old-unit","missing"])
def test_guardian_and_cleanup_units_are_the_current_scope_not_a_stale_receipt(admitted,keys,fault):
    a=admitted;m.prepare(a.config,a.manifest,a.root)
    started,status=populate_probe(a,"before")
    record=status["receipt"]
    for key in keys[:-1]:record=record[key]
    if fault=="missing":del record[keys[-1]]
    else:record[keys[-1]]="inferssd-stage-old-job.service"
    with pytest.raises(ValueError,match="unit differs"):
        m.validate_probe(a.config,a.manifest,"before",started,status,
            encode(m.expected_counts(a.config,a.manifest)),dict(device=30,inode=42),996)


@pytest.mark.parametrize("field,value",[("passed",False),("passed","true"),("passed",None),
    ("error","InterruptedError: stage stopped by trusted controller"),("error",None),
    ("cleanup_error",""),("cleanup_error",None)])
def test_zero_exit_guardian_does_not_override_a_failed_probe_producer(admitted,field,value):
    a=admitted;m.prepare(a.config,a.manifest,a.root)
    started,status=populate_probe(a,"before")
    if field=="passed" and value is None:del status["receipt"]["passed"]
    else:status["receipt"][field]=value
    assert status["returncode"]==status["receipt"]["guardian"]["returncode"]==0
    with pytest.raises(ValueError,match="producer recorded failure"):
        m.validate_probe(a.config,a.manifest,"before",started,status,
            encode(m.expected_counts(a.config,a.manifest)),dict(device=30,inode=42),996)


def test_final_gate_rejects_coherent_current_job_receipt_with_cancel_race(admitted):
    a=admitted;completed(a);folder=a.root/"model-admission/after"
    stage=load(folder/"stage.json");stage["receipt"].update(passed=False,error="InterruptedError: stage stopped by trusted controller")
    (folder/"stage.json").write_bytes(encode(stage))
    finished=load(a.root/"job-finished.json");finished["stages"]["model-readonly-after"]=stage["receipt"]
    (a.root/"job-finished.json").write_bytes(encode(finished))
    proof=load(folder/"probe.json");proof["stage_sha256"]=digest(folder/"stage.json")
    (folder/"probe.json").write_bytes(encode(proof))
    assert stage["receipt"]["guardian"]["returncode"]==0 and finished["cleanup_confirmed"] is True
    with pytest.raises(ValueError,match="producer recorded failure"):m.gate(a.root,a.config,a.manifest)
