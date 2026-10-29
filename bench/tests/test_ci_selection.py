"""Trusted selection/dispatch failure tests. No device or measured-performance claim."""

import copy
import io
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import DEVICES, load
from ci import impact, selection_contract as contract, selection_dispatch as dispatch
from ci.selected_aggregate import aggregate
from ci.selected_controller import jobs_checked
from ci.staged_numerical import validate_csv

POLICY = load(Path(__file__).parents[1] / 'ci/policy.json')
IDENTITY = dict(base='a'*40, head='b'*40, candidate='c'*40)


def plan(paths):
    return impact.plan(paths, POLICY, **IDENTITY)


def template(value, device='cloud-5090'):
    needs = impact.device_requirements(value, device)
    sides = {}
    for side in ('base', 'candidate'):
        targets = sorted({impact.CASES[c][0] for c in needs['numerical']})
        sides[side] = dict(revision=value[side], paths=dict(platform='linux', source='/source/'+side,
            build='/scratch/'+side, cwd='/scratch/'+side, frozen='/build/'+side),
            build=dict(cmake='/usr/bin/cmake', cmake_flags=[], targets=['llama-server', *targets],
                       binary_relative='bin/llama-server', image_relatives=['bin/'+t for t in targets]),
            models={m: dict(smoke=dict(model=m, cwd='/scratch'),
                benchmarks={t: dict(cwd='/scratch') for t in needs['performance'].get(m, {}).get('tiers', [])})
                for m in needs['models']})
    return dict(device=device, sides=sides, numerical=dict(backend='CUDA0', cwd='/scratch/candidate'),
                manifest_path='/protected/manifest.json', control_revision='d'*40,
                control_manifest_sha256='e'*64)


def test_shared_unknown_and_ordinary_do_not_default_to_4k():
    for path in ('engine/src/ssd/model-adapter.cpp', 'engine/ggml/src/ggml-ssd/host/arbiter.cpp',
                 'framework/src/inferssd/launch/typed.py', 'unknown-new-build-input'):
        value = plan([path])
        assert len(impact.selected(value)) == 6
        assert all(c['performance']['tiers'] == ['512'] for c in value['cells'])
    value = plan(['engine/src/ssd/model-adapter.cpp'])
    assert all('model-adapter' in c['numerical'] for c in value['cells'])
    value = plan(['engine/ggml/src/ggml-cpu/ssd/runtime-policy.h'])
    assert all('runtime-policy' in c['numerical'] for c in value['cells'])


def test_specific_coverage_and_renames_union():
    cuda = plan(['engine/ggml/src/ggml-cuda/ssd/mxfp4-compute-adapter.cpp'])
    assert {(c['device'], c['model']) for c in impact.selected(cuda)} == {(d, 'deepseek') for d in DEVICES[:2]}
    metal = plan(['engine/ggml/src/ggml-metal/ggml-metal.m'])
    assert {c['device'] for c in impact.selected(metal)} == {'local-mbp'}
    combined = plan(cuda['changed_paths'] + metal['changed_paths'])
    assert len(impact.selected(combined)) == 4
    long = plan(['engine/src/llama-kv-cache-dsv4.cpp'])
    assert all(c['model'] == 'deepseek' and c['performance']['tiers'] == ['512', '4096'] for c in impact.selected(long))


def test_docs_explicit_not_required_and_no_dispatch():
    value = plan(['README.md', 'docs/architecture.svg'])
    assert not impact.selected(value)
    assert jobs_checked({}, value) == {}
    report = aggregate(value, POLICY, Path('/unused'), {}, {})
    assert report['passed'] and len(report['cells']) == 6
    assert {c['status'] for c in report['cells']} == {'not-required'}


@pytest.mark.parametrize('path', ['/absolute', '../escape', 'a/../b', 'a//b', 'a\\b', ''])
def test_changed_paths_are_canonical(path):
    with pytest.raises(ValueError): plan([path])


def test_plan_cannot_drop_cell_format_or_source():
    value = plan(['engine/src/ssd/model-adapter.cpp'])
    for change in ('source', 'cell', 'numerical', 'path'):
        bad = copy.deepcopy(value)
        if change == 'source': bad['candidate'] = 'e'*40
        elif change == 'cell': bad['cells'][0]['required'] = False
        elif change == 'numerical': bad['cells'][0]['numerical'] = ['backend-add']
        else: bad['changed_paths'] = ['README.md']
        # A new coherent source SHA can be planned; it cannot retain the same
        # admitted hash. Cell/path mutation is rejected by the trusted rebuild.
        if change == 'source': assert impact.fingerprint(bad) != impact.fingerprint(value)
        else:
            with pytest.raises(ValueError): impact.validate(bad, POLICY)


def test_root_config_and_manifest_cross_bind_and_exact_stage_allowlist():
    value = plan(['engine/src/ssd/model-adapter.cpp'])
    manifest = dict(job_id='f'*32, source_revisions={s: value[s] for s in ('base', 'candidate')})
    config, manifest = contract.bind(template(value), manifest, value, POLICY)
    config['manifest_sha256'] = '0'*64
    needs = contract.bound(config, manifest, 'f'*32)
    assert needs['models'] == ['deepseek', 'muse']
    assert 'candidate-numerical-model-adapter' in manifest['stages']
    assert {'base-deepseek-r1-benchmark-512', 'candidate-deepseek-r3-benchmark-512'} <= set(manifest['stages'])
    assert 'base-deepseek-r2-benchmark-512' not in manifest['stages']
    assert 'base-deepseek-benchmark-512' not in manifest['stages']
    assert not any('4096' in key for key in manifest['stages'])
    for target in ('config', 'manifest'):
        c, m = copy.deepcopy(config), copy.deepcopy(manifest)
        if target == 'config': c['sides']['candidate']['build']['targets'].append('unrelated')
        else: m['selection']['plan_sha256'] = 'e'*64
        with pytest.raises(ValueError): contract.bound(c, m, 'f'*32)


def test_selected_missing_device_is_failure_not_unselected():
    value = plan(['engine/ggml/src/ggml-metal/ggml-metal.m'])
    report = aggregate(value, POLICY, Path('/no-artifacts'), {}, {})
    assert not report['passed'] and report['errors'][0]['device'] == 'local-mbp'
    assert len([c for c in report['cells'] if c['status'] == 'not-required']) == 4


def test_add_or_wrong_format_cannot_stand_for_selected_math():
    raw = b'backend_name,op_name,op_params,test_mode,supported,passed,error_message\nCUDA0,ADD,type=f32,test,1,1,\n'
    assert validate_csv(raw, 'CUDA0', 'ADD', 'type=f32') == 1
    with pytest.raises(ValueError): validate_csv(raw, 'CUDA0', 'MUL_MAT_ID', 'type_a=q4_K')
    raw = raw.replace(b'ADD,type=f32', b'MUL_MAT_ID,type_a=q5_K')
    with pytest.raises(ValueError): validate_csv(raw, 'CUDA0', 'MUL_MAT_ID', 'type_a=q4_K')
    with pytest.raises(ValueError): validate_csv(raw.replace(b'test,1,1', b'test,0,1'), 'CUDA0')


class API:
    def __init__(self): self.calls = []
    def call(self, path, data=None, method=None):
        self.calls.append((path, data, method))
        if path == 'actions/workflows/integration-smoke.yml':
            return dict(id=123, path=dispatch.WORKFLOW, state='active')
        assert path == 'actions/workflows/123/dispatches'
        return dict(workflow_run_id=456)


def test_real_dispatch_request_uses_only_fixed_workflow_and_exact_plan():
    api = API()
    job = dict(device='local-mbp', job_id='e'*32, source_revision='c'*40,
               config_sha256='f'*64, selection_plan_sha256='a'*64)
    started = dispatch.start(api, job, 'b'*40, '1-1')
    path, body, method = api.calls[-1]
    assert path == 'actions/workflows/123/dispatches' and method == 'POST'
    assert body == dict(ref='main', inputs=dict(job, selection_parent='1-1'), return_run_details=True)
    assert started['id'] == 456
    run = dict(workflow_id=123, path=dispatch.WORKFLOW, event='workflow_dispatch', head_branch='main',
               head_sha='b'*40, run_attempt=1, display_title='integration-'+'e'*32)
    dispatch.run_identity(run, 123, 'b'*40, 'e'*32)
    for key, wrong in [('head_sha', 'd'*40), ('run_attempt', 2), ('path', '.github/workflows/checkpoint.yml')]:
        with pytest.raises(ValueError): dispatch.run_identity(dict(run, **{key: wrong}), 123, 'b'*40, 'e'*32)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../../escape', 'a\\escape'])
def test_artifact_paths_do_not_escape(tmp_path, name):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, 'w') as archive: archive.writestr(name, 'bad')
    with pytest.raises(ValueError): dispatch.unpack(raw.getvalue(), tmp_path / 'out')


def test_missing_admission_still_publishes_terminal_app_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ci import selected_controller as selected
    from test_private_controller import event_context, pr
    event, env = event_context(tmp_path)
    env.update(DEVICE_CHECK_TOKEN='synthetic-app-token', DEVICE_CHECK_APP_ID='88')
    captured = []
    class Source:
        def __init__(self, *args, **kwargs): pass
        def call(self, path, data=None, method=None):
            if path == 'pulls/12': return pr()
            assert path == 'check-runs'
            captured.append(data)
            return dict(app=dict(id=88), head_sha=data['head_sha'], external_id=data['external_id'])
    monkeypatch.setattr(selected, 'API', Source)
    args = SimpleNamespace(policy=Path(__file__).parents[1]/'ci/policy.json', control=tmp_path/'control',
        plan=tmp_path/'absent.json', artifacts=tmp_path/'absent-artifacts', output=tmp_path/'aggregate.json')
    assert selected.finalize(args, env) == 1
    assert captured[0]['status'] == 'completed' and captured[0]['conclusion'] == 'failure'
    assert captured[0]['head_sha'] == 'b'*40
    assert not load(args.output)['passed']


def test_dispatch_no_run_id_is_not_retried():
    class Ambiguous(API):
        def call(self, path, data=None, method=None):
            result = super().call(path, data, method)
            return None if method == 'POST' else result
    api = Ambiguous()
    job = dict(device='local-mbp', job_id='e'*32, source_revision='c'*40,
               config_sha256='f'*64, selection_plan_sha256='a'*64)
    with pytest.raises(ValueError, match='no exact run ID'):
        dispatch.start(api, job, 'b'*40, '1-1')
    assert sum(call[2] == 'POST' for call in api.calls) == 1


def test_wait_failure_or_deadline_cannot_promote_cleanup():
    class Child:
        def __init__(self): self.cancelled = []
        def call(self, path, data=None, method=None):
            if path.endswith('/cancel'):
                self.cancelled.append(path)
                return None
            return dict(workflow_id=123, path=dispatch.WORKFLOW, event='workflow_dispatch', head_branch='main',
                head_sha='b'*40, run_attempt=1, display_title='integration-'+'e'*32, status='in_progress')
    api = Child()
    times = iter([0, 0, 2, 3])
    with pytest.raises(TimeoutError):
        dispatch.wait(api, dict(id=456, workflow_id=123, controller_sha='b'*40, job_id='e'*32),
                      timeout_s=1, clock=lambda: next(times), sleep=lambda _: None)
    assert api.cancelled == ['actions/runs/456/cancel']


@pytest.mark.parametrize('clean', [True, False])
def test_selected_executor_consumes_one_plan_and_one_lease(tmp_path, monkeypatch, clean):
    """Synthetic broker/SSE transport; explicitly no hardware or timing claim."""
    import json
    from types import SimpleNamespace
    from ci import selected_integration as execution
    from ci import selected_preflight
    from test_ci_staged_benchmark import Session as Recorder, CONFIG
    from test_ci_performance_series import record
    from ci.compare import conditions_digest, history_conditions
    value = plan(['engine/ggml/src/ggml-cuda/ssd/mxfp4-compute-adapter.cpp'])
    config = template(value)

    def synthetic(side, folder):
        return dict(model_identity={'synthetic': True}, server=dict(pid=1 if side=='base' else 2, start_identity=str(folder)),
            build=dict(loaded_libraries=[]), host=dict(hardware='synthetic', os='synthetic', memory_budget_bytes=1,
                os_page_cache_treatment='uncontrolled', effective_settings=dict(native_launch_policy=dict(typed_values={}, template_sha256='c' * 64))))

    config.update(selection_plan=value, selection_plan_sha256=impact.fingerprint(value),
                  manifest_sha256='a'*64, paths=config['sides']['candidate']['paths'],
                  control_revision='d'*40, control_manifest_sha256='e'*64, policy=POLICY, baseline_history=[])
    manifest = tmp_path/'manifest.json'
    config['manifest_path'] = str(manifest)
    for side, spec in config['sides'].items():
        spec['models']['deepseek']['smoke'] = dict(CONFIG,
            model='deepseek', argv=['/build/'+side+'/bin/llama-server'])
        spec['models']['deepseek']['benchmarks']['512'] = dict(CONFIG,
            model='deepseek', argv=['/build/'+side+'/bin/llama-server'])
    manifest.write_text(json.dumps(dict(stages=contract.linux_stages(config,
        impact.device_requirements(value, config['device']), ('base', 'candidate')))))
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    sessions = []
    class Session(Recorder):
        closed = False
        job_id = 'f'*32
        def __init__(self, *args, **kwargs): super().__init__(); sessions.append(self)
        def open(self): return dict(manifest_sha256='a'*64)
        def finish(self): return dict(cleanup_confirmed=clean)
        def close(self): self.closed = True
    monkeypatch.setattr(execution, 'StageSession', Session)
    monkeypatch.setattr(execution, 'protected_binding', lambda *args: config)
    monkeypatch.setattr(execution, 'source_manifest', lambda *args: 'd'*40)
    real_digest = execution.digest
    monkeypatch.setattr(execution, 'digest', lambda p: 'e'*64 if Path(p).name == 'control-source.json' else real_digest(p))
    monkeypatch.setattr(execution, 'run_build', lambda session, side, *args, **kwargs:
        dict(binary=dict(runtime_path='/build/'+side+'/bin/llama-server', sha256='0'*64), toolchain={}, actual_server_version='synthetic', images=[]))
    monkeypatch.setattr(execution, 'selected_numerical', lambda session, cfg, build, cases, output:
        {c: dict(status='passed', synthetic=True) for c in cases})
    monkeypatch.setattr(execution, 'smoke', lambda *args, **kwargs:
        dict(status='passed', cleanup_confirmed=True, completion_requests=1))
    monkeypatch.setattr(selected_preflight, 'smoke', execution.smoke)
    def receipt(cfg, mf, side, model, launch, build, folder, *, destination=None):
        result = synthetic(side, folder)
        if destination is not None:
            from ci.contract import save
            save(destination, result)
        return result
    monkeypatch.setattr(execution, 'receipt', receipt)
    monkeypatch.setattr(selected_preflight, 'receipt', receipt)
    def run(history, output):
        config['baseline_history'] = history
        path.write_text(json.dumps(config))
        args = SimpleNamespace(output=tmp_path/output, config=path, config_sha256=real_digest(path),
                               selection_plan_sha256=impact.fingerprint(value), job_id='f'*32, socket='/unused')
        return execution.run(args), load(args.output/'integration.json'), args.output

    # Without history the cell needs review; nothing else changes.
    code, result, output = run([], 'first')
    assert code == 1
    assert result['cells'][0]['performance'][0]['review_reasons'] == ['no-baseline-history']
    # The base commit's baseline record, measured like the in-run base under the same conditions.
    from ci.compare import METRICS
    measured = load(output/'deepseek/512/r1/base/result.json')
    history = [record(value['base'], model='deepseek', base={metric: measured['metrics'][metric] for metric in METRICS},
                      tokens=measured['tokens'],
                      conditions=conditions_digest(history_conditions(synthetic('base', tmp_path))))]
    code, result, output = run(history, 'out')
    assert code == (0 if clean else 1)
    assert result['cleanup_confirmed'] is clean and (result['status']=='passed') is clean
    assert len(sessions) == 2 and sessions[-1].closed
    assert sessions[-1].started == ['candidate-deepseek-r1-benchmark-512', 'base-deepseek-r1-benchmark-512',
                                    'candidate-deepseek-r2-benchmark-512', 'candidate-deepseek-r3-benchmark-512']
    assert sessions[-1].completions == 4
    assert result['selection_plan_sha256'] == impact.fingerprint(value)
    assert result['baseline_history_sha256'] == impact.fingerprint(history)
    assert result['formal_samples'] == 0
