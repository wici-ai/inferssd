"""Retained Mac host gates over synthetic command output; no host/model runs."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci import mac_host_gate as gate
from ci import mac_host_state as state


CONFIG = dict(host=dict(hardware=dict(memory_total_bytes=48 << 30), resources=dict(memory_budget_bytes=40 << 30)))


def snapshot(second, *, free=38 << 30, swapouts=100, compressions=100):
    page = 16384
    vm = f'Mach Virtual Memory Statistics: (page size of {page} bytes)\nPages free: {free // page}.\nPages inactive: 0.\n'
    vm += ''.join(f'{name}: {value}.\n' for name, value in (
        ('Swapins', 100), ('Swapouts', swapouts), ('Compressions', compressions),
        ('Decompressions', 100), ('Pageins', 100), ('Pageouts', 100)))
    text = dict(vm_stat=vm, swap='total = 2048.00M used = 1024.00M free = 1024.00M (encrypted)\n',
        memory_bytes=str(48 << 30), memory_pressure_level='1\n',
        processes='1 0 0 0:01.00 01:00 1024 launchd\n3 1 62001 0:00.00 00:10 1024 python\n10 1 501 0:00.00 00:10 1024 service\n')
    return dict(schema_version=1, platform='darwin', status='passed', observer_pid=3,
        started_monotonic_ns=int(second * 1e9), finished_monotonic_ns=int(second * 1e9) + 10000000,
        raw={name:dict(returncode=0, error=None, stdout=raw, stderr='', elapsed_s=.001) for name, raw in text.items()},
        errors=[], unavailable=[], vm_stat=state.parse_vm_stat(vm), swap=state.parse_swap(text['swap']),
        memory_bytes=48 << 30, memory_pressure_level=1, processes=state.parse_processes(text['processes']))


def arm(folder, monkeypatch, *, free=38 << 30, swapouts=100, compressions=100):
    folder.mkdir()
    expected = gate.policy(CONFIG)
    before, after = snapshot(1, free=free), snapshot(3.1, free=free)
    check = dict(before=before, after=after, assessment=state.compare_snapshots(before, after, expected['limits']))
    monkeypatch.setattr(state, 'quiet_check', lambda *a, **kw: copy.deepcopy(check))
    gate.before_launch(folder, expected)
    for name, value in zip(gate.SNAPSHOTS, (snapshot(4, free=free),
            snapshot(6, free=free, swapouts=swapouts, compressions=compressions),
            snapshot(7, free=free, swapouts=swapouts, compressions=compressions))):
        monkeypatch.setattr(state, 'snapshot', lambda value=value: copy.deepcopy(value))
        gate.capture(folder, name)
    return gate.retained_arm(folder, expected)


def test_policy_uses_sealed_ram_and_fixed_thresholds():
    value = gate.policy(CONFIG)
    assert value['limits']['min_free_inactive_bytes'] == 36 << 30
    assert value['limits']['max_non_ci_process_rss_bytes'] == 512 << 20
    assert value['limits']['max_non_ci_process_cpu_cores'] == .25
    assert value['limits']['max_non_ci_cpu_cores'] == .5
    assert value['limits']['max_compression_delta_bytes'] == 64 << 20
    for ram in (True, 0, '48', 30 << 30):
        changed = copy.deepcopy(CONFIG); changed['host']['hardware']['memory_total_bytes'] = ram
        with pytest.raises(ValueError): gate.policy(changed)


def test_valid_retention_accepts_static_swap_and_compression_only(tmp_path, monkeypatch):
    summary = arm(tmp_path / 'arm', monkeypatch, compressions=100 + (128 << 20) // 16384)
    assert summary['quiet']['status'] == summary['timed']['status'] == 'passed'
    assert summary['timed']['compression_delta_bytes'] == 128 << 20
    assert set(summary['artifacts']) == {gate.QUIET, *gate.SNAPSHOTS}
    assert gate.retained_quiet(tmp_path / 'arm', gate.policy(CONFIG))['quiet']['status'] == 'passed'


@pytest.mark.parametrize('fault', ['missing', 'symlink', 'raw', 'parsed', 'quiet-result', 'quiet-policy', 'order', 'ram'])
def test_retained_missing_or_changed_evidence_is_refused(tmp_path, monkeypatch, fault):
    folder = tmp_path / 'arm'; arm(folder, monkeypatch)
    path = folder / (gate.QUIET if fault.startswith('quiet') else gate.SNAPSHOTS[1])
    if fault == 'missing': path.unlink()
    elif fault == 'symlink':
        saved = folder / 'other.json'; path.rename(saved); path.symlink_to(saved)
    else:
        value = json.loads(path.read_text())
        if fault == 'raw': value['raw']['swap']['stdout'] = 'invalid'
        elif fault == 'parsed': value['swap']['used_bytes'] += 1
        elif fault == 'quiet-result': value['assessment']['elapsed_s'] += 1
        elif fault == 'quiet-policy': value['policy']['limits']['max_swap_delta_bytes'] += 1
        elif fault == 'order': value['started_monotonic_ns'] = 1
        elif fault == 'ram':
            value['memory_bytes'] = 64 << 30; value['raw']['memory_bytes']['stdout'] = str(64 << 30)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError): gate.retained_arm(folder, gate.policy(CONFIG))


def test_failed_quiet_or_capture_retains_raw_evidence(tmp_path, monkeypatch):
    expected = gate.policy(CONFIG)
    before, after = snapshot(1), snapshot(3.1, free=20 << 30)
    monkeypatch.setattr(state, 'quiet_check', lambda *a, **kw: dict(before=before, after=after,
        assessment=state.compare_snapshots(before, after, expected['limits'])))
    with pytest.raises(ValueError, match='precondition'): gate.before_launch(tmp_path, expected)
    assert (tmp_path / gate.QUIET).is_file()
    failed = snapshot(4); failed['status'] = 'failed'; failed['errors'] = ['timeout']
    monkeypatch.setattr(state, 'snapshot', lambda: failed)
    with pytest.raises(ValueError): gate.capture(tmp_path, gate.SNAPSHOTS[0])
    assert json.loads((tmp_path / gate.SNAPSHOTS[0]).read_text())['errors'] == ['timeout']


@pytest.mark.parametrize('reason', ['swap', 'free', 'prior-slowdown'])
def test_pair_demotes_without_erasing_metrics_or_rerunning(tmp_path, monkeypatch, reason):
    base = arm(tmp_path / 'base', monkeypatch)
    candidate = arm(tmp_path / 'candidate', monkeypatch, free=36 << 30 if reason == 'free' else 38 << 30,
        swapouts=101 if reason == 'swap' else 100)
    compared = dict(status='passed', passed=True, metrics={'decode_first128_tps':{'base':1, 'candidate':2}},
        observed_slowdowns=[], diagnostic_required=False)
    if reason == 'prior-slowdown':
        compared.update(status='needs-review', passed=False, observed_slowdowns=['decode_first128_tps'])
    original = copy.deepcopy(compared)
    result = gate.pair(compared, base, candidate)
    assert result['status'] == 'needs-review' and result['passed'] is False
    assert result['metrics'] == original['metrics'] and result['observed_slowdowns'] == original['observed_slowdowns']
    assert compared == original
    if reason == 'prior-slowdown': assert result['environment']['status'] == 'passed'
    else: assert result['environment']['status'] == 'needs-review'


def test_pair_refuses_a_retired_status_name(tmp_path, monkeypatch):
    base = arm(tmp_path / 'base', monkeypatch)
    candidate = arm(tmp_path / 'candidate', monkeypatch, free=38 << 30, swapouts=100)
    with pytest.raises(ValueError):
        gate.pair(dict(status='inconclusive', passed=False, metrics={}), base, candidate)
