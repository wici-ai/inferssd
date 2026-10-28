#!/usr/bin/env python3
"""Read-only checks of the five Linux settings required by InferSSD profiles."""

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys


@dataclass
class Result:
    name: str
    status: str
    details: list[str] = field(default_factory=list)


def command(arguments):
    return subprocess.check_output(arguments, text=True, stderr=subprocess.PIPE, timeout=15).strip()


def read(path):
    return path.read_text().strip()


def cpu(sys_root):
    policies = sorted((sys_root / 'devices/system/cpu/cpufreq').glob('policy*'))
    if not policies:
        return Result('CPU performance', 'UNKNOWN', ['No cpufreq policies are readable.'])
    details = []
    good = True
    for policy in policies:
        governor = read(policy / 'scaling_governor')
        epp_path = policy / 'energy_performance_preference'
        epp = read(epp_path) if epp_path.exists() else None
        details.append(f'{policy.name}: governor={governor}' + (f', EPP={epp}' if epp is not None else ''))
        good = good and governor == 'performance' and epp in (None, 'performance')
    return Result('CPU performance', 'PASS' if good else 'FAIL', details)


def nvme(sys_root, controllers):
    root = sys_root / 'class/nvme'
    devices = [root / name for name in controllers] if controllers else sorted(root.glob('nvme*'))
    if not devices:
        return Result('NVMe latency QoS = 0', 'UNKNOWN', ['No NVMe controllers found.'])
    good = True
    details = []
    for device in devices:
        value = read(device / 'power/pm_qos_latency_tolerance_us')
        model = read(device / 'model')
        details.append(f'{device.name} ({model}): {value} us')
        good = good and value == '0'
    return Result('NVMe latency QoS = 0', 'PASS' if good else 'FAIL', details)


def aspm(sys_root):
    policy = read(sys_root / 'module/pcie_aspm/parameters/policy')
    good = '[performance]' in policy.split()
    return Result('PCIe ASPM performance', 'PASS' if good else 'FAIL', [policy])


def iommu(sys_root, proc_root):
    tokens = read(proc_root / 'cmdline').split()
    groups_root = sys_root / 'kernel/iommu_groups'
    if not groups_root.is_dir():
        return Result('IOMMU disabled', 'UNKNOWN', ['IOMMU group directory is unavailable.'])
    groups = [path for path in groups_root.iterdir() if path.is_dir()]
    options = [token for token in tokens if token.startswith(('iommu=', 'intel_iommu=', 'amd_iommu='))]
    details = ['boot options: ' + (' '.join(options) or '(none)'), f'IOMMU groups: {len(groups)}']
    if groups:
        return Result('IOMMU disabled', 'FAIL', details)
    # Empty groups alone can also mean that no device was attached to an IOMMU.
    values = {option.partition('=')[0]: option.partition('=')[2] for option in options}
    off = values.get('iommu') == 'off' or values.get('intel_iommu') == 'off' or values.get('amd_iommu') == 'off'
    conflicting = any(value in ('on', 'force') for value in values.values())
    if off and not conflicting:
        return Result('IOMMU disabled', 'PASS', details)
    details.append('No unambiguous IOMMU-off boot option; zero groups alone is insufficient.')
    return Result('IOMMU disabled', 'UNKNOWN', details)


def pci_id(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2}\.[0-7])', value)
    if not match:
        return value.lower()
    return f'{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}'


def gpu(run, device, target, services):
    fields = 'uuid,pci.bus_id,name,clocks.current.graphics,clocks.current.sm,persistence_mode'
    output = run(['nvidia-smi', '-i', device, '--query-gpu=' + fields,
                  '--format=csv,noheader,nounits'])
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 6:
        return Result('GPU clock lock', 'UNKNOWN', ['Expected exactly one GPU; select it with --gpu.'])
    uuid, bus, name, graphics, sm, persistence = [value.strip() for value in rows[0]]
    details = [f'{name} ({uuid}, {bus})', f'target={target} MHz; actual graphics={graphics} MHz, SM={sm} MHz',
               f'persistence mode={persistence}']
    lock_pattern = re.compile(r'GPU clocks set to\s+"?\(gpuClkMin\s+(\d+),\s*gpuClkMax\s+(\d+)\)"?\s+for GPU\s+(\S+)')
    for service in services:
        try:
            properties = run(['systemctl', 'show', service, '-p', 'ActiveState', '-p', 'ExecMainStatus',
                              '-p', 'UnitFileState', '-p', 'InvocationID'])
            values = dict(line.split('=', 1) for line in properties.splitlines() if '=' in line)
            if values.get('ActiveState') != 'active' or values.get('ExecMainStatus') != '0':
                continue
            invocation = values.get('InvocationID', '')
            if not re.fullmatch(r'[0-9a-fA-F]{32}', invocation):
                continue
            journal = run(['journalctl', '-b', '-u', service,
                           '_SYSTEMD_INVOCATION_ID=' + invocation, '--no-pager', '-o', 'cat', '--quiet'])
            acknowledgments = [match for match in lock_pattern.finditer(journal)
                               if pci_id(match[3].rstrip('.')) == pci_id(bus)]
            if not acknowledgments:
                continue
            latest = acknowledgments[-1]
            minimum, maximum = int(latest[1]), int(latest[2])
            details.append(f'{service}: current boot invocation accepted {minimum},{maximum} MHz; '
                           f'boot enablement={values.get("UnitFileState", "unknown")}')
            if (minimum, maximum) != (target, target):
                return Result('GPU clock lock', 'FAIL', details)
            if graphics != str(target) or sm != str(target):
                details.append('The requested lock was applied, but the observed clock differs. '
                               'Do not report the GPU as actually running at the target.')
                return Result('GPU clock lock', 'WARN', details)
            if persistence != 'Enabled':
                details.append('Persistence mode is disabled; verify the lock survives GPU deinitialization.')
                return Result('GPU clock lock', 'WARN', details)
            return Result('GPU clock lock', 'PASS', details)
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    details.append('No successful current-boot clock-lock service record for this GPU. '
                   'Current/max clocks alone cannot prove a lock target. '
                   'Run with sudo to read the journal, or specify --gpu-clock-service.')
    return Result('GPU clock lock', 'UNKNOWN', details)


def collect(sys_root, proc_root, run, controllers, device, target, services):
    checks = [('CPU performance', lambda: cpu(sys_root)),
              ('NVMe latency QoS = 0', lambda: nvme(sys_root, controllers)),
              ('PCIe ASPM performance', lambda: aspm(sys_root)),
              ('GPU clock lock', lambda: gpu(run, device, target, services)),
              ('IOMMU disabled', lambda: iommu(sys_root, proc_root))]
    results = []
    for name, check in checks:
        try:
            results.append(check())
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            results.append(Result(name, 'UNKNOWN', [str(exc)]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvme', action='append', metavar='nvme0', help='Controller to check; default: all. May be repeated.')
    parser.add_argument('--gpu', default='0', help='NVIDIA GPU index, UUID or PCI bus ID; default: 0.')
    parser.add_argument('--gpu-clock', type=int, default=3090, metavar='MHZ')
    parser.add_argument('--gpu-clock-service', action='append', metavar='UNIT', help='Root systemd clock-lock service to inspect.')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('This script checks a Linux host; run it on the inference machine.')
    if args.gpu_clock <= 0 or (args.nvme and any(not re.fullmatch(r'nvme\d+', value) for value in args.nvme)):
        parser.error('--gpu-clock must be positive and --nvme must name a controller such as nvme0.')
    services = args.gpu_clock_service or ['nvidia-gpu-clock-3090.service', 'ssd-llama-performance.service']
    results = collect(Path('/sys'), Path('/proc'), command, args.nvme, args.gpu, args.gpu_clock, services)
    good = all(result.status == 'PASS' for result in results)
    if args.json:
        print(json.dumps({'all_passed': good, 'checks': [asdict(result) for result in results]}, indent=2))
    else:
        print('InferSSD performance preflight (read-only)')
        for result in results:
            print(f'[{result.status}] {result.name}')
            for detail in result.details:
                print('  ' + detail)
        print('All five checks passed.' if good else 'Review every FAIL, WARN or UNKNOWN before launching or benchmarking.')
    return 0 if good else 1


if __name__ == '__main__':
    raise SystemExit(main())
