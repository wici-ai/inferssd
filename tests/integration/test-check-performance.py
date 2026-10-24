#!/usr/bin/env python3
"""Check that missing or contradictory tuning evidence never passes preflight."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('performance_preflight',
       Path(__file__).resolve().parents[2] / 'scripts/check-performance.py')
preflight = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = preflight
spec.loader.exec_module(preflight)


class CheckPerformance(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.sys_root = self.root/'sys'
        self.proc_root = self.root/'proc'
        self.write('sys/devices/system/cpu/cpufreq/policy0/scaling_governor', 'performance')
        self.write('sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference', 'performance')
        self.write('sys/class/nvme/nvme0/power/pm_qos_latency_tolerance_us', '0')
        self.write('sys/class/nvme/nvme0/model', 'Test SSD')
        self.write('sys/module/pcie_aspm/parameters/policy', 'default [performance] powersave')
        self.write('proc/cmdline', 'quiet intel_iommu=off iommu=off')
        (self.sys_root/'kernel/iommu_groups').mkdir(parents=True)
        self.graphics = '3090'
        self.bus = '00000000:02:00.0'
        self.service_status = 'active'
        self.journal = 'GPU clocks set to "(gpuClkMin 3090, gpuClkMax 3090)" for GPU 00000000:02:00.0'

    def write(self, name, content):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def run_command(self, arguments):
        if arguments[0] == 'nvidia-smi':
            # A high clock alone must not establish that a lock is configured.
            return f'GPU-test, {self.bus}, Test GPU, {self.graphics}, {self.graphics}, Enabled'
        if arguments[0] == 'systemctl':
            return f'ActiveState={self.service_status}\nExecMainStatus=0\nUnitFileState=enabled\nInvocationID='+'a'*32
        if arguments[0] == 'journalctl':
            return self.journal
        raise AssertionError(arguments)

    def results(self):
        return preflight.collect(self.sys_root, self.proc_root, self.run_command,
                                 None, '0', 3090, ['test-clock.service'])

    def test_all_five_settings_confirmed(self):
        self.assertTrue(all(result.status == 'PASS' for result in self.results()))

    def test_one_cpu_in_powersave_fails(self):
        self.write('sys/devices/system/cpu/cpufreq/policy1/scaling_governor', 'powersave')
        self.assertEqual(self.results()[0].status, 'FAIL')

    def test_balanced_epp_fails_even_with_performance_governor(self):
        self.write('sys/devices/system/cpu/cpufreq/policy0/energy_performance_preference', 'balance_performance')
        self.assertEqual(self.results()[0].status, 'FAIL')

    def test_nonzero_qos_on_second_controller_fails(self):
        self.write('sys/class/nvme/nvme1/power/pm_qos_latency_tolerance_us', '100000')
        self.write('sys/class/nvme/nvme1/model', 'Second SSD')
        self.assertEqual(self.results()[1].status, 'FAIL')

    def test_missing_sysfs_attribute_is_unknown(self):
        (self.sys_root/'class/nvme/nvme0/power/pm_qos_latency_tolerance_us').unlink()
        self.assertEqual(self.results()[1].status, 'UNKNOWN')

    def test_available_performance_aspm_is_not_active(self):
        self.write('sys/module/pcie_aspm/parameters/policy', '[default] performance powersave')
        self.assertEqual(self.results()[2].status, 'FAIL')

    def test_current_3090_without_lock_record_is_unknown(self):
        self.journal = ''
        self.assertEqual(self.results()[3].status, 'UNKNOWN')

    def test_lock_record_for_a_different_gpu_is_unknown(self):
        self.bus = '00000000:03:00.0'
        self.assertEqual(self.results()[3].status, 'UNKNOWN')

    def test_inactive_service_is_not_proof_of_lock(self):
        self.service_status = 'inactive'
        self.assertEqual(self.results()[3].status, 'UNKNOWN')

    def test_3090_target_with_actual_2790_is_warning(self):
        self.graphics = '2790'
        result = self.results()[3]
        self.assertEqual(result.status, 'WARN')
        self.assertIn('actual graphics=2790', ' '.join(result.details))

    def test_later_different_target_does_not_reuse_old_success(self):
        self.journal += '\nGPU clocks set to "(gpuClkMin 2700, gpuClkMax 2700)" for GPU 00000000:02:00.0'
        self.assertEqual(self.results()[3].status, 'FAIL')

    def test_iommu_off_option_with_active_groups_fails(self):
        (self.sys_root/'kernel/iommu_groups/0').mkdir()
        self.assertEqual(self.results()[4].status, 'FAIL')

    def test_no_groups_without_off_option_is_unknown(self):
        self.write('proc/cmdline', 'quiet')
        self.assertEqual(self.results()[4].status, 'UNKNOWN')

    def test_conflicting_iommu_options_do_not_pass(self):
        self.write('proc/cmdline', 'iommu=off intel_iommu=on')
        self.assertEqual(self.results()[4].status, 'UNKNOWN')


if __name__ == '__main__':
    unittest.main()
