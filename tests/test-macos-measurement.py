#!/usr/bin/env python3
"""Protocol and cleanup regression tests; no weights, disk benchmark or GPU required."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench/macos'))
spec = importlib.util.spec_from_file_location('measure', ROOT / 'bench/macos/measure.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def probability(token):
    top = [{'id': i, 'token': str(i), 'bytes': [48 + i], 'logprob': -float(i)} for i in range(1, 6)]
    return {'id': token, 'token': str(token), 'bytes': [48 + token], 'logprob': -float(token), 'top_logprobs': top}


def final(n=2):
    return {'stop': True, 'stop_type': 'limit', 'truncated': False, 'tokens_predicted': n,
            'tokens': [], 'timings': {'prompt_n': 1, 'prompt_ms': 1.0, 'prompt_per_second': 1000.0, 'predicted_n': n, 'predicted_ms': 10.0, 'predicted_per_second': n * 100.0}}


def events(n=2):
    return [{'tokens': [i], 'completion_probabilities': [probability(i)], 'stop': False} for i in range(1, n + 1)] + [final(n)]


def collect(values, expected=2):
    c = m.Completion(expected)
    for event in values:
        c.add(event)
    return c.validate()


class ProtocolTests(unittest.TestCase):
    def test_full_top5_passes(self):
        self.assertEqual(collect(events()).tokens, [1, 2])

    def test_missing_stop(self):
        with self.assertRaises(ValueError):
            collect(events()[:-1])

    def test_truncated_or_wrong_stop(self):
        for key, value in [('truncated', True), ('stop_type', 'eos'), ('tokens_predicted', 1)]:
            with self.subTest(key=key):
                rows = events(); rows[-1][key] = value
                with self.assertRaises(ValueError): collect(rows)

    def test_missing_probability(self):
        rows = events(); rows[0].pop('completion_probabilities')
        with self.assertRaises(ValueError): collect(rows)

    def test_invalid_top5(self):
        for mutation in ('short', 'nan', 'positive', 'id', 'bytes', 'duplicate'):
            with self.subTest(mutation=mutation):
                rows = events(); p = rows[0]['completion_probabilities'][0]
                if mutation == 'short': p['top_logprobs'].pop()
                if mutation == 'nan': p['top_logprobs'][0]['logprob'] = float('nan')
                if mutation == 'positive': p['logprob'] = 0.5
                if mutation == 'id': p['id'] = 7
                if mutation == 'bytes': p['bytes'] = [256]
                if mutation == 'duplicate': p['top_logprobs'][1]['id'] = 1
                with self.assertRaises(ValueError): collect(rows)

    def test_invalid_or_missing_final_timings(self):
        for key, value in [('timings', None), ('prompt_n', -1), ('predicted_n', 1), ('predicted_ms', float('nan')), ('prompt_ms', -1), ('predicted_per_second', None)]:
            with self.subTest(key=key):
                rows = events()
                if key == 'timings': rows[-1]['timings'] = value
                else: rows[-1]['timings'][key] = value
                with self.assertRaises(ValueError): collect(rows)

    def test_data_after_stop(self):
        with self.assertRaises(ValueError): collect(events() + [events()[0]])

    def test_total_deadline_and_handler_restoration(self):
        handler = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with m.request_deadline(0.02):
                time.sleep(0.2)
        self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))


class FakeServer:
    pid = 12345
    def __init__(self, fail_shutdown=False):
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.fail_shutdown = fail_shutdown
    def poll(self): return self.returncode
    def terminate(self): self.terminated = True
    def kill(self): self.killed = True; self.returncode = -9
    def wait(self, timeout=None):
        if self.fail_shutdown and not self.killed:
            raise subprocess.TimeoutExpired('server', timeout)
        self.returncode = -9 if self.killed else 0
        return self.returncode


class CleanupTests(unittest.TestCase):
    def test_resume_failure_does_not_skip_server(self):
        paused = mock.Mock(identity={'pid': 1})
        paused.resume.side_effect = subprocess.TimeoutExpired('ps', 2)
        server = FakeServer()
        result = m.clean_resources(paused, server, [], threading.Event(), None, None)
        self.assertTrue(server.terminated)
        self.assertEqual(paused.resume.call_count, 2)
        self.assertTrue(result['errors'])

    def test_forced_kill_is_failure(self):
        paused = mock.Mock(identity=None)
        server = FakeServer(fail_shutdown=True)
        result = m.clean_resources(paused, server, [], threading.Event(), None, None)
        self.assertTrue(server.killed)
        self.assertTrue(result['errors'])

    def test_prior_stopped_process_is_not_resumed(self):
        identity = {'pid': 99, 'uid': os.getuid(), 'state': 'T', 'start': 'test', 'command': '/bin/screenpipe'}
        paused = m.PausedScreenpipe()
        with mock.patch.object(m, 'process_identity', return_value=identity), mock.patch.object(m.os, 'kill') as kill:
            paused.pause(99); paused.resume()
            kill.assert_not_called()

    def test_stop_responsibility_published_before_signal(self):
        identity = {'pid': 99, 'uid': os.getuid(), 'state': 'S', 'start': 'test', 'command': '/bin/screenpipe'}
        paused = m.PausedScreenpipe()
        def kill(pid, sig):
            self.assertIsNotNone(paused.identity)
        with mock.patch.object(m, 'process_identity', return_value=identity), mock.patch.object(m.os, 'kill', side_effect=kill):
            paused.pause(99); paused.resume()
        self.assertIsNone(paused.identity)

    def test_reused_pid_is_not_signaled(self):
        old = {'pid': 99, 'uid': os.getuid(), 'state': 'T', 'start': 'old', 'command': '/bin/screenpipe'}
        paused = m.PausedScreenpipe(); paused.identity = old
        with mock.patch.object(m, 'process_identity', return_value=dict(old, start='new')), mock.patch.object(m.os, 'kill') as kill:
            with self.assertRaises(RuntimeError): paused.resume()
            kill.assert_not_called()

    @unittest.skipUnless(os.name == 'posix', 'requires POSIX process signals')
    def test_real_owned_process_is_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / 'screenpipe'
            compiler = shutil.which('cc')
            if compiler is None: self.skipTest('requires a C compiler for an owned signal fixture')
            subprocess.run([compiler, '-x', 'c', '-', '-o', str(executable)], input='#include <stdio.h>\n#include <unistd.h>\nint main(void){puts("ready");fflush(stdout);sleep(30);return 0;}\n', text=True, check=True, capture_output=True)
            proc = subprocess.Popen([str(executable)], stdout=subprocess.PIPE, text=True)
            self.assertEqual(proc.stdout.readline().strip(), 'ready')
            paused = m.PausedScreenpipe()
            try:
                paused.pause(proc.pid)
                deadline = time.monotonic() + 2
                while 'T' not in m.process_identity(proc.pid)['state'] and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIn('T', m.process_identity(proc.pid)['state'])
                paused.resume()
                deadline = time.monotonic() + 2
                while 'T' in m.process_identity(proc.pid)['state'] and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertNotIn('T', m.process_identity(proc.pid)['state'])
            finally:
                proc.kill(); proc.wait(timeout=5)
                proc.stdout.close()

    def test_pending_signal_delivered_after_cleanup_region(self):
        guard = m.TerminationSignals()
        old = signal.signal(signal.SIGTERM, guard.handle)
        try:
            with self.assertRaises(InterruptedError):
                with guard.defer():
                    os.kill(os.getpid(), signal.SIGTERM)
                    self.assertEqual(guard.pending, [signal.SIGTERM])
                    self.assertEqual(guard.depth, 1)
            self.assertEqual(guard.depth, 0)
        finally:
            signal.signal(signal.SIGTERM, old)

    def test_spawned_child_has_no_inherited_blocked_termination_mask(self):
        guard = m.TerminationSignals()
        with guard.defer():
            proc = subprocess.Popen([sys.executable, '-u', '-c',
                'import signal,time;print(sorted(map(int,signal.pthread_sigmask(signal.SIG_BLOCK,[]))),flush=True);time.sleep(30)'], stdout=subprocess.PIPE, text=True)
        try:
            mask = json.loads(proc.stdout.readline())
            self.assertFalse(set(mask) & {signal.SIGTERM, signal.SIGINT, signal.SIGHUP})
            proc.terminate()
            self.assertEqual(proc.wait(timeout=2), -signal.SIGTERM)
        finally:
            if proc.poll() is None: proc.kill(); proc.wait(timeout=5)
            proc.stdout.close()

    def test_deadline_signal_is_deferred_until_child_responsibility_is_published(self):
        guard = m.TerminationSignals()
        proc = None
        try:
            with self.assertRaises(TimeoutError):
                with m.request_deadline(0.01, guard):
                    with guard.defer():
                        proc = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
                        time.sleep(0.03)
                        self.assertIsNotNone(proc)
            self.assertIn(signal.SIGALRM, guard.pending)
        finally:
            if proc is not None: proc.terminate(); proc.wait(timeout=2)


class TraceTests(unittest.TestCase):
    def test_complete_trace_counts_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                '': 'blk.0 0\n',
                '.requests.tsv': 'tensor\texpert\tlogical_bytes\tbegin_ns\thit\tcache_updates\nblk.0\t0\t16\t100\t0\t1\n',
                '.reads.tsv': 'source_path\toffset\tread_bytes\tlogical_bytes\tbegin_ns\tduration_ns\n/source\t0\t16\t16\t100\t20\n',
                '.phases.tsv': 'end_ns\trequests\tacquire_us\tread_wait_us\tcompute_us\tbarrier_us\tlookup_us\tpolicy_us\tallocation_us\trelease_us\tlock_wait_us\n200\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\n',
            }
            for suffix, data in files.items(): (root / ('access-trace.txt' + suffix)).write_text(data)
            log = '[mac_expert_pack] reason=shutdown requests=1 read_requests=1 read_bytes=16 computed_experts=1\n'
            self.assertEqual(m.verify_traces(root, log)['legacy'], 1)
            for malformed in (files['.reads.tsv'].rstrip(), files['.reads.tsv'].replace('\t16\t16\t', '\t8\t16\t'), files['.reads.tsv'].splitlines()[0] + '\n'):
                (root / 'access-trace.txt.reads.tsv').write_text(malformed)
                with self.assertRaises(ValueError): m.verify_traces(root, log)
            (root / 'access-trace.txt.reads.tsv').write_text(files['.reads.tsv'])
            (root / 'access-trace.txt.phases.tsv').write_text(files['.phases.tsv'].splitlines()[0] + '\n')
            with self.assertRaises(ValueError): m.verify_traces(root, log)



class MainProtocolTests(unittest.TestCase):
    def run_driver(self, mode, *, teacher=False, cleanup_failure=False, stop_delay=0, async_sample=False, bad_reference=False):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        binary = root / 'llama-server'; binary.write_bytes(b'fixture executable')
        prompt = root / 'prompt.txt'; prompt.write_text('fixture')
        out = root / 'result'
        args = ['measure.py', '--model', 'dsv', '--model-file', str(root / 'model.gguf'), '--prompt-file', str(prompt),
                '--source', 'mmap', '--binary', str(binary), '--prompt-tokens', '1', '--tokens', '2', '--runs', '1', '--out', str(out)]
        if teacher:
            ref = root / 'reference'; ref.mkdir()
            (ref / 'results.json').write_text(json.dumps([{'tokens': [1, True] if bad_reference else [1, 2]}]))
            (ref / 'prompt.json').write_text('[11]')
            args += ['--teacher-reference', str(ref / 'results.json')]
        requests = []
        sample_started = threading.Event()
        release_sample = threading.Event()
        def response(url, timeout=None):
            route = url if isinstance(url, str) else url.full_url
            if route.endswith('/health'):
                if mode == 'health_interrupt': os.kill(os.getpid(), signal.SIGTERM)
                result = io.BytesIO(b'{}'); result.status = 200; return result
            payload = json.loads(url.data)
            if route.endswith('/tokenize'): return io.BytesIO(b'{"tokens":[11]}')
            requests.append(payload)
            self.assertIs(payload['post_sampling_probs'], False)
            if teacher:
                event = final(1); event.update(tokens=[1], completion_probabilities=[probability(1)])
                if len(requests) == 2 and mode == 'missing_stop': event.pop('stop')
                return io.BytesIO(json.dumps(event).encode())
            values = events()
            if mode == 'missing_probability': values[0].pop('completion_probabilities')
            if mode == 'missing_stop': values.pop()
            if mode == 'truncated': values[-1]['truncated'] = True
            class Stream(io.BytesIO):
                def __iter__(self):
                    for i, event in enumerate(values):
                        if i == 1 and async_sample:
                            self_test.assertTrue(sample_started.wait(1), 'boundary sample did not start')
                            release_sample.set()
                        if event.get('stop'):
                            time.sleep(stop_delay)
                        yield b'data: ' + json.dumps(event).encode() + b'\n\n'
            self_test = self
            return Stream()
        def sample(pid):
            if async_sample and threading.current_thread().name.startswith('ThreadPoolExecutor'):
                sample_started.set()
                if not release_sample.wait(2):
                    raise RuntimeError('stream reader blocked on boundary sampling')
            now = m.monotonic()
            return {'t': now, 'sample_begin': now, 'sample_end': now, 'time': time.time(), 'disks': {}, 'vm': {}, 'rss_kib': 1}
        server = FakeServer(cleanup_failure)
        old_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            with mock.patch.object(sys, 'argv', args), mock.patch.object(m.urllib.request, 'urlopen', side_effect=response), \
                 mock.patch.object(m.subprocess, 'Popen', return_value=server), mock.patch.object(m, 'system_sample', side_effect=sample), \
                 mock.patch.object(m, 'command', return_value='fixture'), mock.patch.object(m, 'snapshot_images', return_value={'images': []}, create=True), \
                 mock.patch.object(m, 'verify_image_continuity', return_value={}, create=True):
                if mode != 'valid' or cleanup_failure or bad_reference:
                    with self.assertRaises((ValueError, RuntimeError, InterruptedError)): m.main()
                else:
                    m.main()
        finally:
            for sig, handler in old_signals.items(): signal.signal(sig, handler)
        self.assertTrue(server.terminated)
        receipt = json.loads((out / 'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'complete' if mode == 'valid' and not cleanup_failure and not bad_reference else 'failed')
        return receipt, out

    def test_health_termination_cannot_continue_to_requests(self):
        receipt, out = self.run_driver('health_interrupt')
        self.assertEqual(receipt['validated_requests'], [])
        self.assertIn(signal.SIGTERM, receipt['pending_signals'])

    def test_valid_generation_receipt(self):
        receipt, out = self.run_driver('valid')
        result = json.loads((out / 'results.json').read_text())[0]
        self.assertEqual(result['decode_evaluations'], 1)
        self.assertEqual(result['output_tokens'], 2)
        self.assertTrue(receipt['performance_eligible'])

    def test_delayed_final_stop_does_not_extend_token_interval(self):
        _, out = self.run_driver('valid', stop_delay=0.03)
        result = json.loads((out / 'results.json').read_text())[0]
        self.assertGreater(result['end'] - result['last_token_event'], 0.025)
        interval = result['last_token_event'] - result['first']
        self.assertAlmostEqual(result['client_decode_tokens_per_second'], 1 / interval)

    def test_boundary_sampling_does_not_block_stream_reader(self):
        self.run_driver('valid', async_sample=True)

    def test_teacher_reference_ids_reject_boolean(self):
        self.run_driver('valid', teacher=True, bad_reference=True)

    def test_teacher_reference_is_hashed(self):
        _, out = self.run_driver('valid', teacher=True)
        binding = json.loads((out / 'teacher-reference.json').read_text())
        self.assertEqual(len(binding['results_sha256']), 64)
        self.assertEqual(len(binding['prompt_sha256']), 64)

    def test_missing_probability_fails_receipt(self): self.run_driver('missing_probability')
    def test_truncated_response_fails_receipt(self): self.run_driver('truncated')
    def test_missing_stop_fails_receipt(self): self.run_driver('missing_stop')
    def test_teacher_last_missing_stop_cannot_reuse_previous_stop(self):
        receipt, out = self.run_driver('missing_stop', teacher=True)
        self.assertEqual(len(receipt['validated_requests']), 1)
    def test_teacher_has_no_performance_eligibility(self):
        receipt, out = self.run_driver('valid', teacher=True)
        self.assertFalse(receipt['performance_eligible'])
    def test_successful_body_with_cleanup_failure_is_not_pass(self): self.run_driver('valid', cleanup_failure=True)


if __name__ == '__main__': unittest.main()
