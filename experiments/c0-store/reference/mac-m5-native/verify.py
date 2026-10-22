#!/usr/bin/env python3
"""Read completed Mac artifacts and compare two requests with recorded F bytes.

Never executes a model or generates expected output. Saved event order is checked;
actual process/library/source identity, path coverage, resources and cleanup remain
separate execution-gate requirements.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def regular_bytes(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('expected a regular non-symlink artifact')
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                              value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or path.is_symlink():
        raise ValueError('artifact changed while reading')
    return data


def object_json(path):
    value = json.loads(regular_bytes(path))
    if not isinstance(value, dict):
        raise ValueError('JSON artifact must be an object')
    return value


def ready(value, expected):
    if not isinstance(value, dict):
        raise ValueError('READY must be an object')
    for key in ('n_vocab', 'float_bytes', 'effective_n_ctx', 'effective_n_batch', 'effective_n_ubatch'):
        if type(value.get(key)) is not int:
            raise ValueError('READY dimensions must be integers')
    if value != expected:
        raise ValueError('READY differs from the recorded Mac profile')


def logits(data, selected):
    if len(data) != 259 * 4:
        raise ValueError('full-vocabulary float32 byte count differs')
    values = struct.unpack('<259f', data)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('nonfinite raw logit')
    if type(selected) is not int or max(range(259), key=values.__getitem__) != selected:
        raise ValueError('selected token is not the lowest-ID raw maximum')


def verify_events(path, expected_ready):
    events = [json.loads(line) for line in regular_bytes(path).splitlines()]
    if len(events) != 7:
        raise ValueError('exact READY/begin/end/DONE transcript for two requests required')
    ready(events[0], expected_ready)
    last_stamp = 0
    for request in (1, 2):
        begin, end, done = events[1 + 3 * (request - 1):4 + 3 * (request - 1)]
        for event, name in ((begin, 'begin'), (end, 'end')):
            if (not isinstance(event, dict) or set(event) != {'event', 'request', 'steady_ns'} or
                    event['event'] != name or type(event['request']) is not int or
                    event['request'] != request or type(event['steady_ns']) is not int or
                    event['steady_ns'] <= last_stamp):
                raise ValueError('request boundary order/identity/timestamp differs')
            last_stamp = event['steady_ns']
        if (not isinstance(done, dict) or done != {'event': 'done', 'request': request} or
                type(done.get('request')) is not int):
            raise ValueError('missing/wrong complete DONE event')


def verify(directory, model):
    directory = Path(directory)
    root = Path(__file__).resolve().parent
    manifest = object_json(root / 'manifest.json')
    if (manifest.get('protocol') != 'mac-reader-raw-v1-resource-v2' or
            manifest.get('protocol_settings', {}).get('requests_per_process') != 2):
        raise ValueError('only the recorded two-request Mac profile is accepted')
    cases = [case for case in manifest['cases'] if case['model'] == model]
    if len(cases) != 1:
        raise ValueError('unknown model case')
    case = cases[0]
    expected_ready = {'event': 'ready', 'n_vocab': 259, 'float_bytes': 4, 'byte_order': 'little',
                      'effective_n_ctx': 256, 'effective_n_batch': 16, 'effective_n_ubatch': 16}
    ready(case['ready'], expected_ready)
    ready(object_json(directory / 'raw-ready.json'), expected_ready)
    verify_events(directory / 'runner-events.stdout', expected_ready)
    result = object_json(directory / 'result.json')
    ready(result.get('raw_ready'), expected_ready)
    if result.get('status') != 'passed' or result.get('requests_complete') is not True:
        raise ValueError('missing worker-attested completion')
    responses = result.get('responses')
    if not isinstance(responses, list) or len(responses) != 2:
        raise ValueError('exactly two completed responses required')
    if {p.name for p in directory.glob('request-*.json')} != {'request-1.json', 'request-2.json'}:
        raise ValueError('exactly two request receipts required')
    expected_files = {f'request-{request}-step-{step}.f32' for request in (1, 2) for step in range(4)}
    if {p.name for p in directory.glob('*.f32')} != expected_files:
        raise ValueError('exactly eight raw rows required')
    if len(case['rows']) != 4 or len(case['selected_ids']) != 4:
        raise ValueError('reference requires four decisions')
    for request in (1, 2):
        receipt = object_json(directory / f'request-{request}.json')
        if type(receipt.get('request')) is not int or receipt['request'] != request:
            raise ValueError('request identity differs')
        rows = receipt.get('rows')
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError('four complete row records required')
        wire_rows = []
        for step, (actual, reference) in enumerate(zip(rows, case['rows'])):
            if not isinstance(actual, dict):
                raise ValueError('row record must be an object')
            geometry = {'step': step, 'position': 3 + step, 'batch_tokens': 4 if step == 0 else 1}
            for record in (actual, reference):
                for key, value in geometry.items():
                    if type(record.get(key)) is not int or record[key] != value:
                        raise ValueError(f'step {step}: {key} differs')
                if type(record.get('selected_id')) is not int or record['selected_id'] != case['selected_ids'][step]:
                    raise ValueError('selected token differs from reference')
            reference_name = f'{model}/step-{step}.f32'
            if reference.get('file') != reference_name:
                raise ValueError('reference path differs')
            data = regular_bytes(root / reference_name)
            if (reference.get('n_vocab') != 259 or reference.get('dtype') != 'IEEE-binary32' or
                    reference.get('byte_order') != 'little' or reference.get('bytes') != 1036 or
                    hashlib.sha256(data).hexdigest() != reference['sha256']):
                raise ValueError('reference bytes/shape differ from manifest')
            logits(data, reference['selected_id'])
            name = f'request-{request}-step-{step}.f32'
            if actual.get('file') != name:
                raise ValueError('unexpected actual raw filename')
            observed = regular_bytes(directory / name)
            logits(observed, actual['selected_id'])
            if observed != data:
                raise ValueError(f'request {request} step {step}: full raw bytes differ')
            wire_rows.append({**reference, 'file': name})
        if responses[request - 1] != {'tokens': case['selected_ids'], 'raw_logits': wire_rows}:
            raise ValueError('worker response differs from complete raw rows')
    return {'status': 'passed-recorded-bytes', 'model': model, 'source_mode': 'alias',
            'requests': 2, 'rows': 8, 'vocabulary': 259, 'prompt_tokens': 4,
            'saved_ready_begin_end_done_order_verified': True,
            'runtime_identity_coverage_resources_cleanup_attested': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-dir', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    try:
        result = verify(args.actual_dir, args.model)
    except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error) as error:
        parser.exit(1, f'Mac reference verification failed: {error}\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
