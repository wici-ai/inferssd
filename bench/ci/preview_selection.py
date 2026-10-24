#!/usr/bin/env python3
"""Unprivileged PR preview only; private admission recomputes this from its own API."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import load, policy_read, save
from ci.impact import plan
from ci.private_controller import API
from ci.selected_controller import changed_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    event = load(os.environ['GITHUB_EVENT_PATH'])
    pr = event['pull_request']
    api = API('wici-ai/inferssd', os.environ['GITHUB_TOKEN'])
    current = api.call('pulls/' + str(pr['number']))
    if any(current[key]['sha'] != pr[key]['sha'] for key in ('base', 'head')):
        raise ValueError('PR source changed during preview')
    policy = policy_read(load(Path(__file__).with_name('policy.json')))
    value = plan(changed_paths(api, current), policy, base=pr['base']['sha'],
                 head=pr['head']['sha'], candidate=os.environ['GITHUB_SHA'])
    save(args.output, dict(preview=True, physical_authorization=False, selection=value))
    print(json.dumps(value, indent=2))
