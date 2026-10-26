# Contributing

Contributions are accepted under the repository's Apache-2.0 license. No CLA or DCO is required. AI-assisted
contributions are welcome when the contributor understands the change, reviews it and takes responsibility
for maintaining it. Support is best effort.

Use a short branch name without slashes and open a pull request. Each commit should build and address one
concern. Use an English ASCII subject in the form `area: summary`, and explain the behavior and validation in
the description. Do not add automated authorship trailers. Preserve the actual contributor's authorship.

## Checks

For a new clone, create the local snapshot alias once with
`git branch upstream 8884f8314ce8ea96d62cd602b6bbf7fc25758a09`.
The snapshot is part of main's history; no remote `upstream` branch is required.

```sh
python3 -m pip install ./framework
python3 -m pytest -q framework/tests tests/integration
bash framework/tests/fixtures/oracle/launch/tools/adaptive/tests/test_launcher_settings.sh
bash framework/tests/fixtures/oracle/launch/tools/adaptive/tests/test_launcher_replica.sh
python3 scripts/public-scan.py --skip upstream
gitleaks git --log-opts="--all"
```

Launcher tests run against stubs on Linux and do not start a model. Missing fixtures and unexpected skipped
tests are failures. The CPU build job runs synthetic tests; CUDA and full-model checks require suitable
hardware. Maintainers run device checks only after reviewing a change. External pull requests do not run
on machines containing credentials or model-serving workloads.

Keep machine access details, credentials, personal paths and deployment information out of code, commits,
pull requests and issues. New files should identify their license. Record third-party origins in NOTICE.

## Review checklist

- Explain the concrete behavior, its cause and the evidence for the change.
- Include a relevant regression test for a behavior fix; use synthetic inputs where possible.
- Keep placement decisions general and derived from measured costs, without machine-specific shortcuts.
- Separate numerical or algorithm changes from source-layout changes.
- Check CPU-only and applicable CUDA/macOS builds when shared code changes.
- Document memory ownership, thread lifetime and resource limits affected by the change.
- Preserve upstream attribution and update provenance for ports.
- Report performance only with the exact source, model, configuration and workload recorded.
- Check the proposed history with gitleaks and public-scan before pushing.

The first release focuses on the supported DeepSeek and Muse configurations. Additional models, new runtime
policies and broad interface redesigns require discussion before implementation.
