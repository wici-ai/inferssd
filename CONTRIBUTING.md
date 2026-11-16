# Contributing

Contributions are welcome and are accepted under the repository's [Apache-2.0](LICENSE) license. No CLA or
DCO is required. AI-assisted contributions are welcome when you understand the change, have reviewed it and
take responsibility for maintaining it. Support is best effort.

Participation in this project is governed by the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

The first release focuses on the supported DeepSeek and Muse configurations. Please open an issue to discuss
additional models, new runtime policies or broad interface changes before implementing them.

## Issues

Open an issue with one of the forms: bug report, WiCi AI desktop app, model or quantization request, feature or
hardware request, performance report or question. The form sets the issue's type label, and its answers on the
operating system, backend and model storage add labels of their own. [Triage](.github/workflows/triage.yml)
assigns each new issue to an owner, labels each pull request with the areas in its title and asks an owner for
a review.

## Pull requests

- Work on a short branch name without `/` and without the name of the tool that produced the change.
- Keep each pull request to one concern. Separate numerical or algorithm changes from source-layout changes.
- A fix to code that InferSSD takes from llama.cpp, rather than to InferSSD's own code, belongs in llama.cpp:
  submit it to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). With the `upstream` alias from
  [Checks](#checks), `git diff upstream:src/llama-vocab.cpp HEAD:engine/src/llama-vocab.cpp` shows what InferSSD
  changed in a file.
- Pull requests are squash-merged, so the pull request title becomes the commit title on `main`.
- Do not add automated trailers. Use `Co-authored-by` only for human co-authors.
- Keep internal code names, task numbers, machine names, run IDs, credentials, personal paths, private
  addresses and links to private repositories out of code, branch names, commits, pull requests and issues.

### Pull request titles

Use `area: summary`, for example:

```text
ci: add isolated Mac base and PR performance comparisons
cuda, metal: fix expert staging
```

- `area` is one of the values below. List several areas separated by `, `.
- Changes to tests use the area of the module under test; there is no `test` area.
- `summary` is English in the imperative mood, starts in lowercase unless it begins with a proper noun, and
  has no trailing period. Accepted capitalized openings are a fixed list of names (such as Linux, Mac and
  Muse) and identifiers with inner capitals or digits (such as DeepSeek, CUDA or Qwen3).
- The whole title is printable ASCII and at most 72 characters.
- The title is also checked by `scripts/public-scan.py`: no personal paths, private addresses, login
  identifiers, task numbers or references to other `wici-ai/` repositories.
- Title a revert as `area: revert "summary"`.

| Area | Scope |
| --- | --- |
| `engine` | Native runtime under `engine/` not covered by a more specific area |
| `ggml` | ggml core, scheduler, allocator and shared SSD modules |
| `cuda` | CUDA backend and CUDA expert executor |
| `metal` | Metal backend and macOS GPU paths |
| `vulkan` | Vulkan backend |
| `cpu` | CPU backend and CPU expert executor |
| `server` | HTTP server, prompt cache and request lifecycle |
| `model` | Model architectures, chat templates and conversion |
| `framework` | The `inferssd` Python package |
| `bench` | Benchmark and measurement tools |
| `ci` | Workflows and CI tooling |
| `docs` | Documentation |
| `build` | CMake, presets and build scripts |
| `scripts` | Repository scripts |
| `repo` | Repository metadata, licensing and community files |

Check a title locally with `python3 scripts/check-pr-title.py "docs: fix quickstart link"`. The same check runs
on every pull request.

## Licensing

- Start new files with `SPDX-License-Identifier: Apache-2.0`.
- Code ported or transcribed from a third-party project keeps its original license and copyright notice.
  Add its origin to [NOTICE](NOTICE).
- Bundled llama.cpp and ggml files keep their MIT notices.

## Checks

For a new clone, create the local snapshot alias once:

```sh
git branch upstream 8884f8314ce8ea96d62cd602b6bbf7fc25758a09
```

Commit `8884f83` in the history of `main` holds the llama.cpp revision named in [NOTICE](NOTICE) without
InferSSD changes. No remote `upstream` branch is required.

```sh
python3 -m pip install ./framework pytest
python3 -m pytest -q bench/tests framework/tests
python3 scripts/inferssd/check-layering.py
python3 scripts/inferssd/check-layout.py
python3 scripts/public-scan.py --skip upstream
gitleaks git --log-opts="--all"
```

Framework lint and type checks are listed in [development checks](framework/README.md#development-checks);
the hosted checks are defined in [.github/workflows](.github/workflows).

Python tests use synthetic inputs and do not start a model. Run them with Python 3.9 or 3.12; the recorded
transcript checks are bound to those versions and skip on others. CI runs both. Unexpected skipped tests are
failures. Native test targets are described in [testing](docs/testing.md). CI builds the documented CMake presets,
runs the native tests labelled `ci` on Linux, macOS and Windows hosted runners, and compiles the CUDA server
whenever the engine or build configuration changes. The macOS and Windows packages are built for pull requests
that change how they are built and after each merge to `main`.

Device performance checks are triggered by maintainers after review and run only for pull requests from
branches in this repository; fork pull requests receive hosted CI only. See [device CI](bench/ci/README.md).

## Review checklist

- Explain the concrete behavior, its cause and the evidence for the change.
- Include a regression test for a behavior fix; use synthetic inputs where possible. Add a test that needs no
  GPU or model to hosted CI with `inferssd_ci_test` (see [testing](docs/testing.md)).
- Keep placement decisions general and derived from measured costs, without machine-specific shortcuts.
- Check CPU-only and applicable CUDA or macOS builds when shared code changes.
- Document memory ownership, thread lifetime and resource limits affected by the change.
- Report performance only with the exact source, model, configuration and workload recorded. See the
  [measurement checklist](docs/benchmark.md#reproducing-a-result).
