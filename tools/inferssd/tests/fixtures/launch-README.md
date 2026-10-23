# Original launcher observations

Expected outputs were produced only by frozen F 8b12192920b339a4ef31c9006d1ebe20304ce45a, before the typed resolver existed. `oracle/launch-manifest.json` binds every original source. Linux uses Bash 5.2.21 on a Linux fixture host; Mac observations execute the original Python module with explicit Darwin/arm64 facts and intercepted process effects, under Python 3.9 and 3.12. No model or native engine is run.

The 58 Linux cases include repeated index reuse/invalidation and preserve complete stdout/stderr/status, metadata command replies, ordered stub calls and artifact bytes. Native/service commands are fail-closed sentinels. CPU/memory/CUDA/port/service replies are fixture facts, not measurements of a GPU. The Linux uid and cgroup-v2 existence are explicit observations, not invented Mac behavior. The 28 Darwin cases capture would-exec argv/env, filesystem publication and all defined errors; they do not claim Apple hardware execution.

Only fixture-root and interpreter spelling are replaced with named placeholders. Mac's random temporary directory and path-derived alias digest are canonicalized explicitly; neither is an algorithm output under comparison. Fixed-path tests separately check the digest and publication paths. Numeric results, argument order, receipts, diagnostics and partial artifacts are not normalized. F's die helper includes the exit-code argument in some messages; those trailing `2` strings are retained.

The original shell suites contain 102 settings assertions plus 6 replica cases (108 total); the design's historical 106 predates two scope-name checks. The original suite source is preserved unchanged. Its existing historical replica rules are not used to generate the new resolver's expected output.

Original dependency-capture mistake: an initial private suite run omitted settings.py and its imports, so four settings-generation assertions failed with ModuleNotFoundError; the first run remains archived. The oracle closure includes those four untouched F modules. No production source was changed to make a test pass.
