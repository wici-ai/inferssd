"""Synthetic recorded text only; no machine probe or native process."""


def core(name, text, **extra):
    return {
        "name": name,
        "kind": "coresets",
        "argv": ["coresets.py", "cpus.txt"],
        "files": {"cpus.txt": text},
        **extra,
    }


def tier(name, logs, ubs=None, **extra):
    ubs = list(logs) if ubs is None else ubs
    files = {f"recorded/ubprobe-{ub}.out": f"runs/{ub}\nignored\n" for ub in logs}
    files.update({f"runs/{ub}/server.log.gz": text for ub, text in logs.items()})
    return {
        "name": name,
        "kind": "ubprobe",
        "argv": ["ubprobe.py", "recorded", *ubs],
        "files": files,
        **extra,
    }


A = "[kimi_vram_tier] reason=allocated x slots="
G = "[kimi_vram_elastic] event=decode_grow x after_slots="
W = "[kimi_vram_elastic] event=watermark_grow x slots="

CASES = [
    core(
        "groups-unions-reserve-one",
        "3 100 10 -\n1 100 10 -\n2 100 10 -\n0 100 10 -\n4 50 99 -\n5 10 999 -\n",
    ),
    core("threshold-anchored-first", "0 100 1 -\n1 92 1 -\n2 91.99999999999999 1 -\n3 85 1 -\n"),
    core("ties-numeric-cpu-order", "9 7 1 -\n2 7 1 -\n1 7 1 -\n"),
    core("capacity-fallback", "0 0 10\n1 -2 20\n2 nan 30\n3 1 999\n"),
    core("negative-ranks", "0 -1 -100 -\n1 0 -100 -\n2 0 -200 -\n"),
    core("nonfinite-ranking", "0 inf 0 -\n1 inf 0 -\n2 0 nan -\n3 0 -inf -\n"),
    core("duplicate-cpu-rows", "0 100 1 -\n0 100 1 -\n1 100 1 -\n"),
    core("siblings-filter", "0 100 1 0-1\n1 100 1 0-1\n2 50 1 2,3\n3 50 1 2,3\n"),
    core("siblings-min-not-first", "2 100 1 2,0\n0 100 1 2,0\n"),
    core("empty-file", ""),
    core("blank-line", "\n"),
    core("missing-frequency", "0 100\n"),
    core("bad-capacity", "0 no 1\n"),
    core("reversed-sibling-range", "0 1 1 2-1\n"),
    core("empty-sibling-list", "0 1 1 ,\n"),
    core("negative-sibling", "0 1 1 -1\n"),
    core("all-siblings-filtered", "1 100 1 0-1\n"),
    core("extra-fields-ignored", "0 1 1 - ignored extra\n"),
    core("read-fault-after-one", "0 1 1\n1 2 2\n", faults={"cpus.txt:next:1": "OSError"}),
    core("open-fault", "0 1 1\n", faults={"cpus.txt:open": "PermissionError"}),
    core("partial-print-fault", "0 100 1\n1 50 1\n", write_fault=5),
    {"name": "core-missing-argv", "kind": "coresets", "argv": ["coresets.py"], "files": {}},
    *[
        {"name": f"parse-list-{i}", "kind": "parse_list", "value": value, "files": {}}
        for i, value in enumerate(
            ("", "0,2-4, 7", "3-1,2,2", "1, ,2", "-1", "1-2-3", "a", " 1 - 2 ")
        )
    ],
    tier(
        "first-grow-and-high-water",
        {"32": A + "2\n" + W + "8\n" + G + "4\n" + G + "9\n" + A + "5\n"},
    ),
    tier("allocated-only", {"16": A + "5\n" + A + "3\n"}),
    tier("watermark-before-allocation", {"32": W + "9\n" + A + "4\n"}),
    tier("missing-events", {"32": "unrelated\n"}),
    tier("zero-allocation", {"32": A + "0\n"}),
    tier("zero-growth", {"32": G + "0\n"}),
    tier("zero-watermark", {"32": W + "0\n"}),
    tier("positive-then-zero-typeerror", {"32": A + "2\n" + G + "0\n"}),
    tier("zero-first-then-positive", {"32": A + "0\n" + A + "5\n"}),
    tier("first-regex-match", {"32": A + "3 " + G + "9\n"}),
    tier("regex-negative-not-recognized", {"32": G + "-2\n" + A + "4\n"}),
    tier("ordered-duplicate-ubs", {"32": A + "4\n", "16": G + "2\n"}, ["32", "16", "32"]),
    tier("missing-path-continues", {"16": A + "2\n"}, ["32", "16"]),
    tier(
        "readline-oserror-continues",
        {"32": A + "4\n", "16": G + "2\n"},
        faults={"recorded/ubprobe-32.out:readline:0": "OSError"},
    ),
    tier("gzip-open-oserror", {"32": A + "4\n"}, faults={"runs/32/server.log.gz:open": "OSError"}),
    tier(
        "gzip-read-oserror",
        {"32": A + "4\n" + G + "8\n"},
        faults={"runs/32/server.log.gz:next:1": "OSError"},
    ),
    tier(
        "gzip-read-valueerror",
        {"32": A + "4\n"},
        faults={"runs/32/server.log.gz:next:0": "ValueError"},
    ),
    tier("print-fault", {"32": A + "4\n"}, write_fault=1),
    {
        "name": "ub-empty-run-path",
        "kind": "ubprobe",
        "argv": ["ubprobe.py", "recorded", "32"],
        "files": {"recorded/ubprobe-32.out": "  \nsecond\n", "/server.log.gz": A + "3\n"},
    },
    {"name": "ub-no-values", "kind": "ubprobe", "argv": ["ubprobe.py", "recorded"], "files": {}},
    {"name": "ub-missing-argv", "kind": "ubprobe", "argv": ["ubprobe.py"], "files": {}},
    *[
        {"name": f"reached-{i}", "kind": "reached", "lines": lines, "files": {}}
        for i, lines in enumerate(
            ([], [A + "1", G + "5", G + "3"], [W + "0"], [A + "0", G + "0"], [W + "3", A + "0"])
        )
    ],
]
