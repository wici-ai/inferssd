# Pull request review

Pull requests are reviewed, verified and merged by an automated reviewer. It applies the rules on this page,
which are also the rules a human reviewer applies. Lessons it learns are recorded in [lessons](lessons.md)
and become rules here.

## Which pull requests

- The reviewer takes a pull request from a collaborator with write access when it is opened, gets new
  commits, is reopened or is marked ready for review. Drafts wait until they are marked ready.
- A pull request from an author without write access waits for a maintainer, because the reviewer builds and
  runs its code on the test devices. The maintainer reads the head commit and starts the reviewer for that
  commit: **Actions → pr-agent → Run workflow** with the pull request number and the head commit, or
  `gh workflow run pr-agent.yml -f number=N -f head=SHA`. The reviewer does not push fixes to such a pull
  request but asks for them; each new commit needs a maintainer to read it and start the reviewer again.
- A pull request that changes this page, [lessons](lessons.md) or the reviewer's workflow is verified like any
  other but merged by a maintainer: it changes the reviewer itself.

## What every pull request meets

1. **The contribution rules.** [CONTRIBUTING](../../CONTRIBUTING.md): one concern per pull request, the title
   format, no internal names, private addresses or links to private repositories, SPDX headers, the origin of
   ported code in NOTICE, fixes to code taken from llama.cpp belong in llama.cpp.
2. **Hosted CI passes** on the head commit.
3. **Existing paths are unchanged.** Existing models, a single GPU and the default configuration behave as on
   `main`: the reproducible configuration below is bitwise equal, work counters are not worse and times show
   no regression. This is measured, never assumed: a change that only looks like it touches a new path can
   still change shared code.
4. **No temporary flags.** A problem is fixed at its cause, not worked around with a command-line option,
   environment variable or special case. A capability is chosen by the framework's strategy, not enabled by
   hand.
5. **No approximations of our own.** Models run as their published GGUF files define them. Quantization,
   routing, top-k or KV-cache approximations are used only where the model's architecture or official recipe
   defines them.

## What merges, by kind of change

| Kind | Merges when |
| --- | --- |
| New capability (a model, a device, multi-GPU) | Existing paths are unchanged and the new path works where it can be run (tiny fixtures, the devices of that class) |
| Performance (a new mechanism, a changed default strategy or placement) | It is measured faster on the devices it changes and slower on none; no measured gain, no merge |
| Bug fix | The bug reproduces on `main`, is gone with the fix, and a new test fails with the fix reverted |
| Documentation, tests, tooling | The text is accurate and the tests or tools run |

## Numerical differences

Outputs are bitwise equal to `main` unless a pull request means to change them. Every difference is
attributed to its cause, with evidence, before anything is decided; the size of a numerical error alone
decides nothing.

| Cause | Evidence | Decision |
| --- | --- | --- |
| Only placement or the elastic VRAM tier differs | The difference is gone in the reproducible configuration | Judged on work counters and time |
| An operation the pull request changes on purpose, same precision (accumulation order, tiling, fusion, batch shape) | The first diverging graph node is that operation; with several changes, reverting this one restores bitwise equality; data types and compiler flags are unchanged | Merges when `test-backend-ops` passes and it is measured faster |
| An operation changed on purpose with less precision (narrower accumulators or intermediates, on-the-fly activation quantization, approximate math, fast-math flags, a lower default KV type, skipped work) | As above, plus the place where precision drops | Merges only where official llama.cpp computes it the same way on that backend, or the model's official recipe defines it |
| An operation the pull request does not mean to change | The first diverging node is outside the intended change | A bug: fixed or sent back |
| A numerical bug fix | The result matches the reference (official llama.cpp on the same backend, or the engine with the SSD path off) and a test fails with the fix reverted | Merges |
| No cause can be shown, or a path that was reproducible stops being reproducible | | A maintainer decides |

## How it is verified

- **On the same device, back to back.** `main` and the pull request merged onto current `main` are built from
  source on the device and run in the same slot, each on a fresh server, from the device's frozen machine
  profile. Devices are never compared with each other.
- **Two configurations.**
  - The reproducible one: every routed expert on the CPU (`vram.budget=0`), read-ahead off, pinned context
    and batch sizes, no prompt cache. Tokens and log probabilities are compared bitwise.
  - The default one, as a user runs it: work counters (bytes and requests read, cache and VRAM hits) and
    times.
- **Framework changes first offline.** Each side's framework derives the launch from every device's frozen
  profile. Identical launches need no device time; a changed launch is a performance change.
- **Devices by the paths a change touches.** CUDA code also runs on ROCm. Vulkan, Metal, Windows, unified
  memory and discrete GPUs are separate classes. Documentation needs no device.
- **Models a little larger than the device's memory**, so the SSD path runs in the default configuration, and
  the model a change is about when it is model specific. Tiny fixtures cover every architecture.

## Fixes by the reviewer

A small, local fix that keeps the pull request's approach (title or header format, a missed edge case, a
missing test, a few lines) is pushed to the branch by the reviewer and verified again. A change of approach
is sent back to the author with the evidence.

## The record

Before merging, the reviewer posts what it reviewed: the head commit, the `main` commit it was merged onto,
what ran where with the comparison, and what did not run. It merges only that exact head.
