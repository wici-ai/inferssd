<!-- SPDX-License-Identifier: Apache-2.0 -->
# Server log lines the desktop app reads

[Back to documentation](README.md)

The WiCi AI desktop app has an Engine Log page that shows what the server is doing: each request it took,
how far a prompt has been read, what was answered and why a request failed. It gets the times, the route, the
caller, the counts and the failure from the [usage events](usage-events.md), which are a contract. The rest it
reads from the server log (`server.log` in the run's directory), which is written for people and has no
contract: the app matches the lines below by their text.

**When you change the wording of one of these lines, or the order they come in, change the app's reader with
it** (its log parser and that parser's tests, which hold real lines from this log), or tell its maintainers
before the engine is released. A line the reader no longer matches does not fail anywhere: the page just shows
less.

Prefer adding a field to the usage events over adding a line here. What the events carry, the app no longer
needs from the log.

## The lines

Matched anywhere in a line, case as written. `N` is a number.

| Line | Written in | What the app takes from it |
| --- | --- | --- |
| `usage events: run <run> appends to <path>` | `inferssd/usage.cpp` | The usage events file to read. |
| `usage events: request <id> to <route>` | `inferssd/usage.cpp` | A request's lines begin here; its ID joins them to its usage events. |
| `\| task N \| usage events: request <id>` | `server-context.cpp`, when a slot takes a task | The same, on the slot's lines. |
| `\| task N \| processing task` | `server-context.cpp` | A model run began. |
| `the slot reuses N of N tokens` | `inferssd/prompt-cache.cpp` | How much of the prompt the slot already held. |
| `\| task N \| new prompt, … task.n_tokens = N` | `server-context.cpp` | The prompt's size. |
| `\| task N \| prompt processing progress, n_tokens = N, … progress = F` | `server-context.cpp` | How far the prompt has been read. |
| `\| task N \| prompt processing done` | `server-context.cpp` | The answer is being written. |
| `\| task N \| ` (nothing after it), then `prompt eval time = … / N tokens (… F tokens per second)` and `eval time = … / N tokens (… F tokens per second)` | `server-context.cpp`, the timings | Tokens read and written, and their speeds. |
| `\| task N \| stop processing: n_tokens = N, truncated = N` | `server-context.cpp` | The run ended, and whether the prompt was cut. |
| `cancel task, id_task = N` | `server-queue.cpp` | The caller dropped the request. |
| `task id = N, error: <text>` | `server-context.cpp` | The run failed, and why. |
| `got exception: <json>` | `server.cpp` | Why a request was refused: the `message` in the JSON. |
| `done request: <METHOD> <path> <address> <status>` | `server-http.cpp` | The request's path and status. |
| `new slot, n_ctx = N` | `server-context.cpp` | The context the model was loaded with. |
| `srv    load_model: loading model …` (at the line's start) | `server-context.cpp` | The model is loading. |
| `main: server is listening on …` (at the line's start) | `server.cpp` | The model is loaded and the server answers. |
| `cleaning up before exit` | `server.cpp` | The server is stopping. |

The app also shows log lines as they are, beside the request they belong to: every line that starts with
`srv`, `slot`, `main` or `que`, the timing lines, and any line with `error`, `exception`, `failed`, `abort` or
`warning` in it. It leaves out the lines of the prompt cache's listing (`update:    - prompt …`,
`- cache state`), `memory_seq_rm`, `sampler chain` and the progress lines, so a new line that is written once
for each layer or each cached prompt should not start with `srv` or `slot`.

## What the order means

One slot serves one request at a time (`inferssd serve` passes `-np 1`), and the app relies on it. A request's
lines run from its `usage events: request` line to the next one. In a server that does not write that line,
`done request` has no task number, so the app gives it to the oldest run that has no path yet.
