<!-- SPDX-License-Identifier: Apache-2.0 -->
# Usage events

[Back to documentation](README.md)

The server can record each inference request it served for the host that runs it: the route it came to, who
sent it, how many tokens it used, how long it took and how it ended. It writes nothing unless `INFERSSD_USAGE_DIR` names a directory. `inferssd serve` and `cli` pass the
variable through to the server unchanged.

## File

The server appends to `usage-events.jsonl` in that directory, creating both if needed (the file with mode 0600
where the OS has modes). Each record is one JSON object on one line that ends in `\n`, written with a single
append; a reader consumes complete lines only. One server process is one run; several runs may append to the
same file, and their event IDs never collide.

This version neither rotates nor prunes the file. After a write error the server stops recording for the rest
of the run and says so once in its log, so a partial line is never followed by another record.

## Records

```json
{"event_id":"Xc2kq0RmB7aTn4Lw:1","type":"request_received","request_id":"req-8b1KfQ2s9mZ0aLx3cV7nT4yR","occurred_at":"2026-11-10T04:59:58Z","route":"/v1/responses","client":"codex","user_agent":"codex_cli_rs/0.169.0"}
{"event_id":"Xc2kq0RmB7aTn4Lw:2","type":"usage_finalized","request_id":"req-8b1KfQ2s9mZ0aLx3cV7nT4yR","occurred_at":"2026-11-10T05:00:03Z","token_count":840,"input_tokens":812,"output_tokens":28,"cached_tokens":700,"prompt_ms":1234,"output_ms":2000,"outcome":"completed"}
```

A reader takes the fields it knows and passes over the rest, so that a later version can add to them.

| Field | Meaning |
| --- | --- |
| `event_id` | `<run>:<seq>`. The run is random for each server process; `seq` starts at 1 and follows the file order. Unique across runs, so a reader that reads a line twice can drop it by this ID. |
| `type` | `request_received` or `usage_finalized`. |
| `request_id` | Random for each HTTP request; pairs its two records. |
| `occurred_at` | UTC, to the second, when the request reached its route or when its usage was final. |
| `route` | On `request_received`: the path the request came to, as the client wrote it, without its query. |
| `client` | On `request_received`, when the request has an API key: the name in a key `inferssd-<name>`, else the key's first 10 characters. See [Who sent a request](#who-sent-a-request). |
| `user_agent` | On `request_received`, when the request has a `User-Agent` header: its printable ASCII, cut to 120 characters. |
| `input_tokens` | Prompt tokens, as the OpenAI-style `prompt_tokens` reports them: prompt-cache hits included. (Anthropic's `input_tokens` leaves the hits out.) |
| `output_tokens` | Generated tokens, reasoning included. |
| `token_count` | `input_tokens + output_tokens`. |
| `cached_tokens` | The part of `input_tokens` the slot already held, so that the server did not read it again. |
| `prompt_ms` | Milliseconds spent reading the rest of the prompt. |
| `output_ms` | Milliseconds spent generating. |
| `outcome` | `completed`, `failed` or `cancelled`. |
| `error` | On a `failed` request: the reason, as the response's `message` gives it, cut to 300 characters. |

`cached_tokens`, `prompt_ms` and `output_ms` are 0 for a request that ended before its prompt was read, whatever
its outcome.

## Who sent a request

The server needs no API key, but most tools send one, and a host that points several tools at one server can
tell them apart by it. The server reads the key from `Authorization: Bearer` or, without that header, from
`X-Api-Key`, and writes `client` from it:

- A host that names a tool sets the tool's key to `inferssd-<name>`. `client` is then the name, its printable
  ASCII, cut to 64 characters.
- Any other key may be a secret for another service. `client` is its first 10 characters, which tell one such
  caller from another; the rest is written nowhere, not here and not in the server log.

The key identifies nothing: any caller can send any key. With `--api-key`, a request is still checked against
the keys given there before it reaches a route.

## The server log

The server log has the same request IDs, so that a record here can be read with the log lines of its request:

```text
srv         begin: usage events: request req-8b1KfQ2s9mZ0aLx3cV7nT4yR to /v1/responses
slot launch_slot_: id  0 | task 12 | usage events: request req-8b1KfQ2s9mZ0aLx3cV7nT4yR
```

The desktop app reads these and other lines of the log by their text: [server log lines the desktop app
reads](server-log.md) lists them.

The first line is written when the request reaches its route, before anything else the route logs; the second
when a slot takes one of the request's tasks, after that task's `processing task` line.

## What is counted

A request to one of these routes writes `request_received` as soon as its handler starts, before its body is
read, so a request that fails to parse is still counted:

- `/completion`, `/completions`, `/v1/completions`
- `/chat/completions`, `/v1/chat/completions`
- `/responses`, `/v1/responses`
- `/v1/messages`

Other routes are not counted, among them `/health`, `/models`, `/props`, `/metrics`, `/tokenize`,
`/apply-template` and `/v1/messages/count_tokens`. Neither are requests refused before they reach a route:
`503 Loading model` while the model loads, or a missing API key.

Each request gets at most one `usage_finalized`, written when all of its work has ended and its response is done:

- Several prompts in one request add up. Several candidates (`n`) share one prompt, so it counts once, and
  their outputs add up. `inferssd serve` runs one slot, so it refuses `n` above 1, which counts as a failed
  request with no tokens.
- A request that fails before it runs, such as one that does not parse, counts 0 and 0.
- A cancelled request keeps what it used: the prompt once a slot has taken it, and the tokens generated until the
  cancellation. One still waiting for a slot counts 0.
- When the server stops on SIGINT or SIGTERM, each open request is written as it stands, as `cancelled`. If the
  process is killed, a request may have only its `request_received` line; its usage is then unknown.
