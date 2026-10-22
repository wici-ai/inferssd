# DeepSeek-V4.1-Flash with Hermes Agent: adaptation record

What had to change for Hermes Agent to run well against the DeepSeek-V4.1-Flash server, why, and how it was
checked. Setup instructions are in [the setup guide](deepseek-v41-flash-hermes-agent.md).

Tested with Hermes Agent `a28a5d03` (2026-10-07) on a Mac against the 166 test server (RTX 5090 with the
16 GB host envelope of the launcher, `m16g4`), thinking at `medium`, real tasks: count the lines of the Python
files in a directory and write them to a file, then, in a resumed session, append the total.

## The problem

A DSV4 sequence cannot be rolled back. Its cache holds a 128-token sliding window of raw keys, compressed rows
for older tokens, and the running state of the compressors. When a new prompt differs from the cached tokens
anywhere before their end, the server can reuse them only from a state saved at or before that point; without
one it processes the whole prompt again. Hermes against the server as of v0.1.14/v0.1.15 (checkpoints off,
prompt cache off):

| What | Result |
| --- | --- |
| Requests inside a session | 6 of 7 processed the whole conversation again, 55-62 s each |
| Second turn of a resumed session | 284 s |
| First request of a new session | the whole 7,109-token prompt, although the previous session started with the same tokens |
| Request after Hermes' title request | the whole 14,296-token conversation again (108 s) |

Five causes, found by comparing the rendered prompts of consecutive requests:

1. Hermes sends earlier assistant turns back re-serialized: content trimmed, tool-call arguments re-encoded
   with sorted keys and no spaces, an empty reasoning padded with a space. Rendered, the history differs from
   the tokens the model generated, inside the previous turn.
2. With checkpoints off nothing could absorb such a difference, and with checkpoints on (upstream behavior)
   taking one stops the prompt batch: with experts streamed from disk, each extra batch streams the experts
   again. A DSV4 checkpoint also held every raw cell the sliding-window cache still had, up to
   n_swa + n_ubatch cells (about 330 MiB with `-ub 8192`).
3. Nothing kept the state at the end of the system prompt and tools, where conversations branch.
4. Requests that are not part of the conversation (title generation, approvals, background review) replace the
   only slot's state, and nothing kept the conversation they displaced.
5. A request that decodes no token (a 1-token reply, EOS or a stop string as the first token, a cancel after
   the prompt) left the expert tier in a state where the next prefill is not prepared. On the test server the
   requests after it failed with "failed to prepare GPU memory for prompt processing" (HTTP 500), three in a
   row, since none could decode a token to reset the state; on another deployment the prefill fell back to
   CPU experts.

## Changes

### Turn memory (`server: render the assistant turns this server generated as generated`)

The server records each assistant turn it generates (content, reasoning and tool-call argument strings, the
last 256 turns) and finds a turn by what survives re-serialization: its trimmed content and its tool calls
with the arguments compared as JSON values. A recognized turn in a request is rendered as generated, so the
prompt matches the cache and the model sees what it actually wrote. Reasoning a client left out, padded or sent
back unchanged is the generated one; reasoning a client changed is kept. This replaces the reasoning memory of
v0.1.14, which restored only the reasoning and only when the client sent none, and whose key kept the
argument order, so it missed every turn whose arguments Hermes reordered.

### Earlier states without stopping the batch (`dsv4: save only the raw window, and earlier states of the last batch`)

- A saved DSV4 state keeps only the raw window that later positions attend to: a checkpoint is 5 MiB instead of
  up to 330 MiB. A full state is that plus about 5 KiB per token of compressed rows.
- V4.1's compressors do not overlap (ratios 2, 2 and 1). At any even position inside the batch just decoded,
  the raw window before it is still in the cache, the compressed rows of the closed blocks are final, and no
  compressor holds a partial block, so the state at that position can be rebuilt after the batch.
  `llama_memory_i::state_write_at()` and `llama_state_seq_{at_align,can_save_at,get_size_at,get_data_at,
  fwrite_at}` expose it; `llama_state_seq_fread_ext` reads a streamed state back. DeepSeek-V4 (overlapping
  CSA) and other memories report that they cannot.

### Checkpoints, shared prefix, prompt cache on disk (`server: checkpoint after the batch, keep the shared prefix and the prompt cache on disk`)

- For a memory that can save earlier states, checkpoints are taken after the batch is decoded, never by
  stopping it. The positions are where a later prompt branches off: a few tokens before the end of the prompt,
  every `checkpoint_every_nt` tokens, and the end of the leading system messages with the tools.
- The chat endpoint renders the leading system messages with the tools on their own to find that shared
  prefix. Its state also goes to the prompt cache, so a new conversation starts from it.
- `--cache-disk DIR` keeps the prompt cache on disk instead of in RAM. States are streamed to files, written
  with `fdatasync` and dropped from the page cache, so they take no memory from the server's cgroup. Each file
  carries a fingerprint of the model file and the build; files of another fingerprint, damaged files and
  leftovers of an interrupted write are removed at startup. A file that fails to load is removed and the
  request processes its prompt.
- Loading a cached prompt through one of its checkpoints restored only the checkpoint's partial state, over
  whatever the slot held. It now restores the prompt's full state first.
- For a state that cannot be rolled back, the slot looks for a cached prompt whenever a new prompt does not
  just extend its own and takes the one that lets it skip the most tokens. The conversation a new request
  displaces is saved first (when it would lose more than half of its tokens, as before).
- A checkpoint of such a state is usable when it lies inside the common prefix. The sliding-window margin that
  upstream applies to SWA checkpoints rejected the most useful checkpoint, the one at the end of the previous
  prompt.

### Prefill phase (`cpu-sink: reopen the prefill phase after a request without a decode step`)

The CPU sink and the VRAM tier each skipped a prefill `begin()` when the previous one was also a prefill, to
drop the repeats that graph reserves issue inside a request. But the tier leaves prefill at `prefill_done`,
not at the next decode graph, so after a request without a decode step the next request's transition was
dropped: no MXFP4 prefill scratch, no armed reclaim. Both now track whether a prefill phase is open. The server
reports `request_end` exactly once for every request it opened, from `release()` if nothing else did, so
cancelled and failed requests leave the prefill phase too.

### Launcher (`launcher: keep checkpoints, and saved prompt states on disk`, `launcher: make the DeepSeek VRAM reserve an opt-in setting`)

- 8 checkpoints and the prompt cache in `PROMPT_CACHE_DIR` (16 GiB by default).
- v0.1.15 had written one deployment's VRAM headroom (a 576 MiB allocator reserve, 576/640 MiB decode
  watermarks, a safety margin of at least 640 MiB) into the standalone launcher, which itself refuses to start
  next to another CUDA process. `VRAM_RESERVE_MIB` (default 0) now sets the reserve, both watermarks and the
  safety margin from one value; with a reserve the launcher shares the GPU. `VRAM_RESERVE_MIB=576` gives the
  v0.1.15 values.

## Checks

The SSD of the test server was in its slow state during these runs (1-3 GiB/s instead of 11.4), so they report
tokens and outcomes; the times were measured afterwards (see [Timings](#timings)).

**Same answers as a full prefill.** Next-token distributions on real text (a novel excerpt in the system
prompt, fixed answers, thinking off), compared with the same prompt processed from scratch:

| Path | Positions | Same top-1 | Mean \|delta log-prob\| over the reference's top 5 |
| --- | ---: | ---: | ---: |
| Prompt appended to the previous turn (the path clients always took; noise floor) | 30 | 30 | 0.358 |
| Restored from a checkpoint taken after the batch, rest processed | 30 | 29 | 0.293 |
| System part processed alone, question appended (noise floor) | 6 | 6 | 0.492 |
| New conversation from the shared-prefix checkpoint | 6 | 6 | 0.545 |
| New conversation from the shared prefix loaded from disk | 6 | 6 | 0.473 |

The one top-1 difference is at a position where the two distributions are close (mean delta 0.22). Greedy
24-token answers after restoring a checkpoint, and after loading a prefix that was saved from inside a batch,
matched the full-prefill answer token for token; in one further run the restored path and the reference took
different words at token 21, a near tie that the reference itself had resolved the other way in an earlier
run.

**Turn memory.** A tool call generated with thinking on (187 characters of reasoning) and off, sent back as
Hermes does (arguments `{"content":...,"mode":420,"path":"notes.txt"}` instead of the generated
`{"path":"notes.txt","content":...,"mode":420}`, empty reasoning as a space): both histories render to the
same tokens, and the next request processes only the tool result.

**Hermes sessions** (tokens processed per request):

| Case | Before | After |
| --- | --- | --- |
| Turn 1, small toolset, cold | 7,109, then mostly the whole conversation again | 7,109, then 48 and 100 |
| Turn 2, resumed session | 6 of 7 requests processed everything | 24, 163, 87 |
| New session, same system prompt | 7,109 | 45 (shared-prefix checkpoint) |
| Default configuration (24 tools, title generation), turn 2 after the title request | 14,296 | 24 (conversation loaded from disk) |
| New session after a server restart | 7,109 | 45 (shared prefix loaded from disk) |

All tasks were answered correctly. The prompt cache held 12 states, 499 MB, after these sessions.

**Prefill phase.** A request after a 1-token reply: before, it failed with HTTP 500, and so did the two
requests after it; after, its prefill opened normally (`prefill_armed`, `prefill_scratch status=ready`). A request
cancelled between the two batches of a 16,000-token prompt closed its prefill phase on release, and the next
request was served normally.

**Builds.** Each commit builds on its own (CUDA 13.0, sm_120a).

## Timings

Measured on 2026-10-09 with the server of this adaptation (`19adb6809`) and the launcher's defaults, on the same
test server with the SSD healthy (10.5-11.5 GiB/s before and after every server run). Only this version was
timed. Hermes ran on the Mac with thinking at `medium` and `TZ=UTC`, so the date in its system prompt stayed
the same through the runs.

The tasks ran in a directory holding a README and three Python files of 26, 16 and 40 lines. The prompts were
in Chinese, as a user typed them:

- **t1**: count the lines of each `.py` file in the current directory, write them to `summary.txt` from the most
  lines to the fewest, one `name lines` per line, and report the total.
- **t2**: append a `TOTAL` line with the total to `summary.txt`, then read the file back to confirm.

Each of three rounds started the server cold (empty prompt cache, the model files dropped from the page cache)
and ran:

| Step | Hermes configuration | Task | What it exercises |
| --- | --- | --- | --- |
| P1 | small toolset (`terminal,file,todo,memory,clarify`, 10 tools), title generation off | new session, t1 | first session on a cold server: the whole 7,108-token prompt |
| P2 | small toolset | P1 resumed, t2 | a resumed session |
| Q1 | small toolset | new session, t1 | a new session from the shared prefix in RAM |
| D1 | Hermes defaults (24 tools, title generation on) | new session, t1 | a new 13,761-token system prompt, then a title request that displaces the conversation |
| D2 | Hermes defaults | D1 resumed, t2 | the conversation loaded back from the prompt cache on disk |
| R1 | small toolset | server restarted with its prompt cache kept; new session, t1 | the shared prefix loaded from disk, expert caches empty |

TTFT is the time from a request to its first streamed data, recorded by a pass-through proxy on the server
machine (the server sends its headers with the first generated token). Medians of the three rounds, ranges in
parentheses:

| Request | Tokens processed / prompt | TTFT |
| --- | ---: | --- |
| P1, first request | 7,108 / 7,108 | 45.3 s (42.9-50.0) |
| Q1, first request | 44 / 7,108 | 3.5 s (3.3-4.1) |
| R1, first request | 44 / 7,108 | 5.5 s (5.1-8.3) |
| P2, first request | 24 / 7,417-7,954 | 2.6 s (2.5-4.1) |
| D1, first request | 13,761 / 13,761 | 108 s (106-114) |
| D2, first request | 24 / 14,174-14,456 | 2.4 s (2.4-3.0), after waiting 11.6 s for the title request |
| Later requests inside a task (after a tool result), 43 requests | 87 (27-225) / 7,234-14,875 | 4.8 s (2.2-9.1) |
| Title request at the end of D1 (not streamed: the whole response) | 274 / 274 | 15.1 s (14.9-15.6) |

- R1's first request processes the same 44 tokens as Q1's but is 2 s slower: after the restart the RAM and
  VRAM expert caches are empty. Loading the state from disk and the rest of the request handling took about
  0.1 s (TTFT minus the server's `prompt_ms`).
- Hermes sends the title request when D1 ends and exits without waiting for it. The script started D2 0.5 s
  later, and with one slot D2's first request waited for the title request to finish. A person reading the
  answer before typing would rarely meet this wait.
- The title request shares no prefix with the conversation and processes its 274 tokens from scratch in
  11.8 s; short prompts run at about 20 tokens/s here (18.7 for the later requests above, median). Hermes
  titled one resumed session again at
  the end of D2; that request started from the title prompt's saved prefix (23 of 253 tokens processed, 5.1 s).

JCT is the wall time of `hermes chat -q` on the Mac, from start to exit:

| Task | JCT | Requests | Sum of TTFT | Sum of generation | Tokens generated | Client and tools |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P1 | 120 s (91-157) | 4 | 56 s | 59 s | 380 | 4 s |
| P2 | 63 s (45-89) | 3 | 16 s | 44 s | 208 | 3 s |
| Q1 | 80 s (63-108) | 4 | 17 s | 59 s | 379 | 4 s |
| D1 | 166 s (156-181) | 3 | 121 s | 42 s | 284 | 4 s |
| D2 | 54 s (49-63) | 3 | 22 s, 11.6 s of it waiting | 30 s | 176 | 4 s |
| R1 | 76 s (70-108) | 4 | 17 s | 57 s | 327 | 4 s |

The columns after JCT are medians of the task's requests (the title request is not counted). "Sum of
generation" runs from each response's first token to its end, reasoning included, at 6.44 tokens/s on
average. "Client and tools" is JCT minus the requests' times: Hermes' start-up, tool runs and the SSH tunnel
from the Mac. Once the prefix is reused, generation takes 70-75% of a task. The model samples at temperature
1.0, so the length of its answers, and with it most of the spread in JCT, varies from round to round.

## Limits and open items

- The cost of the batch splits that post-batch checkpoints avoid was not measured; the timings above cover
  this version only.
- Hermes' title request takes about 15 s of server time per new session on the test server, and a request
  sent meanwhile waits for it. `auxiliary.title_generation.enabled: false` turns it off.
- The turn memory lives in the server process. After a restart, the first resumed turn of an old session
  falls back to the checkpoint before the last assistant turn and processes that turn again.
- A conversation that switches the reasoning effort processes the system prompt again: the effort line comes
  first in DeepSeek's encoding.
- A full state is about 5 KiB per token: V4.1 keeps all compressed rows in one cache sized for its finest
  ratio, and a saved state writes that many rows for every layer.
- The interactive Hermes interface was not tested; the sessions above used `hermes chat -q` with `--resume`,
  which sends the same requests.
