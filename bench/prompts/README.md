# Fixed benchmark text

These two prompts are excerpts from an invented English design document about a
rail telemetry store named Tessaline. The organizations, locations, incidents and
performance figures inside the text are fictional benchmark content, not project
claims. Both excerpts end with the same summarization instruction.

The filenames describe approximate source-text lengths, not rendered model token
counts. Keep the exact UTF-8 bytes, including the final newline. Verify them with
`sha256sum -c SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS` on macOS) from this
directory. Record the actual token IDs and count returned by the tested server's
chat template; they depend on model tokenizer, template and reasoning settings.

`prompt-512.txt` is the short comparison input. `prompt-4096.txt` is retained as a
versioned longer input for explicitly scheduled device acceptance. The short ABBA
driver and client have a 1,024-input-token limit, so they intentionally reject the
long input. Keeping the file does not schedule a run or waive that limit.

Use one user message, reasoning disabled, greedy decoding, a fixed output length,
EOS ignored and prompt caching disabled. A fresh process clears its application
caches but does not make the operating system's file cache cold. Record any OS
cache treatment explicitly and apply it equally to both comparison arms. Preserve
all attempts, including timeout, OOM and incomplete runs; see the parent benchmark
README for the result and ownership contracts.
