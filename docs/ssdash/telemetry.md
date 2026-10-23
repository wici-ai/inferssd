# Runtime JSONL v1

`ssd::telemetry::writer` owns one exclusively created UTF-8 JSONL file per
runtime/context. The context owner joins all users before destroying it. The
writer serializes each complete record under a mutex, flushes it, and assigns a
contiguous sequence beginning at zero. Invalid JSON, duplicate/reserved fields,
invalid event names, nonfinite numbers and malformed UTF-8 fail before
publication. I/O errors return status::io; after an I/O failure the stream stays
failed and refuses further publication. Callers must check every returned result.
No destructor exception is used as an error channel.

Each line has `v`, `ev` and `seq`, plus the event payload:

```json
{"v":1,"ev":"ssd.io.read","seq":1,"read_bytes":12345}
```

The first effective-settings publication uses `ev=ssd.settings`. Its `values`
object contains all fourteen canonical typed settings. Its matching `sources`
object contains `origin` (`cli|file|default`), `location` and the exact trailing
file `receipt`. The settings parser schema is the authority for value types,
units, bounds, producers and scopes. JSONL telemetry is distinct from the
human-readable launch log and is not parsed by searching old log markers.

`ssdash.telemetry.jsonl.read_events` rejects unsupported versions, malformed
JSON, duplicate keys, nonfinite values and sequence gaps/duplicates. Unknown
well-formed event kinds remain inspectable. `effective_settings` requires one
settings event, the complete fourteen-key value/receipt set and typed values.
It consumes actual C++ writer output in the cross-language contract test.

The only new-core environment observations are in `telemetry/debug.cpp`:
`SSDASH_DEBUG=trace,profile,mrc,ledger` and `SSDASH_TRACE_DIR`. They select output
channels/destination only, never placement, numerical kernels or policy. There
is no process-global mutable settings/telemetry owner. The old diagnostic
readers still present outside this foundation module require the separate
supported-profile retirement batch; this document does not imply they are gone.

The production integration contract is a `--ssd-telemetry FILE` option and
context-owned writer, with all publishers joined before its destruction. The
server/context connection is pending in this foundation batch. Defining this
writer and its Python consumer does not by itself certify every old
trace/profile/MRC emitter has been migrated. Numerical equality with channels enabled/disabled remains an
engine integration gate, not a conclusion of JSON protocol unit tests.
