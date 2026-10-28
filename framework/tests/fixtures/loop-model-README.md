# Controller and observation transcripts

These 60 synthetic scenarios execute the online module and Controller class of
the recorded reference revision (see the
[framework README](../../README.md#compatibility-and-failure-behavior)), with its
original Machine, replay, price, prefill and log parser call graph. The loop
sources are bound in `oracle/loop-manifest.json`; the other reference modules and
input model artifacts are bound by their own manifests in this directory.

Each transcript records ordered file/clock/engine/HTTP/output calls, exact float values and mapping order, returned aliases, Machine calls and mutable state, partial output, and partial constructor state after a failure. Python 3.9 and 3.12 expectations are generated independently from the reference. Test engines, HTTP, clock and in-memory files are effects only; no process, request, host probe, model or device runs. Gzip is a logical input stream in this observer; actual gzip behavior is covered by the existing recorded-log tests.

The scenarios include a real model switch with a 5000-token **forecast input**, not an executed request; successful/caught-OSError/uncaught-RuntimeError displacement; no-history, failed replacement, stop and clock faults; repeated prepare aliases and explicit close; partial live lines, truncation, gzip cache keys/identity; ragged traces, matching graph/turn records, and slope/feedback/rounded decision boundaries. Controlled model responses are used only in the named `decide-*` threshold cases.

One `ObservationSession` corresponds to one fresh instance of the reference online module. Module globals are not shared across scenarios. Independent target sessions are an explicit ownership change; no cross-session equivalence to the reference module's hidden global cache is claimed. In the reference, `close()` only stops the engine; this observer records that the event stream remains open. Implicit fixture destruction is outside the transcript and is not a production lifecycle promise.
