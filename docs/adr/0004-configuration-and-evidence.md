# 0004: Separate production configuration from historical evidence

Status: accepted.

The registry defines production keys, ranges, defaults, scope and units. The
Python package emits the same versioned settings contract and reads owned JSONL
telemetry. File/CLI provenance survives launch. It does not translate ambient
legacy environment variables or execute recorded launch comparisons.

Recorded reference artifacts used by tests remain byte-identical and retain
their original names. A smoke run is not a formal benchmark. Formal samples use
fixed prompts and interval definitions; failed samples are not silently retried
or replaced by the best iteration.
