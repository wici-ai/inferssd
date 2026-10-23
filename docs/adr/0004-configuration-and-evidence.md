# 0004: Separate production configuration from historical evidence

Status: accepted for the private v0.1 candidate.

The registry defines production keys, ranges, defaults, scope and units. The
Python package emits the same versioned settings contract and reads owned JSONL
telemetry. File/CLI provenance survives launch. It does not translate ambient
legacy environment variables or execute recorded launch comparisons.

Historical golden artifacts remain byte-identical and retain their old names.
A private candidate smoke run is not a formal benchmark. Final samples retain
fixed prompts and interval definitions; failed samples are not silently retried
or replaced by the best iteration. Public visibility and a formal release need
separate owner authorization.
