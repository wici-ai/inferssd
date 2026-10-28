# Pre-extraction reference fragments

These fragments were generated before the shared-provider changes from the exact
source commit/blob/SHA256 in extraction.json. They stay fixed: changing the
production owner or CUDA initializer must not regenerate the comparison
reference from that new implementation. The member fragment differs from the
original only by removing the leading static on function declarations. This is
an extraction and ownership reference, not an independent parser implementation.
