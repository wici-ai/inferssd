# Security policy

## Deployment

The inference server binds to the loopback address by default and is intended to run on a trusted machine.
Do not expose it directly to an untrusted network. Load only model files and configuration files from
trusted sources.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/wici-ai/inferssd/security/advisories/new).

Do not open a public issue for a security problem, and do not post credentials, model data, deployment
addresses or exploit details publicly.

## Supported versions

InferSSD has no formal release yet. Security fixes land on `main`.

InferSSD bundles llama.cpp and ggml. Security advisories for those projects are assessed against the bundled
version, and applicable fixes are backported.
