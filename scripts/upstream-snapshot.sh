#!/bin/sh
# SPDX-License-Identifier: MIT
# Run on upstream; its history must remain an ancestor of main.
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: $0 <llama.cpp-commit>" >&2
    exit 2
fi
test -z "$(git status --porcelain)" || {
    echo "refusing to replace a dirty working tree" >&2
    exit 1
}
revision=$(git rev-parse --verify "$1^{commit}")
set -- . ':(exclude).github' ':(exclude)AGENTS.md' ':(exclude)CLAUDE.md' \
    ':(exclude)CODEOWNERS' ':(exclude)CONTRIBUTING.md' ':(exclude)SECURITY.md' \
    ':(exclude)README.md' ':(exclude).pre-commit-config.yaml' \
    ':(exclude).gemini' ':(exclude).pi'
git rm -rq --ignore-unmatch -- "$@"
git checkout "$revision" -- "$@"
git commit -m "upstream: llama.cpp $revision"
