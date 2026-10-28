#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Create a prefixed comparison tree without changing a branch or the worktree.
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: $0 <llama.cpp-commit>" >&2
    exit 2
fi
revision=$(git rev-parse --verify "$1^{commit}")
index=$(mktemp)
trap 'rm -f "$index"' EXIT HUP INT TERM
rm -f "$index"
GIT_INDEX_FILE="$index" git read-tree --empty
GIT_INDEX_FILE="$index" git read-tree --prefix=engine/ "$revision"
GIT_INDEX_FILE="$index" git ls-files -z -- engine/.github engine/AGENTS.md engine/CLAUDE.md \
    engine/CODEOWNERS engine/CONTRIBUTING.md engine/SECURITY.md engine/README.md \
    engine/.pre-commit-config.yaml engine/.gemini engine/.pi |
    GIT_INDEX_FILE="$index" git update-index --force-remove -z --stdin
GIT_INDEX_FILE="$index" git write-tree
