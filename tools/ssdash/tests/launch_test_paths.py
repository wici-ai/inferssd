"""Canonicalize only already-normalized fixture roots across /var symlinks.

Mac resolves /var to /private/var; the frozen observer replaces the lexical
fixture root, so a resolved path becomes /private@case or /private@repo. Linux
has no such prefix. No real pathname, output value, argument, or source is erased.
"""


def canonical_roots(value):
    if isinstance(value, str):
        return value.replace("/private@case", "@case").replace("/private@repo", "@repo")
    if isinstance(value, list):
        return [canonical_roots(x) for x in value]
    if isinstance(value, dict):
        return {canonical_roots(k): canonical_roots(v) for k, v in value.items()}
    return value
