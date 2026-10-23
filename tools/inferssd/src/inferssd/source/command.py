"""Opt-in source artifact commands; argv parsing is the new consumer boundary."""

from __future__ import annotations

import argparse
import random
from collections.abc import Callable

from inferssd.source import experts
from inferssd.source.experts import RandomChoice
from inferssd.source.files import LOCAL_FILES, SourceFiles


def add_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    source = commands.add_parser("source", help="read model headers and produce source artifacts")
    kinds = source.add_subparsers(dest="source_command", required=True)
    index = kinds.add_parser("index", help="create or reuse a legacy expert index")
    index.add_argument("model")
    index.add_argument("pack", nargs="?", default=None)
    index.add_argument("directory")
    facts = kinds.add_parser("facts", help="print legacy source shell assignments")
    facts.add_argument("model")
    facts.add_argument("alias", nargs="?", default=None)
    lane = kinds.add_parser("lane-args", help="print lane shape/type arguments; do not run a probe")
    lane.add_argument("directory")
    dense = kinds.add_parser("dense", help="write a legacy dense placement alias index")
    dense.add_argument("model")
    dense.add_argument("homes")
    dense.add_argument("out")
    iolog = kinds.add_parser("iolog", help="write fio input text; do not execute fio or reads")
    iolog.add_argument("directory")
    iolog.add_argument("n", type=int)
    iolog.add_argument("seed", type=int)
    iolog.add_argument("out")
    iolog.add_argument("file", nargs="?", default=None)


def run_source(
    args: argparse.Namespace,
    *,
    filesystem: SourceFiles = LOCAL_FILES,
    random_factory: Callable[[int], RandomChoice] = random.Random,
    emit: Callable[..., None] = print,
) -> None:
    """Dispatch accepted arguments to the same source operations as F."""
    if args.source_command == "index":
        experts.index(args.model, args.pack, args.directory, filesystem=filesystem, emit=emit)
    elif args.source_command == "facts":
        experts.facts(args.model, args.alias, filesystem=filesystem, emit=emit)
    elif args.source_command == "lane-args":
        experts.lane_args(args.directory, filesystem=filesystem, emit=emit)
    elif args.source_command == "dense":
        experts.dense(args.model, args.homes, args.out, filesystem=filesystem, emit=emit)
    elif args.source_command == "iolog":
        experts.iolog(
            args.directory,
            args.n,
            args.seed,
            args.out,
            args.file,
            filesystem=filesystem,
            random_factory=random_factory,
        )
    else:
        raise ValueError(f"unknown source command: {args.source_command}")
