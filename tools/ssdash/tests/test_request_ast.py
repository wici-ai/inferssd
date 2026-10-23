"""Whole commands and helpers retain executable F AST after explicit port extraction."""

from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest

ORACLE = Path(__file__).parent / "fixtures/oracle"


def dump(value):
    return ast.dump(value, include_attributes=False)


class Normalize(ast.NodeTransformer):
    names = {
        "Request": "urllib.request.Request",
        "transport": "urllib.request.urlopen",
        "clock_ns": "time.time_ns",
        "clock": "time.time",
        "open_file": "open",
        "emit": "print",
        "stat": "os.stat",
        "major": "os.major",
        "minor": "os.minor",
        "environ": "os.environ",
        "controller_factory": "Controller",
        "machine_factory": "Machine",
        "glob_paths": "glob.glob",
        "exists": "os.path.exists",
        "basename": "os.path.basename",
        "read_lines": "_lines",
    }

    def visit_Name(self, node):
        if node.id in self.names:
            return ast.parse(self.names[node.id], mode="eval").body
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "cast":
            return self.visit(node.args[1])
        if isinstance(node.func, ast.Name) and node.func.id == "drive_bytes":
            assert all(k.arg == "open_file" for k in node.keywords)
            node.keywords = []
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        # The library command retains original graph command termination. It
        # raises SystemExit directly instead of importing process sys globally.
        if dump(node) == dump(ast.parse("sys.exit(0)").body[0]):
            return ast.parse("raise SystemExit(0)").body[0]
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        assert dump(node) == dump(ast.parse("from machine import Machine").body[0])
        return None  # now the required factory parameter at this same try site

    def visit_ExceptHandler(self, node):
        if node.name == "error":
            assert dump(node.type) == dump(
                ast.parse("(OSError, KeyError, ValueError)", mode="eval").body
            )
            node.name = "e"
            for item in ast.walk(node):
                if isinstance(item, ast.Name) and item.id == "error":
                    item.id = "e"
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        node.args.kwonlyargs = []
        node.args.kw_defaults = []
        return self.generic_visit(node)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def command_after(tree, variable):
    return tree.body[
        next(
            i
            for i, n in enumerate(tree.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == variable
        ) :
    ]


def assert_same(old, new):
    norm = Normalize()
    assert dump(norm.visit(copy.deepcopy(old))) == dump(norm.visit(copy.deepcopy(new)))


@pytest.mark.parametrize("name", ["turn", "drive_bytes", "post"])
def test_request_helpers_ast(name):
    from ssdash.bootstrap import client, requests

    module = client if name == "turn" else requests
    original = "client" if name == "turn" else "boot_client"
    old = function(ast.parse((ORACLE / (original + "_f.py")).read_text()), name)
    new = function(ast.parse(Path(inspect.getsourcefile(module)).read_text()), name)
    assert_same(old, new)


@pytest.mark.parametrize(
    "original,candidate,start",
    [
        ("client", "run_client", "history"),
        ("boot_client", "run_bootstrap", "text"),
        ("converse", "run_conversation", "texts"),
        ("compare_online", "run_report", None),
    ],
)
def test_complete_command_ast(original, candidate, start):
    from ssdash.bootstrap import client, requests
    from ssdash.loop import conversation, report

    modules = {
        "client": client,
        "boot_client": requests,
        "converse": conversation,
        "compare_online": report,
    }
    old = ast.parse((ORACLE / (original + "_f.py")).read_text())
    new = function(ast.parse(Path(inspect.getsourcefile(modules[original])).read_text()), candidate)
    if original == "client":
        old = function(old, "main")
    if start:
        body = command_after(old, start)
    else:
        first = next(i for i, n in enumerate(old.body) if isinstance(n, ast.FunctionDef))
        body = old.body[first:]
        # The run directory becomes the explicit command argument.
        binding = ast.parse("d = sys.argv[1]").body[0]
        assert sum(dump(x) == dump(binding) for x in body) == 1
        body = [x for x in body if dump(x) != dump(binding)]
    assert_same(ast.Module(body=body, type_ignores=[]), ast.Module(body=new.body, type_ignores=[]))
