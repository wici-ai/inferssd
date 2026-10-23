"""Executable AST equivalence with the explicitly approved ownership/port changes."""

from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import pytest

ORACLE = Path(__file__).parent / "fixtures/oracle"


def dump(node):
    return ast.dump(node, include_attributes=False)


class Normalize(ast.NodeTransformer):
    """Only type views, session storage and named effect call sites may change."""

    replacements = {
        "session.files.exists": "os.path.exists",
        "session.files.open": "open",
        "session.read_arm_cached": "_read_arm_cached",
        "self.files.exists": "os.path.exists",
        "self.files.stat": "os.stat",
        "self.arms": "_ARMS",
        "self.live": "_LIVE",
        "self._effects.getenv": "os.environ.get",
        "self._effects.makedirs": "os.makedirs",
        "self._effects.open": "open",
        "self._effects.clock": "time.time",
        "self._engine_factory": "Engine",
        "self._post": "post",
    }

    def visit_Attribute(self, node):
        name = ast.unparse(node)
        if name in self.replacements:
            return ast.parse(self.replacements[name], mode="eval").body
        return self.generic_visit(node)

    def visit_Name(self, node):
        names = {"machine_factory": "Machine", "prefill_feedback": "prefill.feedback"}
        if node.id in names:
            return ast.parse(names[node.id], mode="eval").body
        if node.id == "layer":  # isolated comprehension binding, not a record key
            node.id = "l"
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "cast":
            return self.visit(node.args[1])
        if ast.unparse(node.func) == "self.files.open_binary":
            node.func = ast.Name(id="open", ctx=ast.Load())
            node.args.append(ast.Constant(value="rb"))
        name = ast.unparse(node.func)
        if name == "read_arm":
            node.keywords = [k for k in node.keywords if k.arg != "files"]
        if name == "online.served":
            node.keywords = [k for k in node.keywords if k.arg != "session"]
        if name == "print":
            node.keywords = [k for k in node.keywords if k.arg != "file"]
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_FunctionDef(self, node):
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        node.args.kwonlyargs = []
        node.args.kw_defaults = []
        if node.name == "read_arm_cached":
            node.name = "_read_arm_cached"
            node.args.args = node.args.args[1:]
        if node.body and isinstance(node.body[0], ast.Expr):
            expr = node.body[0].value
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                node.body.pop(0)
        if node.name == "__init__" and node.body:
            # These two assignments only store the caller-owned ports. Check
            # their exact form before removing them from the F comparison.
            expected = [
                "self._engine_factory, self._post = engine_factory, post",
                "self._effects, self._session = effects, session",
            ]
            if dump(node.body[0]) == dump(ast.parse(expected[0]).body[0]):
                assert [dump(x) for x in node.body[:2]] == [
                    dump(ast.parse(x).body[0]) for x in expected
                ]
                del node.body[:2]
        return self.generic_visit(node)


def functions(path, cls=None):
    tree = ast.parse(Path(path).read_text())
    nodes = (
        tree.body
        if cls is None
        else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    )
    return {n.name: n for n in nodes if isinstance(n, ast.FunctionDef)}


def assert_equal(original, candidate):
    normalize = Normalize()
    assert dump(normalize.visit(copy.deepcopy(candidate))) == dump(
        normalize.visit(copy.deepcopy(original))
    )


@pytest.mark.parametrize("name", ["served", "fit_slope", "feedback", "expect", "decide"])
def test_online_calculation_ast(name):
    from ssdash.loop import model

    assert_equal(
        functions(ORACLE / "online_f.py")[name],
        functions(inspect.getsourcefile(model))[name],
    )


def test_session_read_ast():
    from ssdash.loop import records

    assert_equal(
        functions(ORACLE / "online_f.py")["_read_arm_cached"],
        functions(inspect.getsourcefile(records), "ObservationSession")["read_arm_cached"],
    )


@pytest.mark.parametrize(
    "name",
    [
        "__init__",
        "note",
        "_start",
        "config",
        "before",
        "ensure",
        "prepare",
        "context",
        "after",
        "close",
    ],
)
def test_controller_transition_ast(name):
    from ssdash.loop import controller

    assert_equal(
        functions(ORACLE / "controller_f.py", "Controller")[name],
        functions(inspect.getsourcefile(controller), "Controller")[name],
    )
