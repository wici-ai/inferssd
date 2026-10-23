"""Two helpers and two complete command bodies after explicit port/type adapters."""

import ast
import copy
import inspect

import pytest
from test_bootstrap_records import ORACLE

from ssdash.bootstrap import core_sets, tier_slots


class Normalize(ast.NodeTransformer):
    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_Name(self, node):
        node.id = {"open_file": "open", "emit": "print", "cpu_groups": "groups"}.get(
            node.id, node.id
        )
        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr == "argv":
            return ast.Name(id="argv", ctx=ast.Load())
        if isinstance(node.value, ast.Name) and node.value.id == "gzip" and node.attr == "open":
            return ast.Name(id="open_gzip", ctx=ast.Load())
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "cast":
            return self.visit(node.args[1])
        return self.generic_visit(node)


def body(source, function):
    tree = ast.parse(source)
    if function is None:
        return tree.body[next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)) :]
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    statements = node.body
    if isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        statements = statements[1:]
    return statements


@pytest.mark.parametrize(
    "module,original_name,candidate_name",
    [
        (core_sets, "parse_list", "parse_list"),
        (core_sets, None, "run_coresets"),
        (tier_slots, "reached", "reached"),
        (tier_slots, "main", "run_ubprobe"),
    ],
)
def test_complete_executable_body(module, original_name, candidate_name):
    original = (ORACLE / ("coresets_f.py" if module is core_sets else "ubprobe_f.py")).read_text()
    expected = ast.Module(body=body(original, original_name), type_ignores=[])
    actual = ast.Module(body=body(inspect.getsource(module), candidate_name), type_ignores=[])
    assert ast.dump(Normalize().visit(copy.deepcopy(expected))) == ast.dump(
        Normalize().visit(actual)
    )


def test_tier_regex_constant_is_original_F():
    original = ast.parse((ORACLE / "ubprobe_f.py").read_text())
    expression = next(n.value for n in original.body if isinstance(n, ast.Assign))
    assert tier_slots.TIER.pattern == ast.literal_eval(expression.args[0])
