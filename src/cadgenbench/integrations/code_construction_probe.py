"""Execute one formal solution in isolation and export one construction group.

The controller owns this module. User-authored ``solution.py`` only supplies
Python variable names; paths and output locations never come from the model.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path
from typing import Any

_CAPTURE_ASSIGNMENT_NAME = "__cadgenbench_capture_construction_assignment__"
_CAPTURE_USE_NAME = "__cadgenbench_capture_construction_use__"
_COMBINATION_METHODS = frozenset({"cut", "fuse", "intersect"})


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in target.elts))
    return set()


def assigned_names(tree: ast.AST) -> list[str]:
    """List source-level assignment names that could denote constructions."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(
            node,
            (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor),
        ):
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    names.update(_target_names(item.optional_vars))
    return sorted(
        name
        for name in names
        if name not in {_CAPTURE_ASSIGNMENT_NAME, _CAPTURE_USE_NAME}
        and name.isidentifier()
        and not name.startswith("__")
    )[:200]


class _CaptureTransformer(ast.NodeTransformer):
    """Capture actual geometry uses, with assignments retained as a fallback."""

    def __init__(self, selected: set[str]) -> None:
        self.selected = selected
        self.dynamic_depth = 0
        self.assignment_targets: set[str] = set()
        self.use_wrap_count = 0

    @staticmethod
    def _loaded_names(node: ast.AST) -> set[str]:
        return {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }

    def _selected_names(self, node: ast.AST) -> list[str]:
        return sorted(self.selected & self._loaded_names(node))

    def _can_capture_use(self, node: ast.AST) -> bool:
        loaded = self._loaded_names(node)
        return bool(self.selected & loaded) and not bool(
            self.assignment_targets & loaded
        )

    @staticmethod
    def _is_capture_use(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == _CAPTURE_USE_NAME
        )

    def _wrap_use(self, node: ast.expr) -> ast.expr:
        if self._is_capture_use(node) or not self._can_capture_use(node):
            return node
        selected_names = self._selected_names(node)
        if not selected_names:
            return node
        self.use_wrap_count += 1
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_CAPTURE_USE_NAME, ctx=ast.Load()),
                args=[
                    ast.List(
                        elts=[ast.Constant(value=name) for name in selected_names],
                        ctx=ast.Load(),
                    ),
                    node,
                ],
                keywords=[],
            ),
            node,
        )

    def _capture_assignment_statement(self, name: str) -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id=_CAPTURE_ASSIGNMENT_NAME, ctx=ast.Load()),
                args=[
                    ast.Constant(value=name),
                    ast.Name(id=name, ctx=ast.Load()),
                ],
                keywords=[],
            )
        )

    def _instrument_statement(
        self, node: ast.stmt, targets: set[str]
    ) -> ast.stmt | list[ast.stmt]:
        selected_targets = sorted(self.selected & targets)
        if self.dynamic_depth > 0 and selected_targets:
            return [
                node,
                *(
                    ast.copy_location(
                        self._capture_assignment_statement(name), node
                    )
                    for name in selected_targets
                ),
            ]
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        transformed = self.generic_visit(node)
        assert isinstance(transformed, ast.BinOp)
        if isinstance(transformed.op, (ast.Add, ast.Sub)):
            if not isinstance(transformed.left, ast.BinOp):
                transformed.left = self._wrap_use(transformed.left)
            transformed.right = self._wrap_use(transformed.right)
        return transformed

    def visit_Call(self, node: ast.Call) -> ast.AST:
        transformed = self.generic_visit(node)
        assert isinstance(transformed, ast.Call)
        if (
            isinstance(transformed.func, ast.Attribute)
            and transformed.func.attr in _COMBINATION_METHODS
        ):
            transformed.args = [self._wrap_use(argument) for argument in transformed.args]
        return transformed

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.dynamic_depth += 1
        result = self.generic_visit(node)
        self.dynamic_depth -= 1
        return result

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.dynamic_depth += 1
        result = self.generic_visit(node)
        self.dynamic_depth -= 1
        return result

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
        self.dynamic_depth += 1
        result = self.generic_visit(node)
        self.dynamic_depth -= 1
        return result

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST | list[ast.stmt]:
        self.dynamic_depth += 1
        result = self.generic_visit(node)
        self.dynamic_depth -= 1
        return result

    def visit_While(self, node: ast.While) -> ast.AST | list[ast.stmt]:
        self.dynamic_depth += 1
        result = self.generic_visit(node)
        self.dynamic_depth -= 1
        return result

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.stmt]:
        targets = set().union(*(_target_names(target) for target in node.targets))
        previous_targets = self.assignment_targets
        self.assignment_targets = targets
        wraps_before = self.use_wrap_count
        transformed = self.generic_visit(node)
        if self.use_wrap_count == wraps_before and self._can_capture_use(transformed.value):
            transformed.value = self._wrap_use(transformed.value)
        self.assignment_targets = previous_targets
        assert isinstance(transformed, ast.Assign)
        return self._instrument_statement(transformed, targets)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | list[ast.stmt]:
        targets = _target_names(node.target)
        previous_targets = self.assignment_targets
        self.assignment_targets = targets
        wraps_before = self.use_wrap_count
        transformed = self.generic_visit(node)
        if (
            transformed.value is not None
            and self.use_wrap_count == wraps_before
            and self._can_capture_use(transformed.value)
        ):
            transformed.value = self._wrap_use(transformed.value)
        self.assignment_targets = previous_targets
        assert isinstance(transformed, ast.AnnAssign)
        return self._instrument_statement(transformed, targets)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST | list[ast.stmt]:
        transformed = self.generic_visit(node)
        assert isinstance(transformed, ast.AugAssign)
        return self._instrument_statement(transformed, _target_names(transformed.target))

    def visit_With(self, node: ast.With) -> ast.AST | list[ast.stmt]:
        transformed = self.generic_visit(node)
        assert isinstance(transformed, ast.With)
        targets = set().union(
            *(
                _target_names(item.optional_vars)
                for item in transformed.items
                if item.optional_vars is not None
            ),
            set(),
        )
        return self._instrument_statement(transformed, targets)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST | list[ast.stmt]:
        transformed = self.generic_visit(node)
        assert isinstance(transformed, ast.AsyncWith)
        targets = set().union(
            *(
                _target_names(item.optional_vars)
                for item in transformed.items
                if item.optional_vars is not None
            ),
            set(),
        )
        return self._instrument_statement(transformed, targets)


def _shape_values(value: Any, shape_type: type[Any]) -> list[Any]:
    if isinstance(value, shape_type):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [
            shape
            for item in value
            for shape in _shape_values(item, shape_type)
        ]
    for attribute in ("part", "sketch", "line", "shape"):
        try:
            candidate = getattr(value, attribute)
        except (AttributeError, RuntimeError, ValueError):
            continue
        if isinstance(candidate, shape_type):
            return [candidate]
    return []


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def run_probe(
    source_path: Path,
    construction_names: list[str],
    output_path: Path,
    result_path: Path,
) -> int:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    available = assigned_names(tree)
    selected_names = list(dict.fromkeys(construction_names))
    base_result: dict[str, Any] = {
        "construction_names": selected_names,
        "available_names": available,
        "instance_count": 0,
        "instance_counts": {name: 0 for name in selected_names},
    }
    missing_names = [name for name in selected_names if name not in available]
    if missing_names:
        _write_result(
            result_path,
            {
                **base_result,
                "probe_status": "not_found",
                "reason": (
                    "the following requested names are not assigned in the current "
                    f"formal solution.py: {', '.join(missing_names)}"
                ),
            },
        )
        return 0

    import build123d

    assignment_captures: dict[str, list[Any]] = {
        name: [] for name in selected_names
    }
    use_events: list[tuple[set[str], list[Any]]] = []

    def copied_shapes(value: Any) -> list[Any]:
        copied: list[Any] = []
        for shape in _shape_values(value, build123d.Shape):
            try:
                copied.append(shape.copy())
            except (AttributeError, RuntimeError, ValueError):
                copied.append(shape)
        return copied

    def capture_assignment(name: str, value: Any) -> None:
        assignment_captures[name].extend(copied_shapes(value))

    def capture_use(names: list[str], value: Any) -> Any:
        shapes = copied_shapes(value)
        if shapes:
            use_events.append((set(names), shapes))
        return value

    transformed = _CaptureTransformer(set(selected_names)).visit(tree)
    ast.fix_missing_locations(transformed)
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(source_path),
        _CAPTURE_ASSIGNMENT_NAME: capture_assignment,
        _CAPTURE_USE_NAME: capture_use,
    }
    isolated_export = output_path.with_name("discarded-formal-output.step")
    real_export_step = build123d.export_step

    def redirected_export_step(*args: Any, **kwargs: Any) -> Any:
        positional = list(args)
        if len(positional) >= 2:
            positional[1] = str(isolated_export)
        else:
            for key in ("file_name", "file_path", "filename"):
                if key in kwargs:
                    kwargs[key] = str(isolated_export)
                    break
            else:
                raise RuntimeError("Formal export_step call has no output argument")
        return real_export_step(*positional, **kwargs)

    build123d.export_step = redirected_export_step
    try:
        importlib.import_module("build123d.exporters3d").export_step = redirected_export_step
    except (ImportError, AttributeError):
        pass
    sys.path.insert(0, str(source_path.parent))
    try:
        exec(  # noqa: S102 - execution occurs only in the isolated probe subprocess
            compile(transformed, str(source_path), "exec"), namespace, namespace
        )
    except Exception as exc:
        _write_result(
            result_path,
            {
                **base_result,
                "probe_status": "execution_error",
                "reason": f"isolated source execution failed: {type(exc).__name__}: {exc}",
                "instance_count": sum(len(shapes) for _, shapes in use_events),
            },
        )
        raise
    finally:
        sys.path.pop(0)

    captures: list[Any] = [
        shape
        for _, event_shapes in use_events
        for shape in event_shapes
    ]
    used_names = set().union(*(names for names, _ in use_events), set())
    instance_counts: dict[str, int] = {
        name: sum(
            len(event_shapes)
            for event_names, event_shapes in use_events
            if name in event_names
        )
        for name in selected_names
    }
    for name in selected_names:
        if name in used_names:
            continue
        fallback = assignment_captures[name]
        if not fallback and name in namespace:
            fallback = copied_shapes(namespace[name])
        captures.extend(fallback)
        instance_counts[name] = len(fallback)
    if not captures:
        _write_result(
            result_path,
            {
                **base_result,
                "probe_status": "not_renderable",
                "reason": "the requested variables exist but do not contribute a renderable build123d Shape, Part, or Builder",
            },
        )
        return 0

    selected = captures[0] if len(captures) == 1 else build123d.Compound(children=captures)
    try:
        real_export_step(selected, str(output_path))
    except Exception as exc:  # noqa: BLE001 - export failures become tool feedback
        _write_result(
            result_path,
            {
                **base_result,
                "probe_status": "not_renderable",
                "reason": f"the captured construction could not be exported: {type(exc).__name__}: {exc}",
                "instance_count": len(captures),
                "instance_counts": instance_counts,
            },
        )
        return 0
    _write_result(
        result_path,
        {
            **base_result,
            "probe_status": "renderable",
            "reason": "the requested construction group was captured at its downstream geometry uses and exported in isolation",
            "instance_count": len(captures),
            "instance_counts": instance_counts,
        },
    )
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--construction", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_probe(args.source, args.construction, args.output, args.result)


if __name__ == "__main__":
    raise SystemExit(main())
