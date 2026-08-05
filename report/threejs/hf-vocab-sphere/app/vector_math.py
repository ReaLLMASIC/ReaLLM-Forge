from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

_MAX_AST_NODES = 160
_MAX_SCALAR_ABS = 1e9


@dataclass(frozen=True, slots=True)
class VectorExpressionResult:
    expression: str
    label: str
    alias: str
    vector: np.ndarray
    magnitude: float
    referenced_aliases: tuple[str, ...]


def alias_for_index(index: int) -> str:
    """Return Excel-style aliases: A..Z, AA..AZ, BA..."""
    if index < 0:
        raise ValueError("Alias index must be non-negative.")
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def alias_map(vectors: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("vectors must have shape [token_count, hidden_dim].")
    return {alias_for_index(index): values[index] for index in range(values.shape[0])}


@dataclass(slots=True)
class _Value:
    kind: str
    value: float | str | np.ndarray


def spherical_linear_interpolation(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate two vectors along the unit-hypersphere geodesic.

    Direction follows SLERP while magnitude is linearly interpolated. This
    preserves both endpoint vectors exactly and keeps the operation meaningful
    for embedding rows whose L2 norms differ.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    fraction = float(t)
    if not math.isfinite(fraction):
        raise ValueError("SLERP t must be a finite scalar.")
    norm_a = float(np.linalg.norm(first))
    norm_b = float(np.linalg.norm(second))
    if norm_a <= 1e-15 or norm_b <= 1e-15:
        raise ValueError("SLERP requires two non-zero vectors.")
    unit_a = first / norm_a
    unit_b = second / norm_b
    dot = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))

    if dot > 0.9995:
        direction = (1.0 - fraction) * unit_a + fraction * unit_b
        direction_norm = float(np.linalg.norm(direction))
        direction = direction / max(direction_norm, 1e-15)
    elif dot < -0.9995:
        axis_index = int(np.argmin(np.abs(unit_a)))
        basis = np.zeros_like(unit_a)
        basis[axis_index] = 1.0
        orthogonal = basis - float(np.dot(basis, unit_a)) * unit_a
        orthogonal /= max(float(np.linalg.norm(orthogonal)), 1e-15)
        direction = math.cos(math.pi * fraction) * unit_a + math.sin(math.pi * fraction) * orthogonal
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        direction = math.sin((1.0 - fraction) * angle) / sine * unit_a + math.sin(fraction * angle) / sine * unit_b
        direction /= max(float(np.linalg.norm(direction)), 1e-15)

    magnitude = (1.0 - fraction) * norm_a + fraction * norm_b
    return direction * magnitude


def vector_dimension_metrics(
    vector: np.ndarray, max_angle_degrees: float = 5.0, eps: float = 1e-12
) -> dict[str, float | int]:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    energy = values * values
    total = float(np.sum(energy))
    if total <= eps:
        return {
            "effective_dimension": 0.0,
            "dimensions_for_angle": 0,
            "continuous_dimensions_for_angle": 0.0,
            "retained_energy": 0.0,
            "resulting_angle_degrees": 0.0,
        }

    probabilities = energy / total
    square_sum = float(np.sum(probabilities * probabilities))
    effective_dimension = 1.0 / max(square_sum, eps)
    sorted_probabilities = np.sort(probabilities)[::-1]
    cumulative = np.cumsum(sorted_probabilities)
    angle = math.radians(float(max_angle_degrees))
    required_energy = math.cos(angle) ** 2
    k = int(np.searchsorted(cumulative, required_energy, side="left")) + 1
    k = min(max(k, 1), values.size)
    previous_energy = float(cumulative[k - 2]) if k > 1 else 0.0
    current_coordinate_energy = float(sorted_probabilities[k - 1])
    fractional_k = (k - 1) + (required_energy - previous_energy) / max(current_coordinate_energy, eps)
    fractional_k = max(1.0, min(float(k), float(fractional_k)))
    retained_energy = float(cumulative[k - 1])
    resulting_angle = math.degrees(math.acos(math.sqrt(min(1.0, retained_energy))))
    return {
        "effective_dimension": float(effective_dimension),
        "dimensions_for_angle": int(k),
        "continuous_dimensions_for_angle": float(fractional_k),
        "retained_energy": retained_energy,
        "resulting_angle_degrees": float(resulting_angle),
    }


ModelFunction = Callable[[str, list[float | str | np.ndarray]], np.ndarray]


class _SafeVectorEvaluator:
    def __init__(self, aliases: dict[str, np.ndarray], model_function: ModelFunction | None = None):
        self.aliases = {name.upper(): np.asarray(vector, dtype=np.float64) for name, vector in aliases.items()}
        self.model_function = model_function
        self.referenced: set[str] = set()

    def evaluate(self, expression: str) -> np.ndarray:
        text = str(expression or "").strip()
        if not text:
            raise ValueError("Vector expression cannot be blank.")
        if len(text) > 500:
            raise ValueError("Vector expression is too long (maximum 500 characters).")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            message = exc.msg or "invalid syntax"
            raise ValueError(f"Invalid vector expression: {message}.") from exc
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError("Vector expression is too complex.")
        result = self._visit(tree.body)
        if result.kind != "vector":
            raise ValueError("The expression must evaluate to a vector, not a scalar.")
        vector = np.asarray(result.value, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError("The resultant vector contains NaN or infinite values.")
        return vector

    def _visit(self, node: ast.AST) -> _Value:
        if isinstance(node, ast.Name):
            name = node.id.upper()
            vector = self.aliases.get(name)
            if vector is None:
                available = ", ".join(list(self.aliases)[:12])
                suffix = "…" if len(self.aliases) > 12 else ""
                raise ValueError(f"Unknown vector alias {node.id!r}. Available aliases: {available}{suffix}.")
            self.referenced.add(name)
            return _Value("vector", vector.copy())

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            scalar = float(node.value)
            if not math.isfinite(scalar) or abs(scalar) > _MAX_SCALAR_ABS:
                raise ValueError("Scalar constants must be finite and reasonably sized.")
            return _Value("scalar", scalar)

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip().casefold()
            if not text or len(text) > 40:
                raise ValueError("String arguments must be short, non-blank selectors.")
            return _Value("string", text)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._visit(node.operand)
            sign = 1.0 if isinstance(node.op, ast.UAdd) else -1.0
            return _Value(operand.kind, operand.value * sign)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only named vector functions are supported.")
            function_name = node.func.id.casefold()
            if node.keywords:
                raise ValueError("Vector functions accept positional arguments only.")
            if function_name in {"mean", "avg", "average"}:
                if not node.args:
                    raise ValueError(f"{node.func.id} requires at least one vector argument.")
                values = [self._visit(argument) for argument in node.args]
                if any(value.kind != "vector" for value in values):
                    raise ValueError(f"{node.func.id} accepts vector arguments only.")
                return _Value(
                    "vector",
                    np.mean(np.stack([np.asarray(value.value) for value in values], axis=0), axis=0),
                )
            if function_name in {"norm", "invnorm", "inverse_norm", "wov"}:
                if self.model_function is None:
                    raise ValueError(f"{node.func.id} requires model auxiliary tensors loaded by the projection API.")
                values = [self._visit(argument) for argument in node.args]
                args: list[float | str | np.ndarray] = [
                    np.asarray(value.value)
                    if value.kind == "vector"
                    else (str(value.value) if value.kind == "string" else float(value.value))
                    for value in values
                ]
                return _Value("vector", self.model_function(function_name, args))
            if function_name != "slerp":
                raise ValueError(
                    "Only mean(...), slerp(A, B, t), norm(...), invnorm(...), and wov(...) vector functions are supported."
                )
            if len(node.args) != 3:
                raise ValueError("slerp requires exactly three positional arguments: slerp(A, B, t).")
            first = self._visit(node.args[0])
            second = self._visit(node.args[1])
            fraction = self._visit(node.args[2])
            if first.kind != "vector" or second.kind != "vector" or fraction.kind != "scalar":
                raise ValueError("slerp requires two vectors followed by a scalar t.")
            return _Value(
                "vector",
                spherical_linear_interpolation(
                    np.asarray(first.value),
                    np.asarray(second.value),
                    float(fraction.value),
                ),
            )

        if isinstance(node, ast.BinOp):
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if left.kind != right.kind:
                    raise ValueError("Addition and subtraction require two vectors or two scalars.")
                sign = 1.0 if isinstance(node.op, ast.Add) else -1.0
                return _Value(left.kind, left.value + sign * right.value)

            if isinstance(node.op, ast.Mult):
                if left.kind == "vector" and right.kind == "vector":
                    raise ValueError("Vector-by-vector multiplication is not supported; multiply by a scalar instead.")
                if left.kind == "scalar" and right.kind == "scalar":
                    return _Value("scalar", float(left.value) * float(right.value))
                if left.kind == "vector":
                    return _Value("vector", np.asarray(left.value) * float(right.value))
                return _Value("vector", np.asarray(right.value) * float(left.value))

            if isinstance(node.op, ast.Div):
                if right.kind != "scalar":
                    raise ValueError("Division is only supported by a scalar denominator.")
                denominator = float(right.value)
                if abs(denominator) <= 1e-15:
                    raise ValueError("Division by zero is not allowed.")
                if left.kind == "scalar":
                    return _Value("scalar", float(left.value) / denominator)
                return _Value("vector", np.asarray(left.value) / denominator)

        raise ValueError(
            "Unsupported expression syntax. Use vector aliases, numeric scalars, parentheses, +, -, *, /, mean(...), slerp(A, B, t), norm(...), invnorm(...), and wov(...)."
        )


def evaluate_vector_expression(
    expression: str, aliases: dict[str, np.ndarray], model_function: ModelFunction | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    evaluator = _SafeVectorEvaluator(aliases, model_function=model_function)
    vector = evaluator.evaluate(expression)
    return vector, tuple(sorted(evaluator.referenced, key=lambda name: (len(name), name)))


def evaluate_vector_expressions(
    vectors: np.ndarray,
    expressions: Iterable[object],
    model_function: ModelFunction | None = None,
) -> list[VectorExpressionResult]:
    """Evaluate request-like objects with ``expression`` and optional ``label`` attributes."""
    aliases = alias_map(vectors)
    results: list[VectorExpressionResult] = []
    for index, item in enumerate(expressions):
        expression = str(getattr(item, "expression", "") or "").strip()
        requested_label = str(getattr(item, "label", "") or "").strip()
        vector, referenced = evaluate_vector_expression(expression, aliases, model_function=model_function)
        magnitude = float(np.linalg.norm(vector))
        if not math.isfinite(magnitude):
            raise ValueError(f"Resultant {index + 1} has a non-finite magnitude.")
        if magnitude <= 1e-12:
            raise ValueError(f"Resultant {index + 1} is zero or near-zero and has no direction to project.")
        alias = f"R{index + 1}"
        results.append(
            VectorExpressionResult(
                expression=expression,
                label=requested_label or alias,
                alias=alias,
                vector=vector,
                magnitude=magnitude,
                referenced_aliases=referenced,
            )
        )
    return results
