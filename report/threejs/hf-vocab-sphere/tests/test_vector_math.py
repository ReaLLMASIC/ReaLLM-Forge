from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.vector_math import (
    alias_for_index,
    evaluate_vector_expression,
    evaluate_vector_expressions,
    vector_dimension_metrics,
)


def test_aliases_are_excel_style() -> None:
    assert [alias_for_index(i) for i in (0, 25, 26, 27, 51, 52)] == ["A", "Z", "AA", "AB", "AZ", "BA"]


def test_vector_arithmetic_and_average() -> None:
    aliases = {
        "A": np.array([4.0, 0.0, 0.0]),
        "B": np.array([1.0, 2.0, 0.0]),
        "C": np.array([1.0, 1.0, 3.0]),
        "D": np.array([0.0, 0.0, 2.0]),
    }
    result, references = evaluate_vector_expression("(A - B) + C", aliases)
    np.testing.assert_allclose(result, np.array([4.0, -1.0, 3.0]))
    assert references == ("A", "B", "C")

    average, _ = evaluate_vector_expression("(A + B + C) / 3 + D", aliases)
    np.testing.assert_allclose(average, (aliases["A"] + aliases["B"] + aliases["C"]) / 3 + aliases["D"])

    helper_average, helper_references = evaluate_vector_expression("mean(A, B, C) + D", aliases)
    np.testing.assert_allclose(helper_average, average)
    assert helper_references == ("A", "B", "C", "D")


def test_expression_results_receive_labels() -> None:
    vectors = np.eye(3)
    rows = evaluate_vector_expressions(
        vectors,
        [SimpleNamespace(expression="A - B + C", label="analogy")],
    )
    assert rows[0].alias == "R1"
    assert rows[0].label == "analogy"
    assert rows[0].magnitude == pytest.approx(np.sqrt(3.0))


def test_slerp_interpolates_direction_and_magnitude() -> None:
    aliases = {
        "A": np.array([2.0, 0.0, 0.0]),
        "B": np.array([0.0, 4.0, 0.0]),
    }
    midpoint, references = evaluate_vector_expression("slerp(A, B, 0.5)", aliases)
    expected_direction = np.array([2**-0.5, 2**-0.5, 0.0])
    np.testing.assert_allclose(midpoint, expected_direction * 3.0, atol=1e-8)
    assert references == ("A", "B")

    start, _ = evaluate_vector_expression("slerp(A, B, 0)", aliases)
    end, _ = evaluate_vector_expression("slerp(A, B, 1)", aliases)
    np.testing.assert_allclose(start, aliases["A"])
    np.testing.assert_allclose(end, aliases["B"])


def test_slerp_allows_finite_extrapolation() -> None:
    aliases = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    result, references = evaluate_vector_expression("slerp(A, B, -0.5)", aliases)
    assert references == ("A", "B")
    assert np.all(np.isfinite(result))


def test_unsafe_vector_math_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_vector_expression("__import__('os').system('echo nope')", {"A": np.ones(3)})
    with pytest.raises(ValueError):
        evaluate_vector_expression("A * A", {"A": np.ones(3)})


def test_zero_resultant_is_rejected() -> None:
    vectors = np.eye(2)
    with pytest.raises(ValueError, match="zero or near-zero"):
        evaluate_vector_expressions(
            vectors,
            [SimpleNamespace(expression="A - A", label="zero")],
        )


def test_model_functions_can_be_nested() -> None:
    aliases = {"A": np.array([1.0, 2.0])}
    calls: list[str] = []

    def model_function(name, args):
        calls.append(name)
        return np.asarray(args[0]) * (2.0 if name == "norm" else 0.5)

    result, references = evaluate_vector_expression(
        "invnorm(norm(A, 0, 'attn', 'input'), 'final')", aliases, model_function=model_function
    )
    np.testing.assert_allclose(result, aliases["A"])
    assert references == ("A",)
    assert calls == ["norm", "invnorm"]


def test_vector_dimension_metrics_participation_ratio() -> None:
    assert vector_dimension_metrics(np.array([1.0, 0.0, 0.0]))["effective_dimension"] == pytest.approx(1.0)
    assert vector_dimension_metrics(np.array([1.0, 1.0, 1.0]))["effective_dimension"] == pytest.approx(3.0)
    skewed = vector_dimension_metrics(np.array([10.0, 1.0, 0.0]))
    assert skewed["effective_dimension"] == pytest.approx((101.0**2) / 10001.0)
    assert skewed["dimensions_for_angle"] >= 1
