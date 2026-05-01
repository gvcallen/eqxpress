import operator
import pytest
import jax.numpy as jnp
from typing import Any

# Adjust this import to match your project structure
from eqxpress import AbstractExpression

# --- 1. Setup: Concrete Dummy Expression for Testing ---

class DummyExpr(AbstractExpression[..., Any]):
    """A minimal concrete implementation for testing the base class."""
    value: Any
    
    def __call__(self, *args, **kwargs) -> Any:
        return self.value

# --- 2. Tests ---

def test_cannot_instantiate_base_class():
    """Ensures the abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractExpression()

@pytest.mark.parametrize("op_fn, val1, val2, expected", [
    (operator.add, 10.0, 2.0, 12.0),       # __add__
    (operator.sub, 10.0, 2.0, 8.0),        # __sub__
    (operator.mul, 10.0, 2.0, 20.0),       # __mul__
    (operator.truediv, 10.0, 2.0, 5.0),    # __truediv__
    (operator.pow, 10.0, 2.0, 100.0),      # __pow__
])
def test_arithmetic_operators(op_fn, val1, val2, expected):
    """Tests standard forward arithmetic operators."""
    expr1 = DummyExpr(val1)
    expr2 = DummyExpr(val2)
    
    composed_expr = op_fn(expr1, expr2)
    
    # 1. Verify it returns a deferred expression node
    assert isinstance(composed_expr, AbstractExpression)
    
    # 2. Verify the mathematical evaluation is correct
    assert composed_expr() == expected

@pytest.mark.parametrize("op_fn, scalar, val, expected", [
    (lambda s, e: s + e, 5.0, 10.0, 15.0),   # __radd__
    (lambda s, e: s - e, 5.0, 10.0, -5.0),   # __rsub__
    (lambda s, e: s * e, 5.0, 10.0, 50.0),   # __rmul__
    (lambda s, e: s / e, 50.0, 10.0, 5.0),   # __rtruediv__
])
def test_reverse_arithmetic_operators(op_fn, scalar, val, expected):
    """Tests reverse arithmetic (e.g., scalar + Expression)."""
    expr = DummyExpr(val)
    
    composed_expr = op_fn(scalar, expr)
    
    assert isinstance(composed_expr, AbstractExpression)
    assert composed_expr() == expected

def test_unary_negation():
    """Tests the __neg__ operator."""
    expr = DummyExpr(10.0)
    
    composed_expr = -expr
    
    assert isinstance(composed_expr, AbstractExpression)
    assert composed_expr() == -10.0

@pytest.mark.parametrize("op_fn, val1, val2, expected", [
    (operator.gt, 10.0, 2.0, True),    # __gt__
    (operator.gt, 2.0, 10.0, False),
    (operator.lt, 2.0, 10.0, True),    # __lt__
    (operator.ge, 10.0, 10.0, True),   # __ge__
    (operator.le, 10.0, 2.0, False),   # __le__
])
def test_comparison_operators(op_fn, val1, val2, expected):
    """Tests comparison operators."""
    expr1 = DummyExpr(val1)
    expr2 = DummyExpr(val2)
    
    composed_expr = op_fn(expr1, expr2)
    
    assert isinstance(composed_expr, AbstractExpression)
    assert composed_expr() == expected

def test_pytree_evaluation():
    """
    Tests that operations successfully map over PyTrees, 
    as promised by the base class docstring.
    """
    expr1 = DummyExpr({"a": jnp.array([1.0, 2.0]), "b": 5.0})
    expr2 = DummyExpr({"a": jnp.array([3.0, 4.0]), "b": 2.0})
    
    composed_expr = expr1 + expr2
    result = composed_expr()
    
    # Check that tree structure is preserved
    assert isinstance(result, dict)
    assert "a" in result and "b" in result
    
    # Check the math was mapped element-wise correctly
    assert jnp.allclose(result["a"], jnp.array([4.0, 6.0]))
    assert result["b"] == 7.0