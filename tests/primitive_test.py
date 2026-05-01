import operator
import pytest
import jax
import jax.numpy as jnp

# Adjust this import to match your project structure
from eqxpress.primitives import (
    Lambda,
    Constant,
    Binary,
    Where,
    Method,
    Map,
)

# --- Dummy Objects for Testing ---

class DummyModel:
    """A dummy class to test the Method node."""
    def __init__(self, state: float):
        self.state = state

    def predict(self, x: float) -> float:
        return self.state * x


# --- Tests ---

def test_lambda():
    """Tests that Lambda wraps and evaluates a standard callable."""
    expr = Lambda(fn=lambda x, y: x + y)
    
    assert expr(3, 4) == 7
    assert expr(x=10, y=20) == 30

def test_constant():
    """Tests that Constant ignores inputs and returns its value."""
    expr = Constant(value=42.0)
    
    assert expr() == 42.0
    # Should safely ignore any positional or keyword arguments passed during graph evaluation
    assert expr(1, 2, "ignore_me", kwarg1=True) == 42.0

def test_binary():
    """Tests Binary evaluation with both Expressions and raw constants."""
    # 1. Expression + Expression
    expr1 = Binary(fn=operator.add, left=Constant(10), right=Constant(5))
    assert expr1() == 15
    
    # 2. Expression + Constant
    expr2 = Binary(fn=operator.sub, left=Constant(10), right=3)
    assert expr2() == 7

    # 3. Constant + Expression
    expr3 = Binary(fn=operator.mul, left=4, right=Constant(5))
    assert expr3() == 20

def test_where():
    """Tests conditional branching via JAX lax.cond."""
    # Note: cond requires the branches to return the same shape/dtype.
    expr = Where(
        condition=Constant(True),
        true_branch=Constant(100.0),
        false_branch=Constant(0.0)
    )
    assert expr() == 100.0

    expr_false = Where(
        condition=Constant(False),
        true_branch=Constant(100.0),
        false_branch=Constant(0.0)
    )
    assert expr_false() == 0.0

    # Test with JAX tracing/jitting to ensure lax.cond behaves correctly under JIT
    jitted_expr = jax.jit(expr)
    assert jitted_expr() == 100.0

def test_method():
    """Tests dynamic method resolution and execution."""
    model = DummyModel(state=5.0)
    expr = Method(path="predict")
    
    # Expected: model.predict(10.0) -> 5.0 * 10.0 = 50.0
    result = expr(model, 10.0)
    assert result == 50.0
    
    # Test kwargs forwarding
    # We'll use a lambda attached to the object temporarily just for testing kwargs
    model.dynamic_method = lambda a, b=0: a + b
    expr_kwargs = Method(path="dynamic_method")
    assert expr_kwargs(model, 5, b=15) == 20

def test_method_missing_args():
    """Tests that Method raises a ValueError if no object is provided."""
    expr = Method(path="predict")
    
    with pytest.raises(ValueError, match="requires at least one positional argument"):
        expr()

def test_map():
    """Tests mapping a function over a base expression."""
    expr = Map(
        fn=lambda x: x ** 2,
        base_expression=Constant(4.0)
    )
    assert expr() == 16.0
    
    # Test with a raw constant instead of an Expression node
    expr_raw = Map(
        fn=lambda x: x + 10,
        base_expression=5.0
    )
    assert expr_raw() == 15.0

def test_composition():
    """Integration test checking that nodes can be composed deeply."""
    # Graph: (10 + 5) * (100 if True else 0)
    left_branch = Binary(fn=operator.add, left=Constant(10), right=Constant(5))
    right_branch = Where(
        condition=Constant(True), 
        true_branch=Constant(100.0), 
        false_branch=Constant(0.0)
    )
    
    root = Binary(fn=operator.mul, left=left_branch, right=right_branch)
    
    # (10 + 5) * 100.0 = 1500.0
    assert root() == 1500.0