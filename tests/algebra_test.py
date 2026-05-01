import pytest
import jax.numpy as jnp
from eqxpress.primitives import Constant

# Adjust this import to match your project structure
from eqxpress.algebra import (
    Stack, Index, Mask, Reduce, Sum, Negate, Derivative, Flatness, Diagonal, OffDiagonal
)

# --- Dummy Objects for Testing ---

class DummyContext:
    """Mock context object for the Derivative test."""
    def __init__(self, step: float):
        self.f_scaled = step

# --- Tests ---

def test_stack_pytree():
    """Tests that Stack safely combines multiple PyTrees."""
    expr1 = Constant({"a": jnp.array([1.0]), "b": jnp.array([2.0])})
    expr2 = Constant({"a": jnp.array([3.0]), "b": jnp.array([4.0])})
    
    stacked = Stack(base_expression=(expr1, expr2), axis=0)
    result = stacked()
    
    assert jnp.allclose(result["a"], jnp.array([[1.0], [3.0]]))
    assert jnp.allclose(result["b"], jnp.array([[2.0], [4.0]]))

def test_index_pytree():
    """Tests that Index slices leaves of a PyTree correctly."""
    expr = Constant({"data": jnp.array([[1, 2], [3, 4], [5, 6]])})
    
    indexer = Index(base_expression=expr, indices=0)
    result = indexer()
    
    assert jnp.allclose(result["data"], jnp.array([1, 2]))

def test_mask():
    """Tests boolean masking mapped over a leading dimension."""
    data = jnp.array([
        [[1, 2], [3, 4]], 
        [[5, 6], [7, 8]]
    ])
    expr = Constant({"mat": data})
    
    mask = jnp.array([[True, True], [False, False]])
    
    masker = Mask(base_expression=expr, mask=mask)
    result = masker()
    
    expected = jnp.array([[1, 2], [5, 6]])
    assert jnp.allclose(result["mat"], expected)

def test_reduce():
    """Tests reduction operations over PyTrees."""
    expr = Constant({"data": jnp.array([[1.0, 2.0], [3.0, 4.0]])})
    
    reducer = Reduce(base_expression=expr, fn=jnp.max, axis=0)
    result = reducer()
    
    assert jnp.allclose(result["data"], jnp.array([3.0, 4.0]))

def test_sum_list_of_pytrees():
    """Tests that Sum safely handles PyTrees."""
    expr1 = Constant({"val": jnp.array(10.0)})
    expr2 = Constant({"val": jnp.array(20.0)})
    expr3 = Constant({"val": jnp.array(5.0)})
    
    summer = Sum(base_expression=[expr1, expr2, expr3])
    result = summer()
    
    assert result["val"] == 35.0

def test_negate_pytree():
    """Tests the explicit Negate AST node over a PyTree."""
    expr = Constant({"val1": jnp.array(10.0), "val2": jnp.array(-5.0)})
    
    negated = Negate(base_expression=expr)
    result = negated()
    
    assert result["val1"] == -10.0
    assert result["val2"] == 5.0

def test_derivative():
    """Tests numerical differentiation referencing a context argument."""
    data = jnp.array([0.0, 1.0, 4.0, 9.0])
    expr = Constant({"y": data})
    ctx = DummyContext(step=1.0)
    
    deriv = Derivative(base_expression=expr, step_attr="f_scaled", arg_index=1)
    
    result = deriv(None, ctx) 
    expected = jnp.gradient(data, 1.0)
    
    assert jnp.allclose(result["y"], expected)

def test_flatness():
    """Tests Flatness (Derivative subclass)."""
    expr = Constant(jnp.array([1.0, 2.0, 3.0]))
    ctx = DummyContext(step=1.0)
    
    flat = Flatness(base_expression=expr, step_attr="f_scaled", arg_index=1)
    result = flat(None, ctx)
    
    assert jnp.allclose(result, jnp.array([1.0, 1.0, 1.0]))

def test_diagonal():
    """Tests diagonal extraction."""
    data = jnp.array([
        [[1, 2], [3, 4]], 
        [[5, 6], [7, 8]]
    ])
    expr = Constant({"mat": data})
    
    diag = Diagonal(base_expression=expr)
    result = diag()
    
    expected = jnp.array([[1, 4], [5, 8]])
    assert jnp.allclose(result["mat"], expected)

def test_off_diagonal_class():
    """Tests the explicit OffDiagonal AST node class."""
    data = jnp.array([
        [[1, 2, 3], 
         [4, 5, 6],
         [7, 8, 9]]
    ])
    expr = Constant(data)
    
    off_diag = OffDiagonal(base_expression=expr, n_ports=3)
    result = off_diag()
    
    expected = jnp.array([[2, 3, 4, 6, 7, 8]])
    assert jnp.allclose(result, expected)