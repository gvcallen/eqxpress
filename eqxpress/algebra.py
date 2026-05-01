"""
Algebraic nodes for array manipulations.

Exported at root.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import operator
from functools import reduce
from typing import Any, Callable, Union
import equinox as eqx

from eqxpress.base import AbstractExpression, ExprInputs, ExprOutputs, tree_op
from eqxpress.primitives import Map


class Stack(AbstractExpression[ExprInputs, ExprOutputs]):
    """Stacks the results of multiple AbstractExpressions along an axis."""
    base_expression: tuple[AbstractExpression, ...]
    axis: int = eqx.field(default=-1, static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        results = [op(*args, **kwargs) for op in self.base_expression]
        # Maps jnp.stack element-wise across the leaves of all trees
        return jtu.tree_map(lambda *leaves: jnp.stack(leaves, axis=self.axis), *results)

class Index(AbstractExpression[ExprInputs, ExprOutputs]):
    """Slices or indexes the output of another AbstractExpression."""
    base_expression: AbstractExpression
    indices: Any = eqx.field(static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.base_expression(*args, **kwargs)
        return jtu.tree_map(lambda x: x[self.indices], data)

class Mask(AbstractExpression[ExprInputs, ExprOutputs]):
    """Applies a boolean mask to the output of an AbstractExpression."""
    base_expression: AbstractExpression
    mask: jnp.ndarray

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.base_expression(*args, **kwargs)
        # vmap ensures we apply the mask correctly across lead dimensions (e.g. freq)
        mask_fn = jax.vmap(lambda x: x[self.mask])
        return jtu.tree_map(mask_fn, data)
    
class Reduce(AbstractExpression[ExprInputs, ExprOutputs]):
    """Applies a reduction (e.g., jnp.max, jnp.mean) over a specific axis."""
    base_expression: AbstractExpression
    fn: Callable = eqx.field(static=True)
    axis: Union[int, tuple[int, ...], None] = eqx.field(default=None, static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.base_expression(*args, **kwargs)
        return jtu.tree_map(lambda x: self.fn(x, axis=self.axis), data)

class Sum(AbstractExpression[ExprInputs, ExprOutputs]):
    """Evaluates multiple AbstractExpressions and sums their outputs together."""
    base_expression: tuple[AbstractExpression, ...] | list[AbstractExpression]

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        results = [op(*args, **kwargs) for op in self.base_expression]
        # functools.reduce safely adds multiple PyTrees together element-wise
        return reduce(lambda x, y: jtu.tree_map(operator.add, x, y), results)

class Negate(Map):
    """
    Applies element-wise negation to a single AbstractExpression's output.
    """
    def __init__(self, base_expression: AbstractExpression):
        self.base_expression = base_expression
        # Wrap operator.neg in tree_op to ensure PyTree safety just like other Maps
        self.fn = tree_op(operator.neg)

class Derivative(AbstractExpression[ExprInputs, ExprOutputs]):
    """Computes numerical derivative with respect to a context attribute."""
    base_expression: AbstractExpression
    step_attr: str = eqx.field(static=True)
    axis: int = eqx.field(default=0, static=True)
    order: int = eqx.field(default=1, static=True)
    arg_index: int = eqx.field(default=1, static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.base_expression(*args, **kwargs)
        dx = getattr(args[self.arg_index], self.step_attr)
        
        def _compute_grad(x):
            for _ in range(self.order):
                x = jnp.gradient(x, dx, axis=self.axis)
            return x
            
        return jtu.tree_map(_compute_grad, data)

class Flatness(Derivative):
    """Enforces gain flatness by computing the first derivative."""
    order: int = eqx.field(default=1, static=True)
    
class Diagonal(AbstractExpression[ExprInputs, ExprOutputs]):
    """Extracts the diagonals of matrices."""
    base_expression: AbstractExpression

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.base_expression(*args, **kwargs)
        return jtu.tree_map(jax.vmap(jnp.diag), data)

class OffDiagonal(Mask):
    """Extracts off-diagonal elements as a dedicated AST node."""
    def __init__(self, base_expression: AbstractExpression, n_ports: int, **kwargs):
        self.base_expression = base_expression
        # Compute the mask dynamically during initialization
        self.mask = ~jnp.eye(n_ports, dtype=bool)