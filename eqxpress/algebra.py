"""
Algebraic nodes for array manipulations.

Exported at root.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Union
import equinox as eqx

from eqxpress.base import AbstractExpression, ExprInputs, ExprOutputs, tree_op
from eqxpress.nodes import Map

class Stack(AbstractExpression[ExprInputs, ExprOutputs]):
    """Stacks the results of multiple AbstractExpressions along an axis."""
    AbstractExpressions: tuple[AbstractExpression, ...]
    axis: int = eqx.field(default=-1, static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        results = [op(*args, **kwargs) for op in self.AbstractExpressions]
        return jnp.stack(results, axis=self.axis)

class Index(AbstractExpression[ExprInputs, ExprOutputs]):
    """Slices or indexes the output of another AbstractExpression."""
    AbstractExpression: AbstractExpression
    indices: Any = eqx.field(static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        return self.AbstractExpression(*args, **kwargs)[self.indices]

class Mask(AbstractExpression[ExprInputs, ExprOutputs]):
    """Applies a boolean mask to the output of an AbstractExpression."""
    AbstractExpression: AbstractExpression
    mask: jnp.ndarray

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.AbstractExpression(*args, **kwargs)
        # vmap ensures we apply the mask correctly across lead dimensions (e.g. freq)
        return jax.vmap(lambda x: x[self.mask])(data)
    
class Reduce(AbstractExpression[ExprInputs, ExprOutputs]):
    """Applies a reduction (e.g., jnp.max, jnp.mean) over a specific axis."""
    AbstractExpression: AbstractExpression
    fn: Callable = eqx.field(static=True)
    axis: Union[int, tuple[int, ...], None] = eqx.field(default=None, static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        return self.fn(self.AbstractExpression(*args, **kwargs), axis=self.axis)

class Sum(AbstractExpression[ExprInputs, ExprOutputs]):
    """Evaluates multiple AbstractExpressions and sums their outputs together."""
    AbstractExpressions: tuple[AbstractExpression, ...] | list[AbstractExpression]

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        results = [op(*args, **kwargs) for op in self.AbstractExpressions]
        return sum(results[1:], start=results[0])
    
class Negate(Map):
    """
    Applies an arbitrary function to a single AbstractExpression's output.
    """
    def __init__(self, other):
        return super().__init__(fn=tree_op(AbstractExpression.neg), AbstractExpression=other)

class Derivative(AbstractExpression[ExprInputs, ExprOutputs]):
    """Computes numerical derivative with respect to a context attribute."""
    AbstractExpression: AbstractExpression
    step_attr: str = eqx.field(static=True)
    axis: int = eqx.field(default=0, static=True)
    order: int = eqx.field(default=1, static=True)
    arg_index: int = eqx.field(default=1, static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.AbstractExpression(*args, **kwargs)
        # Grabs the step size (e.g. freq.f_scaled) from the context argument
        dx = getattr(args[self.arg_index], self.step_attr)
        
        for _ in range(self.order):
            data = jnp.gradient(data, dx, axis=self.axis)
        return data

class Flatness(Derivative):
    """Enforces gain flatness by computing the first derivative."""
    order: int = eqx.field(default=1, static=True)
    
class Diagonal(AbstractExpression[ExprInputs, ExprOutputs]):
    """Extracts the diagonals of matrices."""
    AbstractExpression: AbstractExpression

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        data = self.AbstractExpression(*args, **kwargs)
        return jax.vmap(jnp.diag)(data)

class OffDiagonal(Mask):
    """Extracts off-diagonal elements."""
    def __init__(self, AbstractExpression: AbstractExpression, n_ports: int, **kwargs):
        mask = ~jnp.eye(n_ports, dtype=bool)
        # We initialize the parent Mask class with the generated eye mask
        super().__init__(AbstractExpression=AbstractExpression, mask=mask, **kwargs)