"""
Core operators.
"""

from __future__ import annotations
import operator
from typing import Callable, Any, Union

import jax
import equinox as eqx

from eqxpress import AbstractExpression, ExprInputs, ExprOutputs

class Lambda(AbstractExpression[ExprInputs, ExprOutputs]):
    """
    Wraps a standard Python or JAX callable with the same domain as the operator.
    """
    fn: Callable[ExprInputs, ExprOutputs]

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        return self.fn(*args, **kwargs)


class Constant(AbstractExpression[ExprInputs, ExprOutputs]):
    """
    Returns a fixed constant array or scalar.
    """
    value: ExprOutputs
    
    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        return self.value


class Binary(AbstractExpression[ExprInputs, ExprOutputs]):
    """
    Returns the result of a callable that accepts the result of two operators.
    
    The functional callable ``fn`` must have the signature ``f(left, right)``.
    """
    fn: Callable[[Any, Any], ExprOutputs]
    left: Union[AbstractExpression[ExprInputs, Any], Any]
    right: Union[AbstractExpression[ExprInputs, Any], Any]

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        val_left = self.left(*args, **kwargs) if isinstance(self.left, AbstractExpression) else self.left
        val_right = self.right(*args, **kwargs) if isinstance(self.right, AbstractExpression) else self.right
        return self.fn(val_left, val_right)


class Where(AbstractExpression[ExprInputs, ExprOutputs]):
    """
    A conditional branching node using `jax.lax.cond`.

    Evaluates a boolean condition (from an Operator) and returns the output
    of either `true_branch` or `false_branch` depending on the condition.
    """
    condition: Union[AbstractExpression[ExprInputs, Any], Any]
    true_branch: Union[AbstractExpression[ExprInputs, ExprOutputs], ExprOutputs]
    false_branch: Union[AbstractExpression[ExprInputs, ExprOutputs], ExprOutputs]

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        cond_val = self.condition(*args, **kwargs) if isinstance(self.condition, AbstractExpression) else self.condition
        
        def true_fn(_: Any) -> ExprOutputs:
            return self.true_branch(*args, **kwargs) if isinstance(self.true_branch, AbstractExpression) else self.true_branch
        
        def false_fn(_: Any) -> ExprOutputs:
            return self.false_branch(*args, **kwargs) if isinstance(self.false_branch, AbstractExpression) else self.false_branch
        
        return jax.lax.cond(cond_val, true_fn, false_fn, operand=None)


class Method(AbstractExpression[ExprInputs, ExprOutputs]):
    """
    Dynamically accesses and executes a method on the first argument.
    """
    path: str = eqx.field(static=True)

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        if not args:
            raise ValueError(f"Method operator '{self.path}' requires at least one positional argument.")
            
        obj = args[0] 
        method_args = args[1:]
        
        func = operator.attrgetter(self.path)(obj)
        return func(*method_args, **kwargs)


class Map(AbstractExpression[ExprInputs, ExprOutputs]):
    """
    Applies an arbitrary function to a single operator's output.
    """
    fn: Callable[[Any], ExprOutputs]
    operator: Union[AbstractExpression[ExprInputs, Any], Any]

    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        val = self.operator(*args, **kwargs) if isinstance(self.operator, AbstractExpression) else self.operator
        return self.fn(val)