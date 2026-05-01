from __future__ import annotations
from abc import abstractmethod
import operator
from typing import Any, Generic, TypeVar, Union, ParamSpec
import equinox as eqx

from eqxpress.utils.tree import tree_op

ExprInputs = ParamSpec("P")
ExprOutputs = TypeVar("OpOutputs")

class AbstractExpression(eqx.Module, Generic[ExprInputs, ExprOutputs]):
    r"""
    A composable callable for building delayed computation graphs over PyTrees.
    
    ### Overview
    This class allows easily combining mappings over the same input/output space.
    
    For example, you may have a `Loss` class with a number of child classes
    (MSE, RMSE etc.) that accepts (y_true, y_pred) and outputs an error.
    By simply inheriting from `AbstractExpression` and overriding `__call__`,
    you can now seemlessly combine loss functions into new ones using operators
    like addition, subtraction etc.
    
    Note that, semantically, it only makes sense to inherit from `AbstractExpression`
    if operations on your expression space are semantically the same as operations
    on your output space. For the above example, since "adding loss functions"
    is semantically the same as "adding loss values", this is an applicable example.

    ### Mathematical Formulation
    This class constructs an algebra over a field for the space of mappings 
    $f: X \to Y$. If the output space $Y$ is a vector space (e.g., tensors or 
    JAX arrays), then the space of all functions mapping into $Y$ is natively 
    a vector space. 
    
    By overloading standard Python operators, this class implements point-wise 
    operations on these mathematical mappings directly:
    * Addition: $(f + g)(x) = f(x) + g(x)$
    * Scalar Multiplication: $(c \cdot f)(x) = c \cdot f(x)$
    * Element-wise Multiplication: $(f \cdot g)(x) = f(x) \cdot g(x)$

    ### Assumptions
    The combined graph is mathematically valid provided the concrete `__call__` 
    methods return compatible PyTrees whose leaves are arrays or numeric scalars. 
    Operations are automatically wrapped in JAX tree utilities, allowing safe 
    element-wise computation and scalar broadcasting across complex structures.

    ### Example
    ```python
    class Scale(Operator):
        def __call__(self, x): return {"data": x * 2.0}

    class Shift(Operator):
        def __call__(self, x): return {"data": x + 5.0}

    # 1. Build the deferred computation graph
    # Mathematically defines h(x) = Scale(x) + 10 * Shift(x)
    h = Scale() + 10 * Shift()

    # 2. Evaluate the combined mapping
    # Operations map point-wise over the dictionary keys automatically
    y = h(10.0) 
    
    # y == {"data": (10.0 * 2.0) + 10 * (10.0 + 5.0)} 
    # y == {"data": 170.0}
    ```
    """
    
    @abstractmethod
    def __call__(self, *args: ExprInputs.args, **kwargs: ExprInputs.kwargs) -> ExprOutputs:
        raise NotImplementedError("Operator nodes must implement __call__.")

    # --- Arithmetic Operators ---

    def __add__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.add))

    def __sub__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.sub))

    def __mul__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.mul))

    def __truediv__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.truediv))

    def __pow__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.pow))

    # --- Unary Operators ---
    
    def __neg__(self) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.algebra import Negate
        return Negate(self)

    # --- Reverse Arithmetic (for <scalar> + <Operator>) ---

    def __radd__(self, other: Any) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.add))

    def __rsub__(self, other: Any) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.sub))

    def __rmul__(self, other: Any) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.mul))
        
    def __rtruediv__(self, other: Any) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=other, right=self, fn=tree_op(operator.truediv))

    # --- Comparison Operators ---

    def __gt__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.gt))

    def __lt__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.lt))
        
    def __ge__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.ge))

    def __le__(self, other: Union[AbstractExpression[ExprInputs, Any], Any]) -> AbstractExpression[ExprInputs, ExprOutputs]:
        from eqxpress.nodes import Binary
        return Binary(left=self, right=other, fn=tree_op(operator.le))