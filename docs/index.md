**eqxpress** is a lightweight library for composing expressions over input-output PyTree spaces. It allows you to build deferred computation graphs and abstract syntax trees (ASTs) using standard Python operators.

Since **eqxpress** is built natively on Equinox, each new instance of `AbstractExpression` is a new PyTree, and therefore any PyTree leaves (weights, arrays, parameters) are propagated throughout nested operations.

| **eqxpress** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/eqxpress/eqxpress](https://github.com/eqxpress/eqxpress) |
| **Docs** | [gvcallen.github.io/eqxpress](https://gvcallen.github.io/eqxpress) |

## Installation
`eqxpress` can be installed via `pip`:

``
pip install eqxpress
``

## Example: Composing Modules

If you have an existing hierarchy of Equinox modules with matching input/output signatures, simply inheriting from AbstractExpression grants them instant, declarative composition.

A common use case is building composite loss functions for optimization:

```python
import jax.numpy as jnp
from eqxpress import AbstractExpression

class MSELoss(AbstractExpression):
    def __call__(self, y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)

class L2Regularization(AbstractExpression):
    def __call__(self, y_true, y_pred):
        return jnp.sum(y_pred ** 2)

# Build a deferred computation graph
total_loss = MSELoss() + 0.01 * L2Regularization()

# Evaluate the combined mapping
y_t = jnp.array([1.0, 0.0])
y_p = jnp.array([0.9, 0.1])

loss_value = total_loss(y_true=y_t, y_pred=y_p)
```
