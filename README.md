**eqxpress** is a lightweight library for composing expressions over input-output space.

| **eqxpress** |  |
|-------------|-------|
| **Author**  | Gary Allen |
| **Homepage** | [github.com/eqxpress/eqxpress](https://github.com/eqxpress/eqxpress) |
| **Docs** | [gvcallen.github.io/eqxpress](https://gvcallen.github.io/eqxpress) |

## Installation
`eqxpress` can be installed via `pip`.

``
pip install eqxpress
``

## Example

```python
class Scale(AbstractExpression):
    def __call__(self, x): return {"data": x * 2.0}

class Shift(AbstractExpression):
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