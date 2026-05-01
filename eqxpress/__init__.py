from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass

from eqxpress.base import AbstractExpression, ExprInputs, ExprOutputs
from eqxpress.primitives import Lambda, Constant, Binary, Where, Method, Map
from eqxpress.algebra import Stack, Index, Mask, Reduce, Sum, Negate, Derivative, Flatness, Diagonal, OffDiagonal

__all__ = [
    'AbstractExpression', 'ExprInputs', 'ExprOutputs',
    'Lambda', 'Constant', 'Binary', 'Where', 'Method', 'Map',
    'Stack', 'Index', 'Mask', 'Reduce', 'Sum', 'Negate', 'Derivative', 'Flatness', 'Diagonal', 'OffDiagonal'
]