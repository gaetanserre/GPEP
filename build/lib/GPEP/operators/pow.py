"""
Power operator for Expressions.

Supports both value ** expr and expr ** value depending on the `r` flag.
"""

from .operator import Operator
import numpy as np


class Pow(Operator):
    """Power operator node.

    Parameters
    ----------
    expr : Expression
        Exponent/base expression depending on `r`.
    r : bool, optional
        If True, computes expr ** value instead of value ** expr.
    """

    def __init__(self, expr, r=False):
        self.expr = expr
        self.r = r
        super().__init__()

    def eval(self, value):
        """Evaluate the power using numpy.power.

        Parameters
        ----------
        value : numeric or ndarray

        Returns
        -------
        numeric or ndarray
            numpy.power(value, expr.eval()) or numpy.power(expr.eval(), value)
        """
        if not self.r:
            return np.power(value, self.expr.eval())
        else:
            return np.power(self.expr.eval(), value)

    def str(self, expr):
        """Return string representation for power.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
        """
        if not self.r:
            return f"({str(expr)} ^ {self.expr})"
        else:
            return f"({self.expr} ^ {str(expr)})"
