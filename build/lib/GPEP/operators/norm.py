"""
Norm operator for Expressions.

Computes the vector norm (numpy.linalg.norm) of an evaluated value with given order.
"""

from .operator import Operator
import numpy as np


class Norm(Operator):
    """Norm operator node.

    Parameters
    ----------
    expr : Expression
        Order of the norm (e.g. 2 for Euclidean norm) or an Expression/Const.
    """

    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        """Return numpy.linalg.norm(value, order).

        Parameters
        ----------
        value : ndarray

        Returns
        -------
        float
            Computed norm.
        """
        return np.linalg.norm(value, self.expr.eval())

    def str(self, expr):
        """String representation for norm.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
        """
        return f"||{str(expr)}||_{self.expr}"
