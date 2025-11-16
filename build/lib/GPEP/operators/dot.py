#
# Created in 2024 by Gaëtan Serré
#

"""
Dot-product operator for Expressions.

Computes the dot product between evaluated values and a stored vector expression.
"""

from .operator import Operator
import numpy as np


class Dot(Operator):
    """
    Dot product operator node.

    Parameters
    ----------
    expr : Expression
        Right-hand side vector expression to dot with.
    """

    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        """
        Return numeric dot product result.

        Parameters
        ----------
        value : numeric or ndarray
            Left-hand evaluated value.

        Returns
        -------
        numeric or ndarray
            value · expr.eval()
        """
        return np.dot(value, self.expr.eval())

    def str(self, expr):
        """
        Return string representation of the dot product node.

        Parameters
        ----------
        expr : Expression
            The inner expression.

        Returns
        -------
        str
        """
        return f"({str(expr)} · {self.expr})"
