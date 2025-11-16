"""
Multiplication operator for Expressions.

Performs multiplication with a stored expression proxy and provides a string form.
"""

from .operator import Operator


class Mul(Operator):
    """Multiplication operator node.

    Parameters
    ----------
    expr : Expression
        Right-hand expression or scalar to multiply by.
    """

    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        """Return product of value and expr.eval().

        Parameters
        ----------
        value : numeric or ndarray

        Returns
        -------
        numeric or ndarray
            value * expr.eval()
        """
        return value * self.expr.eval()

    def str(self, expr):
        """Return string representation for multiplication.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
            Formatted string representing the multiplication.
        """
        return f"({str(expr)} * {self.expr})"
