"""
Equality operator for Expressions.

Compares evaluated value to a stored expression proxy.
"""

from .operator import Operator


class Eq(Operator):
    """Equality comparison operator node.

    Parameters
    ----------
    expr : Expression
        Right-hand side expression to compare against.
    """

    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        """Return boolean result of equality comparison.

        Parameters
        ----------
        value : any

        Returns
        -------
        bool or ndarray
        """
        return value == self.expr.eval()

    def str(self, expr):
        """Return string representation for equality.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
        """
        return f"({str(expr)} == {self.expr})"
