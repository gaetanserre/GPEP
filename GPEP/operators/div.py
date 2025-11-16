#
# Created in 2024 by Gaëtan Serré
#

"""
Division operator for Expressions.

Supports both value / expr and expr / value depending on the `r` flag.
"""

from .operator import Operator


class Div(Operator):
    """
    Division operator node.

    Parameters
    ----------
    expr : Expression
        Divisor or dividend depending on `r`.
    r : bool, optional
        If True, computes expr / value instead of value / expr.
    """

    def __init__(self, expr, r=False):
        self.expr = expr
        self.r = r
        super().__init__()

    def eval(self, value):
        """
        Return numeric division result.

        Parameters
        ----------
        value : numeric or ndarray
            Left-hand evaluated value.

        Returns
        -------
        numeric or ndarray
            value / expr.eval() or expr.eval() / value
        """
        if not self.r:
            return value / self.expr.eval()
        else:
            return self.expr.eval() / value

    def str(self, expr):
        """
        Return string representation of the division node.

        Parameters
        ----------
        expr : Expression
            The left/right expression.

        Returns
        -------
        str
        """
        if not self.r:
            return f"({str(expr)} / {self.expr})"
        else:
            return f"({self.expr} / {str(expr)})"
