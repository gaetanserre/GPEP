"""
Subtraction operator for Expressions.

Supports both left and right subtraction depending on the `r` flag and
produces a readable string for composed expressions.
"""

from .operator import Operator


class Sub(Operator):
    """Subtraction operator node.

    Parameters
    ----------
    expr : Expression
        The expression to subtract.
    r : bool, optional
        If True, computes expr - value instead of value - expr.
    """

    def __init__(self, expr, r=False):
        self.expr = expr
        self.r = r
        super().__init__()

    def eval(self, value):
        """Evaluate the subtraction.

        Parameters
        ----------
        value : numeric or ndarray
            Evaluated left-hand (or right-hand when r=True) value.

        Returns
        -------
        numeric or ndarray
            Result of the subtraction.
        """
        if not self.r:
            return value - self.expr.eval()
        else:
            return self.expr.eval() - value

    def str(self, expr):
        """String representation for subtraction.

        Parameters
        ----------
        expr : Expression
            Expression rendering of the inner expression.

        Returns
        -------
        str
            Formatted string representing the subtraction.
        """
        if not self.r:
            return f"({str(expr)} - {self.expr})"
        else:
            return f"({self.expr} - {str(expr)})"
