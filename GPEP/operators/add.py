#
# Created in 2024 by Gaëtan Serré
#

"""
Add operator for Expressions.

Defines the Add class, representing an addition operation in an expression tree.
"""

from .operator import Operator


class Add(Operator):
    """
    Addition operator node.

    Parameters
    ----------
    expr : Expression
        Right-hand expression to add.
    """

    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        """Return numeric addition result.

        Parameters
        ----------
        value : numeric or ndarray
            Left-hand evaluated value.

        Returns
        -------
        numeric or ndarray
            value + expr.eval()
        """
        return value + self.expr.eval()

    def str(self, expr):
        """Return string representation of the addition node.

        Parameters
        ----------
        expr : Expression
            Expression rendering of the inner/left expression.

        Returns
        -------
        str
            Formatted string representing the addition.
        """
        return f"({str(expr)} + {self.expr})"
