#
# Created in 2024 by Gaëtan Serré
#

"""
Base operator abstraction for GPEP.

Defines the minimal Operator interface expected by the expression system:
- eval(value) computes the operator action on a numeric/array value.
- str(expr) returns a textual representation used when building expression strings.
"""


class Operator:
    """Abstract operator node used in Expression.op_list.

    Parameters
    ----------
    None

    Notes
    -----
    Subclasses should implement eval(value) and str(expr). The base class
    provides no concrete implementation.
    """

    def __init__(self):
        pass

    def eval(self, value):
        """Apply the operator to a numeric/array value.

        Parameters
        ----------
        value : numeric or ndarray
            Input value to which the operator is applied.

        Returns
        -------
        numeric or ndarray
            Result of the operator application.
        """
        pass

    def str(self, expr):
        """Return a string representation of the operator when applied to expr.

        Parameters
        ----------
        expr : Expression
            The inner expression.

        Returns
        -------
        str
            Human-readable string representing this operator applied to expr.
        """
        pass
