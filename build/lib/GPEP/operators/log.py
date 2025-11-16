"""
Natural logarithm operator for Expressions.

Applies numpy.log to evaluated values and provides a printable form.
"""

from .operator import Operator
import numpy as np


class Log(Operator):
    """Natural logarithm operator node."""

    def __init__(self):
        super().__init__()

    def eval(self, value):
        """Return numpy.log(value).

        Parameters
        ----------
        value : numeric or ndarray

        Returns
        -------
        numeric or ndarray
        """
        return np.log(value)

    def str(self, expr):
        """Return string representation for log.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
        """
        return f"log({str(expr)})"
