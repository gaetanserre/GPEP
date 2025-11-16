"""
Exponential operator for Expressions.

Applies numpy.exp to evaluated values and provides a string form.
"""

from .operator import Operator
import numpy as np


class Exp(Operator):
    """Exponential operator node."""

    def __init__(self):
        super().__init__()

    def eval(self, value):
        """Return numpy.exp(value).

        Parameters
        ----------
        value : numeric or ndarray

        Returns
        -------
        numeric or ndarray
        """
        return np.exp(value)

    def str(self, expr):
        """Return string representation for exponential.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
        """
        return f"exp({str(expr)})"
