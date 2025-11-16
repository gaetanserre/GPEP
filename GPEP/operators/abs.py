#
# Created in 2024 by Gaëtan Serré
#

"""
Absolute-value operator for Expressions.

Applies numpy.abs to the evaluated value and provides a printable form.
"""

from .operator import Operator
import numpy as np


class Abs(Operator):

    def __init__(self):
        super().__init__()

    def eval(self, value):
        """Return absolute value (numpy.abs).

        Parameters
        ----------
        value : numeric or ndarray

        Returns
        -------
        numeric or ndarray
        """
        return np.abs(value)

    def str(self, expr):
        """String representation for absolute value.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
        """
        return f"|{str(expr)}|"
