#
# Created in 2024 by Gaëtan Serré
#

"""
Smooth function specialization for GPEP.

Provides a Function subclass encoding interpolation constraints for L-smooth
functions.
"""

from .function import Function
from ..expression import Expression
from ..const import Const


class SmoothFunction(Function):
    """Function container enforcing L-smooth interpolation constraints.

    Parameters
    ----------
    L : float
        Smoothness constant (L > 0). Stored internally as Expression(Const(L)).

    Reference
    ---------
    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
    Mathematical Programming, 161(1-2), 307-345.
    <https://arxiv.org/pdf/1502.05666.pdf>`_
    """

    def __init__(self, L):
        """
        Initialize the SmoothFunction.

        Parameters
        ----------
        L : float
            Smoothness constant.
        """
        self.L = Expression(Const(L))
        super().__init__("Smooth")

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        """
        Generate the two-point interpolation constraint for L-smoothness.

        Parameters
        ----------
        x1, x2 : Variable or Expression
            Input points or tracked expressions.
        f1, f2 : Variable
            Value proxy variables corresponding to x1 and x2.
        g1, g2 : Variable
            Gradient proxy variables corresponding to x1 and x2.

        Returns
        -------
        Constraint
            Constraint expressing the smoothness two-point inequality.
        """
        return (
            -self.L / 4 * (x1 - x2).norm() ** 2
            + 1 / 2 * (g1 + g2).dot(x1 - x2)
            + 1 / (4 * self.L) * (g1 - g2).norm() ** 2
            <= f1 - f2
        )
