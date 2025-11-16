#
# Created in 2024 by Gaëtan Serré
#

"""
Smooth strongly convex function specialization for GPEP.

This module provides a Function subclass encoding interpolation constraints
for L-smooth, mu-strongly convex functions following the smooth-strongly convex
interpolation framework.

"""

from .function import Function
from ..expression import Expression
from ..const import Const


class SmoothStronglyConvexFunction(Function):
    """
    Function container enforcing L-smoothness and mu-strong convexity interpolation constraints.

    Parameters
    ----------
    L : float
        Smoothness constant (L > 0). Internally stored as Expression(Const(L)).
    mu : float
        Strong convexity constant (0 <= mu < L). Internally stored as Expression(Const(mu)).

    Notes
    -----
    The two-point interpolation inequality implemented in gen_2_points_constraint
    follows the formulation from [1].

    Reference
    ---------
    `[1] A. Taylor, J. Hendrickx, F. Glineur (2017).
    Smooth strongly convex interpolation and exact worst-case performance of first-order methods.
    Mathematical Programming, 161(1-2), 307-345.
    <https://arxiv.org/pdf/1502.05666.pdf>`_

    """

    def __init__(self, L, mu):
        """
        Initialize the SmoothStronglyConvexFunction.

        Parameters
        ----------
        L : float
            Smoothness constant.
        mu : float
            Strong convexity constant.
        """
        self.L = Expression(Const(L))
        self.mu = Expression(Const(mu))
        super().__init__("Smooth Strongly Convex")

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        """
        Generate the two-point interpolation constraint for (L, mu)-smooth strongly convex functions.

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
            Constraint expressing the interpolation inequality (as in [1]) ensuring
            compatibility with L-smoothness and mu-strong convexity.
        """
        return (
            g2.dot(x1 - x2)
            + (1 / (2 * self.L)) * (g1 - g2).norm() ** 2
            + self.mu
            / (2 * (1 - self.mu / self.L))
            * (x1 - x2 - (1 / self.L) * (g1 - g2)).norm() ** 2
            <= f1 - f2
        )
