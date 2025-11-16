#
# Created in 2024 by Gaëtan Serré
#

"""
Convex Lipschitz function specialization for GPEP.

Provides a ConvexFunction variant that enforces a Lipschitz bound on the gradient.
"""

from .convex_function import ConvexFunction
from ..expression import Expression
from ..const import Const


class ConvexLipschitzFunction(ConvexFunction):
    """
    Convex function with a Lipschitz-gradient constraint.

    Parameters
    ----------
    M : float
        Lipschitz constant for the gradient (positive scalar). Stored internally
        as an Expression(Const(M)).
    """

    def __init__(self, M):
        """
        Initialize the ConvexLipschitzFunction.

        Parameters
        ----------
        M : float
            Lipschitz constant for the gradient.
        """
        self.M = Expression(Const(M))
        super().__init__()
        self.name = "Convex Lipschitz"

    def gen_1_point_constraint(self, x, f, g):
        """
        Generate the single-point Lipschitz-gradient constraint.

        Parameters
        ----------
        x : Variable or Expression
            Point or expression for which the constraint applies.
        f : Variable
            Value proxy associated with x.
        g : Variable
            Gradient proxy associated with x.

        Returns
        -------
        Constraint
            Constraint enforcing ||g||^2 <= M^2.
        """
        return g.norm() ** 2 <= self.M**2
