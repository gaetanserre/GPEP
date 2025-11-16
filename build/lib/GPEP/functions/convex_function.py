#
# Created in 2024 by Gaëtan Serré
#

"""
Convex function specialization for GPEP.

Defines interpolation constraints appropriate for convex functions.
"""

from .function import Function


class ConvexFunction(Function):
    """
    Function container specialized for convex interpolation constraints.

    Parameters
    ----------
    None
        This class does not require constructor arguments; it initializes a
        Function with the name "Convex".
    """

    def __init__(self):
        super().__init__("Convex")

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        """
        Generate the two-point interpolation constraint for convexity.

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
            Constraint representing f2 + g2.dot(x1 - x2) <= f1 (convexity inequality).
        """
        return f2 + g2.dot(x1 - x2) <= f1
