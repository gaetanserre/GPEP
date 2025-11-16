#
# Created in 2024 by Gaëtan Serré
#

"""
Smooth convex function specialization for GPEP.

Provides a Function subclass encoding interpolation constraints for L-smooth
convex functions (special case mu=0 of SmoothStronglyConvexFunction).
"""

from .smooth_strongly_convex_function import SmoothStronglyConvexFunction


class SmoothConvexFunction(SmoothStronglyConvexFunction):
    """
    Function container enforcing L-smooth convex interpolation constraints.

    Parameters
    ----------
    L : float
        Smoothness constant (L > 0). This class instantiates the
        SmoothStronglyConvexFunction with mu=0.
    """

    def __init__(self, L):
        """
        Initialize the SmoothConvexFunction.

        Parameters
        ----------
        L : float
            Smoothness constant.
        """
        super().__init__(L, mu=0)
        self.name = "Smooth Convex"
