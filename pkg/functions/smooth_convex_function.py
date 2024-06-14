#
# Created in 2024 by Gaëtan Serré
#

from .smooth_strongly_convex_function import SmoothStronglyConvexFunction


class SmoothConvexFunction(SmoothStronglyConvexFunction):
    def __init__(self, L):
        super().__init__(L, mu=0)
