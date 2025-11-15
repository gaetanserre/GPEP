#
# Created in 2024 by Gaëtan Serré
#

from .function import Function
from ..expression import Expression
from ..const import Const


class SmoothStronglyConvexFunction(Function):
    def __init__(self, L, mu):
        self.L = Expression(Const(L))
        self.mu = Expression(Const(mu))
        super().__init__("Smooth Strongly Convex")

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        return (
            g2.dot(x1 - x2)
            + (1 / (2 * self.L)) * (g1 - g2).norm() ** 2
            + self.mu
            / (2 * (1 - self.mu / self.L))
            * (x1 - x2 - (1 / self.L) * (g1 - g2)).norm() ** 2
            <= f1 - f2
        )
