#
# Created in 2024 by Gaëtan Serré
#

from .function import Function
from ..expression import Expression
from ..const import Const


class SmoothFunction(Function):
    def __init__(self, L):
        self.L = Expression(Const(L))
        super().__init__("Smooth")

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        return (
            -self.L / 4 * (x1 - x2).norm() ** 2
            + 1 / 2 * (g1 + g2).dot(x1 - x2)
            + 1 / (4 * self.L) * (g1 - g2).norm() ** 2
            <= f1 - f2
        )
