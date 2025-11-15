#
# Created in 2024 by Gaëtan Serré
#

from .convex_function import ConvexFunction
from ..expression import Expression
from ..const import Const


class ConvexLipschitzFunction(ConvexFunction):
    def __init__(self, M):
        self.M = Expression(Const(M))
        super().__init__()
        self.name = "Convex Lipschitz"

    def gen_1_point_constraint(self, x, f, g):
        return g.norm() ** 2 <= self.M**2
