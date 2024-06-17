#
# Created in 2024 by Gaëtan Serré
#

from .convex_function import ConvexFunction
import numpy as np


class ConvexLipschitzFunction(ConvexFunction):
    def __init__(self, M):
        self.M = M
        super().__init__()

    def gen_1_point_constraint(self, x, f, g):
        return np.linalg.norm(g) ** 2 - self.M**2
