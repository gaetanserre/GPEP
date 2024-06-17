#
# Created in 2024 by Gaëtan Serré
#

from .function import Function
import numpy as np


class SmoothStronglyConvexFunction(Function):
    def __init__(self, L, mu):
        self.L = L
        self.mu = mu
        super().__init__()

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        return (
            f1
            + np.dot(g1, x2 - x1)
            + (1 / (2 * self.L)) * np.linalg.norm(g2 - g1) ** 2
            + ((self.mu * self.L) / (2 * (self.L - self.mu)))
            * np.linalg.norm(x1 - x2 - (1 / self.L) * (g1 - g2)) ** 2
            - f2
        )
