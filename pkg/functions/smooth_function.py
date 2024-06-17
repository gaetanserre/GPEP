#
# Created in 2024 by Gaëtan Serré
#

from .function import Function
import numpy as np


class SmoothFunction(Function):
    def __init__(self, L):
        self.L = L
        super().__init__()

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        return (
            -f1
            + f2
            - self.L / 4 * np.linalg.norm(x1 - x2) ** 2
            + 1 / 2 * np.dot(g1 + g2, x1 - x2)
            + 1 / (4 * self.L) * np.linalg.norm(g1 - g2) ** 2
        )
