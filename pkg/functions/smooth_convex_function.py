#
# Created in 2024 by Gaëtan Serré
#

from .function import Function
import numpy as np


class SmoothConvexFunction(Function):
    def __init__(self, L):
        self.L = L
        super().__init__()

    def gen_constraint(self, x1, x2, f1, f2, g1, g2):
        return (
            f1
            + np.dot(g1, x2 - x1)
            + (1 / (2 * self.L)) * np.linalg.norm(g2 - g1) ** 2
            - f2
        )
