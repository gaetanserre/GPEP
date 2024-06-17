#
# Created in 2024 by Gaëtan Serré
#

from .function import Function
import numpy as np


class ConvexFunction(Function):
    def __init__(self):
        super().__init__()

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        return f2 - f1 + np.dot(g2, (x1 - x2))
