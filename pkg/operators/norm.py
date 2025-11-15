#
# Created in 2024 by Gaëtan Serré
#

from .operator import Operator
import numpy as np


class Norm(Operator):
    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        return np.linalg.norm(value, self.expr.eval())

    def str(self, expr):
        return f"||{expr}||_{self.expr}"
