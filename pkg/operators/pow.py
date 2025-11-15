#
# Created in 2024 by Gaëtan Serré
#

from .operator import Operator
import numpy as np


class Pow(Operator):
    def __init__(self, expr, r=False):
        self.expr = expr
        self.r = r
        super().__init__()

    def eval(self, value):
        if not self.r:
            return np.power(value, self.expr.eval())
        else:
            return np.power(self.expr.eval(), value)

    def str(self, expr):
        if not self.r:
            return f"({expr} ^ {self.expr})"
        else:
            return f"({self.expr} ^ {expr})"
