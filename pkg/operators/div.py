#
# Created in 2024 by Gaëtan Serré
#

from .operator import Operator


class Div(Operator):
    def __init__(self, expr, r=False):
        self.expr = expr
        self.r = r
        super().__init__()

    def eval(self, value):
        if not self.r:
            return value / self.expr.eval()
        else:
            return self.expr.eval() / value
