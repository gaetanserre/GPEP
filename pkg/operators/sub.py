#
# Created in 2024 by Gaëtan Serré
#

from .operator import Operator


class Sub(Operator):
    def __init__(self, expr, r=False):
        self.expr = expr
        self.r = r
        super().__init__()

    def eval(self, value):
        if not self.r:
            return value - self.expr.eval()
        else:
            return self.expr.eval() - value

    def str(self, expr):
        if not self.r:
            return f"({expr} - {self.expr})"
        else:
            return f"({self.expr} - {expr})"
