#
# Created in 2024 by Gaëtan Serré
#
from .operator import Operator


class Add(Operator):
    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        return value + self.expr.eval()
