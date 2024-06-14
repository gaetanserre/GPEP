#
# Created in 2024 by Gaëtan Serré
#

from .operator import Operator


class Ge(Operator):
    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def eval(self, value):
        return self.expr.eval() <= value
