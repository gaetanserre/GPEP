#
# Created in 2024 by Gaëtan Serré
#

from .eq import Eq


class Ne(Eq):
    def eval(self, value):
        return not super().eval(value)

    def str(self, expr):
        return f"({expr} != {self.expr})"
