#
# Created in 2024 by Gaëtan Serré
#

from ..expression import Expression


class Variable:
    def __init__(self, id, value=None):
        self.id = id
        self.value = value

    def eval(self):
        if self.value is None:
            raise ValueError(f"Variable '{self.id}' has no value assigned.")
        return self.value

    def set_value(self, value):
        self.value = value

    def to_expr(self):
        return Expression(self)

    def __str__(self):
        return f"var({self.id})"

    def __add__(self, other):
        return self.to_expr().__add__(other)

    def __radd__(self, other):
        return self.to_expr().__radd__(other)

    def __sub__(self, other, r=False):
        return self.to_expr().__sub__(other, r)

    def __rsub__(self, other):
        return self.to_expr().__rsub__(other)

    def __mul__(self, other):
        return self.to_expr().__mul__(other)

    def __rmul__(self, other):
        return self.to_expr().__rmul__(other)

    def __neg__(self):
        return self.to_expr().__neg__()

    def __pow__(self, other):
        return self.to_expr().__pow__(other)

    def __rpow__(self, other):
        return self.to_expr().__rpow__(other)

    def __truediv__(self, other, r=False):
        return self.to_expr().__truediv__(other, r)

    def __rtruediv__(self, other):
        return self.to_expr().__rtruediv__(other)

    def dot(self, other):
        return self.to_expr().dot(other)

    def __matmul__(self, other):
        return self.dot(other)

    def __lt__(self, other):
        return self.to_expr().__lt__(other)

    def __gt__(self, other):
        return self.to_expr().__gt__(other)

    def __le__(self, other):
        return self.to_expr().__le__(other)

    def __ge__(self, other):
        return self.to_expr().__ge__(other)

    def __eq__(self, other):
        return self.to_expr().__eq__(other)

    def __ne__(self, other):
        return self.to_expr().__ne__(other)

    def norm(self, other=2):
        return self.to_expr().norm(other)

    def __hash__(self):
        return hash(self.id)
