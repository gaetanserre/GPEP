#
# Created in 2024 by Gaëtan Serré
#

from ..const import Const
from ..operators import *
from .constraint import Constraint
import numpy as np


class Expression:
    def __init__(self, var, op_list=[]):
        self.var = var
        self.op_list = op_list

    def eval(self):
        val = self.var.eval()
        for op in self.op_list:
            val = op.eval(val)
        return val

    @staticmethod
    def conv_to_const(c):
        if isinstance(c, (int, float)):
            return Const(c)
        return c

    def __add__(self, other):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Add(other)])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other, r=False):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Sub(other, r)])

    def __rsub__(self, other):
        return self.__sub__(other, r=True)

    def __mul__(self, other):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Mul(other)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Expression(self.var, self.op_list + [Mul(Const(-1))])

    def __pow__(self, other, r=False):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Pow(other, r)])

    def __rpow__(self, other):
        return self.__pow__(other, r=True)

    def __truediv__(self, other, r=False):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Div(other, r)])

    def __rtruediv__(self, other):
        return self.__truediv__(other, r=True)

    def dot(self, other):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Dot(other)])

    def __matmul__(self, other):
        return self.dot(other)

    def __lt__(self, other):
        other = self.conv_to_const(other)
        return Constraint(self, other, lambda x, y: x < y)

    def __gt__(self, other):
        other = self.conv_to_const(other)
        return Constraint(other, self, lambda x, y: x < y)

    def __le__(self, other):
        other = self.conv_to_const(other)
        return Constraint(self, other, lambda x, y: x <= y)

    def __ge__(self, other):
        other = self.conv_to_const(other)
        return Constraint(other, self, lambda x, y: x <= y)

    def __eq__(self, other):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Eq(other)])

    def __ne__(self, other):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Ne(other)])

    def norm(self, other=2):
        other = self.conv_to_const(other)
        return Expression(self.var, self.op_list + [Norm(other)])

    def abs(self):
        return Expression(self.var, self.op_list + [Abs()])

    def exp(self):
        return Expression(self.var, self.op_list + [Exp()])

    def log(self):
        return Expression(self.var, self.op_list + [Log()])

    def __hash__(self):
        return hash((self.var, tuple(self.op_list)))


class ExpressionList(Expression):
    def __init__(self, e_list, aggr_cls):
        super().__init__(aggr_cls(e_list))


class Min:
    def __init__(self, e_list):
        self.e_list = e_list

    def eval(self):
        return np.min([e.eval() for e in self.e_list])

    def __hash__(self):
        return hash(tuple(self.e_list))


class Max:
    def __init__(self, e_list):
        self.e_list = e_list

    def eval(self):
        return np.max([e.eval() for e in self.e_list])

    def __hash__(self):
        return hash(tuple(self.e_list))


def emin(e_list):
    return ExpressionList(e_list, Min)


def emax(e_list):
    return ExpressionList(e_list, Max)
