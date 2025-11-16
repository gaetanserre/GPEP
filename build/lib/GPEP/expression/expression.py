#
# Created in 2024 by Gaëtan Serré
#

"""
Expression abstraction for GPEP.

This module defines the Expression wrapper used across the project to build
composed numerical expressions from a base variable or constant and a list of
operators. Expressions support lazy composition via operator overloads and can
be evaluated to concrete numeric values using eval(). Comparison operators
produce Constraint objects from the project's constraint subsystem.

Main components:
- Expression: core composable expression type that wraps a variable/constant and a list of operator nodes.
- ExpressionList: helper to build aggregated expressions such as min/max over many expressions.
- Min/Max and emin/emax: small aggregators exposing an eval() to compute the aggregate value.

These docstrings describe intent and public behavior; for operator semantics consult
the operators package and constraint.py in this project.
"""

from ..const import Const
from ..operators import *
from .constraint import Constraint
import numpy as np


class Expression:
    """
    Composable numeric expression built from a base variable/constant and operators.

    Parameters
    ----------
    var : Variable or Const
        The base variable or constant for the expression. Must implement eval().
    op_list : list, optional
        Sequence of operator nodes to apply (each providing eval/str). Defaults to [].

    Notes
    -----
    Operator overloads append operator nodes and return new Expression instances.
    Comparison overloads return Constraint objects.
    """

    def __init__(self, var, op_list=[]):
        self.var = var
        self.op_list = op_list

    def eval(self):
        """
        Evaluate the expression by applying all operators in sequence.

        Returns
        -------
        numeric
            The numeric result of evaluating the base var and applying each operator in op_list.
        """
        val = self.var.eval()
        for op in self.op_list:
            val = op.eval(val)
        return val

    @staticmethod
    def conv_to_const(c):
        """
        Convert plain numeric types to a Const-wrapped Expression when needed.

        Parameters
        ----------
        c : int | float | Expression
            Value to convert. If int or float, it will be wrapped as Expression(Const(c)).

        Returns
        -------
        Expression or original
            Expression(Const(c)) if c is numeric, otherwise c unchanged.
        """
        if isinstance(c, (int, float)):
            return Expression(Const(c))
        return c

    def __str__(self):
        """
        Return a human-readable representation of the expression.

        Returns
        -------
        str
            A readable nested representation built by folding operator .str(...) calls.
        """

        def rec_aux(op_list):
            if op_list == []:
                return str(self.var)
            else:
                return op_list[-1].str(rec_aux(op_list[:-1]))

        return rec_aux(self.op_list)

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
        return Constraint(self, other, lambda x, y: x < y, "<")

    def __gt__(self, other):
        other = self.conv_to_const(other)
        return Constraint(other, self, lambda x, y: x < y, "<")

    def __le__(self, other):
        other = self.conv_to_const(other)
        return Constraint(self, other, lambda x, y: x <= y, "<=")

    def __ge__(self, other):
        other = self.conv_to_const(other)
        return Constraint(other, self, lambda x, y: x <= y, "<=")

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
    """
    Expression that aggregates a list of expressions using an aggregator class.

    Parameters
    ----------
    e_list : iterable
        Iterable of Expression instances to aggregate.
    aggr_cls : class
        Aggregator class that accepts e_list in its constructor and exposes eval().

    Notes
    -----
    The aggregator class must implement eval() to compute the aggregated value.
    """

    def __init__(self, e_list, aggr_cls):
        super().__init__(aggr_cls(e_list))


class Min:
    """
    Aggregator returning the minimum value among a list of expressions.

    Parameters
    ----------
    e_list : iterable
        Iterable of Expression objects whose values will be compared.
    """

    def __init__(self, e_list):
        self.e_list = e_list

    def eval(self):
        """
        Evaluate all expressions and return their minimum as a scalar/array.

        Returns
        -------
        numeric
            Minimum of evaluated expressions (numpy scalar/array as returned by np.min).
        """
        return np.min([e.eval() for e in self.e_list])

    def __hash__(self):
        return hash(tuple(self.e_list))


class Max:
    """
    Aggregator returning the maximum value among a list of expressions.

    Parameters
    ----------
    e_list : iterable
        Iterable of Expression objects whose values will be compared.
    """

    def __init__(self, e_list):
        self.e_list = e_list

    def eval(self):
        """
        Evaluate all expressions and return their maximum as a scalar/array.

        Returns
        -------
        numeric
            Maximum of evaluated expressions (numpy scalar/array as returned by np.max).
        """
        return np.max([e.eval() for e in self.e_list])

    def __hash__(self):
        return hash(tuple(self.e_list))


def emin(e_list):
    """
    Construct an ExpressionList that evaluates to the element-wise minimum.

    Parameters
    ----------
    e_list : iterable
        Iterable of Expression objects.

    Returns
    -------
    ExpressionList
        ExpressionList wrapping a Min aggregator.
    """
    return ExpressionList(e_list, Min)


def emax(e_list):
    """
    Construct an ExpressionList that evaluates to the element-wise maximum.

    Parameters
    ----------
    e_list : iterable
        Iterable of Expression objects.

    Returns
    -------
    ExpressionList
        ExpressionList wrapping a Max aggregator.
    """
    return ExpressionList(e_list, Max)
