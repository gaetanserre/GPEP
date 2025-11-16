#
# Created in 2024 by Gaëtan Serré
#

"""
Constraint representation for GPEP.

Represents a binary relation between two Expression objects used by the
problem/solver layers. Constraint wraps two expressions and a binary operator
(callable) describing the relation; a short symbolic form is stored for
pretty-printing.
"""


class Constraint:
    """
    Binary constraint between two expressions.

    Parameters
    ----------
    expr1 : Expression
        Left-hand expression (evaluatable via eval()).
    expr2 : Expression
        Right-hand expression (evaluatable via eval()).
    op : callable
        Binary operator implementing the relation (e.g. lambda x, y: x < y).
    sym : str
        Symbolic representation used for printing (e.g. '<', '<=', '==').
    """

    def __init__(self, expr1, expr2, op, sym):
        self.expr1 = expr1
        self.expr2 = expr2
        self.op = op
        self.sym = sym

    def eval(self):
        """
        Evaluate the relation between expr1 and expr2.

        Returns
        -------
        bool or ndarray
            Result of op(expr1.eval(), expr2.eval()).
        """
        return self.op(self.expr1.eval(), self.expr2.eval())

    def c_eval(self):
        """
        Canonical evaluation for solvers: expr1 - expr2.

        Returns
        -------
        numeric or ndarray
            Difference expr1.eval() - expr2.eval(), useful for constraint residuals.
        """
        return self.expr1.eval() - self.expr2.eval()

    def __str__(self):
        """
        Return a human-readable representation of the constraint.

        Returns
        -------
        str
            A readable representation in the form "expr1 sym expr2".
        """
        return f"{self.expr1} {self.sym} {self.expr2}"
