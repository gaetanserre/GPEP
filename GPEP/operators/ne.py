"""

Inequality operator for Expressions.

Negation of Eq; provides boolean inequality and string form.
"""

from .eq import Eq


class Ne(Eq):
    """Inequality operator node (not equal)."""

    def eval(self, value):
        """Return negation of Eq.eval().

        Parameters
        ----------
        value : any

        Returns
        -------
        bool or ndarray
        """
        return not super().eval(value)

    def str(self, expr):
        """Return string representation for inequality.

        Parameters
        ----------
        expr : Expression

        Returns
        -------
        str
        """
        return f"({str(expr)} != {self.expr})"
