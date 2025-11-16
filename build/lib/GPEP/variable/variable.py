#
# Created in 2024 by Gaëtan Serré
#

"""
Variable abstraction for GPEP.

This module defines the Variable class, a leaf Expression representing a named
variable with an assignable numeric value.
"""

from ..expression import Expression


class Variable(Expression):
    """
    Named variable expression.

    Parameters
    ----------
    id : str
        Identifier for the variable.
    value : number, optional
        Initial numeric value assigned to the variable. Default is None.

    Notes
    -----
    Variable.eval() raises ValueError if value is None.
    """

    def __init__(self, id, value=None):
        """
        Initialize a Variable.

        Parameters
        ----------
        id : str
            Variable identifier.
        value : number, optional
            Initial value for the variable.
        """
        super().__init__(self)
        self.id = id
        self.value = value

    def set_value(self, value):
        """
        Assign a numeric value to the variable.

        Parameters
        ----------
        value : number
            Value to assign to the variable.
        """
        self.value = value

    def eval(self):
        """
        Return the assigned value or raise if unset.

        Returns
        -------
        number
            The assigned numeric value.

        Raises
        ------
        ValueError
            If no value has been assigned.
        """
        if self.value is None:
            raise ValueError(f"Variable '{self.id}' has no value assigned.")
        return self.value

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        """
        Return a short human-readable representation of the variable.

        Returns
        -------
        str
            A string in the format "var(id)".
        """
        return f"var({self.id})"
