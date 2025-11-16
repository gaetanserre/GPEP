#
# Created in 2024 by Gaëtan Serré
#


class Const:
    """
    A constant expression.

    Parameters
    ----------
    value : any
        The constant value.
    """

    def __init__(self, value):
        self.value = value

    def eval(self):
        return self.value

    def __str__(self):
        return f"{self.value}"
