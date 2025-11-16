#
# Created in 2024 by Gaëtan Serré
#

from .expression import Expression, emax, emin
from .variable import Variable
from .const import Const
from .gpep import GPEP

__all__ = ["Expression", "Variable", "Const", "GPEP", "emax", "emin"]

__version__ = "0.0.1"
