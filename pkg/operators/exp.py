#
# Created in 2024 by Gaëtan Serré
#
from .operator import Operator
import numpy as np


class Exp(Operator):
    def __init__(self):
        super().__init__()

    def eval(self, value):
        return np.exp(value)

    def str(self, expr):
        return f"exp({expr})"
