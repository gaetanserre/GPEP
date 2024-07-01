#
# Created in 2024 by Gaëtan Serré
#
from .operator import Operator
import numpy as np


class Log(Operator):
    def __init__(self):
        super().__init__()

    def eval(self, value):
        return np.log(value)
