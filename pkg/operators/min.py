#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


class Min:
    def __init__(self, le):
        self.le = le

    def eval(self):
        return np.min([e.eval() for e in self.le])
