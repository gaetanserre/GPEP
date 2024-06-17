#
# Created in 2024 by Gaëtan Serré
#

import numpy as np


class Max:
    def __init__(self, le):
        self.le = le

    def eval(self):
        return np.max([e.eval() for e in self.le])
