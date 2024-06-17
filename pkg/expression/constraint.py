#
# Created in 2024 by Gaëtan Serré
#


class Constraint:
    def __init__(self, expr1, expr2, op):
        self.expr1 = expr1
        self.expr2 = expr2
        self.op = op

    def eval(self):
        return self.op(self.expr1.eval(), self.expr2.eval())

    def c_eval(self):
        return self.expr1.eval() - self.expr2.eval()
