#
# Created in 2024 by Gaëtan Serré
#
from ..variable import Variable


class Function:
    def __init__(self):
        self.point_counter = 0
        self.points = {}
        self.values = {}
        self.grads = {}
        self.stat_ids = []

        self.expr_counter = 0
        self.expr = {}

    def add_point_(self, v):
        self.points[v.id] = v
        fv = Variable(f"f_{v.id}").to_expr()
        gv = Variable(f"g_{v.id}").to_expr()
        self.values[v.id] = fv
        self.grads[v.id] = gv
        return fv, gv

    def add_expr_(self, e):
        self.expr[f"e{self.expr_counter}"] = e
        fe = Variable(f"f_e{self.expr_counter}").to_expr()
        ge = Variable(f"g_e{self.expr_counter}").to_expr()
        self.values[f"e{self.expr_counter}"] = fe
        self.grads[f"e{self.expr_counter}"] = ge
        self.expr_counter += 1
        return fe, ge

    def __call__(self, v):
        if isinstance(v, Variable) and v.id in self.points:
            return self.values[v.id]
        else:
            fv, _ = self.add_expr_(v)
            return fv

    def grad(self, v):
        if isinstance(v, Variable) and v.id in self.points:
            return self.grads[v.id]
        else:
            _, gv = self.add_expr_(v)
            return gv

    def gen_initial_point(self):
        v = Variable(f"x{self.point_counter}")
        self.add_point_(v)
        self.point_counter += 1
        return v

    def get_stationary_point(self):
        self.stat_ids.append(self.point_counter)
        return self.gen_initial_point()

    def set_points(self, points):
        for i, k in enumerate(self.points.keys()):
            self.points[k].set_value(points[i])

    def set_values(self, values):
        for i, k in enumerate(self.values.keys()):
            self.values[k].var.set_value(values[i])

    def set_grads(self, values):
        for i, k in enumerate(self.grads.keys()):
            self.grads[k].var.set_value(values[i])

    def gen_constraint(self, x1, x2, f1, f2, g1, g2):
        pass

    @staticmethod
    def merge_dicts(d1, d2):
        d = d1.copy()
        d.update(d2)
        return d

    def create_interpolation_constraints(self):
        points = self.merge_dicts(self.points, self.expr)
        constraints = []
        for k1, x1 in points.items():
            for k2, x2 in points.items():
                if k1 == k2:
                    continue
                constraints.append(
                    self.gen_constraint(
                        x1.eval(),
                        x2.eval(),
                        self.values[k1].eval(),
                        self.values[k2].eval(),
                        self.grads[k1].eval(),
                        self.grads[k2].eval(),
                    )
                )
        return constraints
