#
# Created in 2024 by Gaëtan Serré
#
from ..variable import Variable
from ..expression import Expression


class Function:
    def __init__(self):
        self.init_counter = 0
        self.stat_counter = 0
        self.init_points = {}
        self.stat_points = {}
        self.init_values = {}
        self.stat_values = {}
        self.init_grad = {}
        self.stat_grad = {}

    @staticmethod
    def merge_dicts(d1, d2):
        d = d1.copy()
        d.update(d2)
        return d

    def __call__(self, v):
        if v.id in self.init_points:
            return self.init_values[f"f{v.id[1:]}"]
        else:
            return self.stat_values[f"fs{v.id[2:]}"]

    def gen_initial_point(self):
        v = Variable(f"x{self.init_counter}")
        fv = Variable(f"f{self.init_counter}").to_expr()
        gv = Variable(f"g{self.init_counter}").to_expr()
        self.init_points[f"x{self.init_counter}"] = v
        self.init_values[f"f{self.init_counter}"] = fv
        self.init_grad[f"g{self.init_counter}"] = gv
        self.init_counter += 1
        return v

    def get_stationary_point(self):
        v = Variable(f"xs{self.stat_counter}")
        fv = Variable(f"fs{self.stat_counter}").to_expr()
        gv = Variable(f"gs{self.stat_counter}").to_expr()
        self.stat_points[f"xs{self.stat_counter}"] = v
        self.stat_values[f"fs{self.stat_counter}"] = fv
        self.stat_grad[f"gs{self.stat_counter}"] = gv
        self.stat_counter += 1
        return v

    def grad(self, v):
        if v.id in self.init_points:
            return self.init_grad[f"g{v.id[1:]}"]
        else:
            return self.stat_grad[f"gs{v.id[2:]}"]

    def set_initial_points(self, points):
        for i, k in enumerate(self.init_points.keys()):
            self.init_points[k].set_value(points[i])

    def set_stationary_points(self, points):
        for i, k in enumerate(self.stat_points.keys()):
            self.stat_points[k].set_value(points[i])

    def set_initial_values(self, values):
        for i, k in enumerate(self.init_values.keys()):
            self.init_values[k].var.set_value(values[i])

    def set_stationary_values(self, values):
        for i, k in enumerate(self.stat_values.keys()):
            self.stat_values[k].var.set_value(values[i])

    def set_initial_grad(self, values):
        for i, k in enumerate(self.init_grad.keys()):
            self.init_grad[k].var.set_value(values[i])

    def set_stationary_grad(self, values):
        for i, k in enumerate(self.stat_grad.keys()):
            self.stat_grad[k].var.set_value(values[i])

    @staticmethod
    def get_value_id(id):
        if id.startswith("xs"):
            return f"fs{id[:2]}"
        else:
            return f"f{id[1:]}"

    @staticmethod
    def get_grad_id(id):
        if id.startswith("xs"):
            return f"gs{id[:2]}"
        else:
            return f"g{id[1:]}"

    def gen_constraint(self, x1, x2, f1, f2, g1, g2):
        pass

    def create_interpolation_constraints(self):
        points = self.merge_dicts(self.init_points, self.stat_points)
        values = self.merge_dicts(self.init_values, self.stat_values)
        gradients = self.merge_dicts(self.init_grad, self.stat_grad)

        constraints = []
        for k1, x1 in points.items():
            for k2, x2 in points.items():
                if k1 == k2:
                    continue
                x1 = x1.eval()
                x2 = x2.eval()
                f1 = values[self.get_value_id(k1)].eval()
                f2 = values[self.get_value_id(k2)].eval()
                g1 = gradients[self.get_grad_id(k1)].eval()
                g2 = gradients[self.get_grad_id(k2)].eval()
                constraints.append(self.gen_constraint(x1, x2, f1, f2, g1, g2))
        return constraints
