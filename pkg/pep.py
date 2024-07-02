#
# Created in 2024 by Gaëtan Serré
#
from .expression import emin
import numpy as np
import cma


class PEP:
    def __init__(self, f):
        self.f = f
        self.initial_conditions = []
        self.metric = []

    def set_initial_condition(self, constraint):
        self.initial_conditions.append(constraint)

    def set_metric(self, metric):
        self.metric.append(metric)

    def solve(self, d, tolx=1e-9, backend="Python", verbose=1):
        nb_points = len(self.f.points)
        nb_values = len(self.f.values)
        nb_grads = len(self.f.grads)

        def F(x, only_obj=False, verbose=False):
            x = np.array(x)
            self.f.set_points(x[: nb_points * d].reshape(nb_points, d))
            thresh = nb_points * d
            self.f.set_grads(x[thresh : thresh + nb_grads * d].reshape(nb_grads, d))
            thresh += nb_grads * d
            self.f.set_values(x[thresh:])

            self.f.set_stat_grads(d)

            obj = -emin(self.metric).eval()

            constraints = self.f.create_interpolation_constraints()
            for init_constraint in self.initial_conditions:
                constraints.append(init_constraint.c_eval())
            constraints = np.array(constraints)
            if verbose:
                print("Obj=", -obj, "Constraints=", constraints)

            if only_obj:
                return -obj

            lambda_ = np.zeros_like(constraints)
            lambda_[constraints > 0] = 1e15 * max(1, np.abs(obj))

            return obj + np.sum(lambda_ * constraints)

        n_comp = nb_points * d + nb_grads * d + nb_values
        l, u = -10, 10
        x = list(np.random.uniform(l, u, n_comp))
        sigma = 10

        lo, up = [], []
        for _ in range(n_comp):
            lo.append(l)
            up.append(u)
        if backend == "Python":
            n_comp = nb_points * d + nb_grads * d + nb_values
            options = {
                "verbose": verbose,
                # "bounds": [lo, up],
                "tolx": tolx,
            }
            res = cma.fmin(
                F,
                x,
                sigma,
                options,
            )
            return res[0], F(res[0], verbose=True, only_obj=True)
        else:
            raise ValueError(f"Unknown {backend} backend")

    def print_info(self):
        f = self.f
        points = {}
        for k, v in f.points.items():
            points[k] = v.eval()
        expr = {}
        for k, v in f.expr.items():
            expr[k] = v.eval()
        values = {}
        for k, v in f.values.items():
            values[k] = v.eval()
        grads = {}
        for k, v in f.grads.items():
            grads[k] = v.eval()
        grad_norms = {}
        for k, v in f.grads.items():
            grad_norms[k] = v.norm().eval()
        stat_grads = {}
        for k, v in f.stat_grads.items():
            stat_grads[k] = v.eval()
        stat_grads_norm = {}
        for k, v in f.stat_grads.items():
            stat_grads_norm[k] = v.norm().eval()
        print(f"\n\nPoints: {points}")
        print(f"Expr: {expr}")
        print(f"Values: {values}")
        print(f"Grads: {grads}")
        print(f"Grad norms: {grad_norms}")
        print(f"Stat grads: {stat_grads}")
        print(f"Stat grad norms: {stat_grads_norm}")
