#
# Created in 2024 by Gaëtan Serré
#
from .operators import Min
import numpy as np
from .lcmaes_interface import *
import lcmaes
import cma


class PEP:
    def __init__(self, f):
        self.f = f
        self.initial_condition = None
        self.metric = []

    def set_initial_condition(self, constraint):
        self.initial_condition = constraint

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

            obj = -Min(self.metric).eval()

            constraints = (
                self.f.create_interpolation_constraints()
                + self.f.create_stationary_constraints()
            )
            constraints.append(self.initial_condition.c_eval())
            constraints = np.array(constraints)
            if verbose:
                print("Obj=", -obj, "Constraints=", constraints)

            if only_obj:
                return -obj

            lambda_ = np.zeros_like(constraints)
            lambda_[constraints > 0] = 1_000_000 * max(1, np.abs(obj))

            return obj + np.sum(lambda_ * constraints)

        n_comp = nb_points * d + nb_grads * d + nb_values
        x = list(np.random.uniform(-100, 100, n_comp))
        sigma = 100

        if backend == "C++":
            lambda_ = 50
            cmasols = pcmaes(
                to_fitfunc(F),
                to_params(
                    x,
                    lambda_,
                    sigma,  # all remaining parameters are optional
                    str_algo=b"vdcma",  # b=bytes, because unicode fails
                    xtolerance=tolx,
                ),
            )
            bcand = cmasols.best_candidate()
            bx = lcmaes.get_candidate_x(bcand)

            print(f"Run status: {cmasols.run_status()}.")

            return bx, -F(bx, verbose=True)

        elif backend == "Python":
            n_comp = nb_points * d + nb_grads * d + nb_values
            options = {
                "verbose": verbose,
                "tolx": tolx,
            }
            res = cma.fmin(
                F,
                x,
                sigma,
                options,
            )
            return res[0], F(res[0], only_obj=True)
        else:
            raise ValueError(f"Unknown {backend} backend")
