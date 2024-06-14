#
# Created in 2024 by Gaëtan Serré
#
import numpy as np
import cma


class PEP:
    def __init__(self, f):
        self.f = f
        self.metric = None

    def set_metric(self, metric):
        self.metric = metric

    def solve(self, d):
        nb_init = len(self.f.init_points)
        nb_stat = len(self.f.stat_points)

        def F(x, verbose=False):
            self.f.set_initial_points(x[: nb_init * d].reshape(nb_init, d))
            thresh = nb_init * d
            self.f.set_stationary_points(
                x[thresh : thresh + nb_stat * d].reshape(nb_stat, d)
            )
            thresh += nb_stat * d
            self.f.set_initial_grad(
                x[thresh : thresh + nb_init * d].reshape(nb_init, d)
            )
            thresh += nb_init * d
            self.f.set_stationary_grad(
                x[thresh : thresh + nb_stat * d].reshape(nb_stat, d)
            )
            thresh += nb_stat * d
            self.f.set_initial_values(x[thresh : thresh + nb_init])
            thresh += nb_init
            self.f.set_stationary_values(x[thresh:])

            obj = -self.metric.eval()

            constraints = self.f.create_interpolation_constraints()
            constraints.append(
                (
                    (self.f.init_points["x0"] - self.f.init_points["x1"]).norm() ** 2
                    - 1
                ).eval()
            )
            constraints = np.array(constraints)
            if verbose:
                print("Obj=", obj, "Constraints=", constraints)

            lambda_ = np.zeros_like(constraints)
            lambda_[constraints > 0] = 1_000_000 * max(1, np.abs(obj))

            return obj + np.sum(lambda_ * constraints)

        """ x = np.array(
            [
                1.33843546,
                -0.84380117,
                -0.57861767,
                -0.32693879,
                -0.07581888,
                0.5004061,
                0.72409807,
                1.44746625,
                -0.25373331,
                -0.53517332,
                -0.90941674,
                -0.1298704,
                0.79151394,
                1.18355305,
                -0.18684466,
                -0.20123535,
                1.58393524,
                1.22591515,
                0.59257516,
                0.80882403,
                -1.76633809,
                -0.17594056,
                -0.1422258,
                1.58828004,
                1.16766711,
                0.58716871,
                0.83793365,
                -1.72039615,
                -1.97960223,
                -3.32209948,
            ]
        )

        print(F(x, verbose=True)) """

        n_comp = nb_init * (2 * d + 1) + nb_stat * (2 * d + 1)

        options = {
            # "bounds": [[0, 0, 0], [np.inf, np.inf, np.inf]],
            "verbose": 1,
            "maxiter": 5000,
        }

        res = cma.fmin(
            F,
            [0] * n_comp,
            1,
            options,
        )

        return res[0], -F(res[0], verbose=True)
