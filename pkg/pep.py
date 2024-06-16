#
# Created in 2024 by Gaëtan Serré
#
import numpy as np
import lcmaes
import cma


class PEP:
    def __init__(self, f):
        self.f = f
        self.metric = None

    def set_metric(self, metric):
        self.metric = metric

    def solve(self, d, backend="Python"):
        nb_points = len(self.f.points)
        nb_values = len(self.f.values)
        nb_grads = len(self.f.grads)

        def F(x, verbose=False):
            x = np.array(x)
            self.f.set_points(x[: nb_points * d].reshape(nb_points, d))
            thresh = nb_points * d
            self.f.set_grads(x[thresh : thresh + nb_grads * d].reshape(nb_grads, d))
            thresh += nb_grads * d
            self.f.set_values(x[thresh:])

            obj = -self.metric.eval()

            constraints = (
                self.f.create_interpolation_constraints()
                + self.f.create_stationary_constraints()
            )
            constraints.append(
                ((self.f.points["x0"] - self.f.points["x1"]).norm() ** 2 - 1).eval()
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

        n_comp = nb_points * d + nb_grads * d + nb_values
        x = [1] * n_comp
        sigma = 50
        max_iter = 10_000

        if backend == "C++":
            lambda_ = 50
            seed = 0
            p = lcmaes.make_simple_parameters(x, sigma, lambda_, seed)
            p.set_max_iter(max_iter)
            objfunc = lcmaes.fitfunc_pbf.from_callable(lambda x, n: F(x))
            cmasols = lcmaes.pcmaes(objfunc, p)
            bcand = cmasols.best_candidate()
            bx = lcmaes.get_candidate_x(bcand)

            return bx, -F(bx, verbose=True)
        elif backend == "Python":
            n_comp = nb_points * d + nb_grads * d + nb_values
            print(f"Number of components: {n_comp}")
            options = {
                "verbose": 1,
                "maxiter": max_iter,
                "tolx": 1e-8,
            }

            res = cma.fmin(
                F,
                x,
                sigma,
                options,
            )

            return res[0], -F(res[0], verbose=True)
        else:
            raise ValueError(f"Unknown {backend} backend")
