from pkg.functions import *
from pkg import PEP
import argparse
from math import sqrt


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="Python")
    return parser.parse_args()


def optimal_performance(L, n, d, backend="Python"):
    f = SmoothConvexFunction(L=L)

    xs = f.get_stationary_point()
    fs = f(xs)

    x0 = f.gen_initial_point()
    f0 = f(x0)

    pep = PEP(f)
    pep.set_initial_condition((f0 - fs) ** 2 <= 1)

    theta_tilde = [1]
    for i in range(n):
        if i < n - 1:
            theta_tilde.append((1 + sqrt(4 * theta_tilde[i] ** 2 + 1)) / 2)
        else:
            theta_tilde.append((1 + sqrt(8 * theta_tilde[i] ** 2 + 1)) / 2)
    theta_tilde.reverse()

    x = x0
    y_new = x0
    x_grad = f.grad(x)

    for i in range(n):
        y_old = y_new
        y_new = x - 1 / L * x_grad
        x = (
            y_new
            + (theta_tilde[i] - 1)
            * (2 * theta_tilde[i + 1] - 1)
            / theta_tilde[i]
            / (2 * theta_tilde[i] - 1)
            * (y_new - y_old)
            + (2 * theta_tilde[i + 1] - 1) / (2 * theta_tilde[i] - 1) * (y_new - x)
        )
        x_grad = f.grad(x)

    pep.set_metric(x_grad.norm() ** 2)
    res = pep.solve(d, backend)
    print(f.points, f.expr, (f0 - fs).eval())
    return res


if __name__ == "__main__":
    args = cli()
    n = 1
    L = 3
    d = 4
    print(optimal_performance(L, n, d, args.backend))
