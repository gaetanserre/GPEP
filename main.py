from pkg.functions import *
from pkg import PEP
import argparse
from math import sqrt


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="Python")
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    n = 3
    L = 3
    gamma = 1 / L
    f = SmoothConvexFunction(L=L)
    xs = f.get_stationary_point()
    fs = f(xs)
    x0 = f.gen_initial_point()
    pep = PEP(f)
    pep.set_initial_condition((x0 - xs).norm() ** 2 <= 1)
    theta_new = 1
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * f.grad(y)
        theta_old = theta_new
        if i < n - 1:
            theta_new = (1 + sqrt(4 * theta_new**2 + 1)) / 2
        else:
            theta_new = (1 + sqrt(8 * theta_new**2 + 1)) / 2

        y = (
            x_new
            + (theta_old - 1) / theta_new * (x_new - x_old)
            + theta_old / theta_new * (x_new - y)
        )
    pep.set_metric(f(y) - fs)
    print(pep.solve(4, backend=args.backend))
    """ y = f(xs)
  pep = PEP(f)
  pep.solve(2)
  print(f(x0).eval()) """
