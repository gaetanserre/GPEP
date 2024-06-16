from pkg.functions import *
from pkg import PEP
import argparse


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="C++")
    return parser.parse_args()


if __name__ == "__main__":
    args = cli()
    n = 4
    L = 3
    mu = 1
    gamma = 1 / L
    f = SmoothConvexFunction(L)
    x0 = f.gen_initial_point()
    xs = f.get_stationary_point()
    x = x0
    for _ in range(n):
        x = x - gamma * f.grad(x)
    pep = PEP(f)
    pep.set_metric((f(xs) - f(x)).abs())
    print(pep.solve(1, backend=args.backend))
    """ y = f(xs)
  pep = PEP(f)
  pep.solve(2)
  print(f(x0).eval()) """
