from pkg.functions import *
from pkg import PEP
from pkg.expression import emin, emax
import argparse
from math import sqrt
import numpy as np

from pkg.variable import Variable
from pkg.expression import Expression


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="Python")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--tolx", type=float, default=1e-9)
    return parser.parse_args()


def optimal_performance(
    L, mu, n, t, sigma, k, gamma, d, tolx=1e-9, backend="Python", verbose=1
):
    f = SmoothFunction(L=L)

    xs = f.get_stationary_point()
    fs = f(xs)

    x0s = [f.gen_initial_point() for _ in range(n)]
    kernel = lambda x, y: (-(x - y).norm() ** 2 / (2 * sigma**2)).exp()
    dkernel = lambda x, y: (x - y) * kernel(x, y) / sigma**2

    x = [x0s[i] for i in range(n)]
    for _ in range(t):
        updates = [0] * n
        for i in range(n):
            s = 0
            for j in range(n):
                s += -k * f.grad(x[j]) * kernel(x[i], x[j]) + dkernel(x[i], x[j])
            updates[i] = s / n
        for i in range(n):
            x[i] = x[i] + gamma * updates[i]

    """ x = []
    for _ in range(t):
        updates = []
        for i in range(n):
            s = 0
            for j in range(n):
                s += -f.grad(x0s[j])
            updates.append(s / n)
        for i in range(n):
            x.append(x0s[i] + gamma * updates[i]) """

    pep = PEP(f)
    dists = [(x0 - xs).norm() ** 2 for x0 in x0s]
    pep.set_initial_condition(emax(dists) <= 1)
    for x0 in x:
        pep.set_metric(f(x0) - fs)
    res = pep.solve(d, tolx, backend, verbose)

    pep.print_info()

    return res


if __name__ == "__main__":
    args = cli()
    n = 2
    t = 1
    L = 3
    mu = 0.1
    d = 4
    sigma = 0.1
    gamma = 1 / L
    k = 1
    res = optimal_performance(
        L, mu, n, t, sigma, k, gamma, d, args.tolx, args.backend, args.verbose
    )
    print(f"Optimal performance estimation: {res[1]}.")
