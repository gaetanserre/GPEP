from pkg.functions import *
from pkg import PEP
import argparse
from math import sqrt


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="Python")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--tolx", type=float, default=1e-9)
    return parser.parse_args()


def optimal_performance(L, n, gamma, d, tolx=1e-9, backend="Python", verbose=1):
    f = SmoothConvexFunction(L=L)

    xs = f.get_stationary_point()
    fs = f(xs)

    xn = f.gen_initial_point()
    fn, gn = f.oracle(xn)

    xnp1 = xn - gamma * gn
    fnp1 = f(xnp1)

    init_lyapunov = n * (fn - fs) + (L / 2) * (xn - xs).norm() ** 2
    final_lyapunov = (n + 1) * (fnp1 - fs) + (L / 2) * (xnp1 - xs).norm() ** 2

    pep = PEP(f)
    # pep.set_initial_condition(gn.norm() <= 1e-15)
    # pep.set_initial_condition(final_lyapunov <= 100)

    pep.set_metric(final_lyapunov - init_lyapunov)
    res = pep.solve(d, tolx, backend, verbose)

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
        grads[k] = (v.norm()).eval()
    print(f"\n\nPoints: {points}")
    print(f"Expr: {expr}")
    print(f"Values: {values}")
    print(f"Grads: {grads}")
    print(f"Lyapunov: {final_lyapunov.eval(), init_lyapunov.eval()}")
    return res


if __name__ == "__main__":
    args = cli()
    n = 10
    L = 1
    d = 1
    gamma = 1 / L
    res = optimal_performance(L, gamma, n, d, args.tolx, args.backend, args.verbose)
    print(f"Optimal performance estimation: {res[1]}.")
