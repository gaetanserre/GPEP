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


def optimal_performance(L, n, d, tolx=1e-9, backend="Python", verbose=1):
    f = SmoothConvexFunction(L=L)

    xs = f.get_stationary_point()
    fs = f(xs)

    x0 = f.gen_initial_point()
    f0 = f(x0)

    pep = PEP(f)
    pep.set_initial_condition((f0 - fs) <= 1)

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
    print(f"(f0 - fs): {(f0 - fs).eval()}\n\n")
    return res


if __name__ == "__main__":
    args = cli()
    n = 2
    L = 3
    d = 5
    res = optimal_performance(L, n, d, args.tolx, args.backend, args.verbose)
    print(f"Optimal performance estimation: {res[1]}.")
