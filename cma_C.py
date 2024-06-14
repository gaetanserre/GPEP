import numpy as np
import lcmaes
import cma as CMA


def cma(f, x0, sigma, max_iter):
    lambda_ = 10
    seed = 0
    p = lcmaes.make_simple_parameters(x0, sigma, lambda_, seed)
    p.set_max_iter(max_iter)
    p.set_str_algo("acmaes")
    obj_func = lcmaes.fitfunc_pbf.from_callable(f)
    cma_sols = lcmaes.pcmaes(obj_func, p)
    bx = lcmaes.get_candidate_x(cma_sols.best_candidate())
    return bx, f(bx, len(bx))


""" def cma_bounded(f, bounds, x0, sigma, max_iter):
    lambda_ = 10
    seed = 0
    gp = lcmaes.make_genopheno_pwqb(bounds[0], bounds[1], len(x0))
    p = lcmaes.make_parameters_pwqb(x0, sigma, gp, lambda_, seed)
    p.set_max_iter(max_iter)
    obj_func = lcmaes.fitfunc_pbf.from_callable(lambda x, n: f(x))
    cma_sols = lcmaes.pcmaes_pwqb(obj_func, p)
    bx = lcmaes.get_best_candidate_pheno(cma_sols, gp)
    return bx, f(bx) """


""" y = 10
f = lambda x : y * np.dot(x, x)
bounds = [[1] * 10, [10] * 10]
x0 = list(np.random.uniform(-10, 10, 10))
sigma = 1
max_iter = 1000
print(cma(f, bounds, x0, sigma, max_iter)) """


MAX_EVALS = 5000


def F(x, n):
    x = np.array(x)
    gamma = 1
    mu = 0.1
    L = 1

    fy0 = x[0]
    fx0 = x[1]
    dim_point = int(len(x) / 4)
    gy0 = x[2 : 2 + dim_point]
    gx0 = x[2 + dim_point : 2 + 2 * dim_point]
    y0 = x[2 + 2 * dim_point : 2 + 3 * dim_point]
    x0 = x[2 + 3 * dim_point :]

    obj = -np.linalg.norm((x0 - gamma * gx0) - (y0 - gamma * gy0)) ** 2

    c1 = np.linalg.norm(x0 - y0) ** 2 - 1
    c2 = (
        fx0
        + np.dot(gx0, y0 - x0)
        + 1 / (2 * L) * np.linalg.norm(gy0 - gx0) ** 2
        + ((mu * L) / (2 * (L - mu)))
        * np.linalg.norm(x0 - y0 - (1 / L) * (gx0 - gy0)) ** 2
        - fy0
    )
    c3 = (
        fy0
        + np.dot(gy0, x0 - y0)
        + 1 / (2 * L) * np.linalg.norm(gx0 - gy0) ** 2
        + ((mu * L) / (2 * (L - mu)))
        * np.linalg.norm(y0 - x0 - (1 / L) * (gy0 - gx0)) ** 2
        - fx0
    )

    constraints = np.array([c1, c2, c3])
    lambda_ = np.zeros_like(constraints)
    lambda_[constraints > 0] = 1_000_000 * max(1, np.abs(obj))

    return obj + np.sum(lambda_ * constraints)


def F_verbose(x):
    x = np.array(x)
    gamma = 1
    mu = 0.1
    L = 1

    fy0 = x[0]
    fx0 = x[1]
    dim_point = int(len(x) / 4)
    gy0 = x[2 : 2 + dim_point]
    gx0 = x[2 + dim_point : 2 + 2 * dim_point]
    y0 = x[2 + 2 * dim_point : 2 + 3 * dim_point]
    x0 = x[2 + 3 * dim_point :]

    obj = -np.linalg.norm((x0 - gamma * gx0) - (y0 - gamma * gy0)) ** 2

    c1 = np.linalg.norm(x0 - y0) ** 2 - 1
    c2 = (
        fx0
        + np.dot(gx0, y0 - x0)
        + 1 / (2 * L) * np.linalg.norm(gy0 - gx0) ** 2
        + ((mu * L) / (2 * (L - mu)))
        * np.linalg.norm(x0 - y0 - (1 / L) * (gx0 - gy0)) ** 2
        - fy0
    )
    c3 = (
        fy0
        + np.dot(gy0, x0 - y0)
        + 1 / (2 * L) * np.linalg.norm(gx0 - gy0) ** 2
        + ((mu * L) / (2 * (L - mu)))
        * np.linalg.norm(y0 - x0 - (1 / L) * (gy0 - gx0)) ** 2
        - fx0
    )

    constraints = np.array([c1, c2, c3])
    return -obj, constraints


""" x0 = np.random.uniform(-10, 10, 5)
y0 = np.random.uniform(-10, 10, 5)
fx0 = np.random.uniform(-10, 10)
fy0 = np.random.uniform(-10, 10)
gx0 = np.random.uniform(-10, 10, 5)
gy0 = np.random.uniform(-10, 10, 5)
print(F(x0, y0, fx0, fy0, gx0, gy0, 1, 0.1, 1, verbose=True)) """

d = 7

res = cma(
    F,
    [0] * (4 * d + 2),
    1,
    MAX_EVALS,
)
""" d = 7
options = {
    # "bounds": [[0, 0, 0], [np.inf, np.inf, np.inf]],
    "verbose": 1,
    "maxiter": MAX_EVALS,
}
res = CMA.fmin(
    lambda x: F(
        x[:d],
        x[d : 2 * d],
        x[2 * d],
        x[2 * d + 1],
        x[2 * d + 2 : 3 * d + 2],
        x[3 * d + 2 :],
        1,
        0.1,
        1,
    ),
    [0] * (4 * d + 2),
    1,
    options,
) """

print(res)

print(F_verbose(res[0]))

""" x = res[0]

_ = F(
    x[:d],
    x[d : 2 * d],
    x[2 * d],
    x[2 * d + 1],
    x[2 * d + 2 : 3 * d + 2],
    x[3 * d + 2 :],
    1,
    0.1,
    1,
    verbose=True,
)
obj = F(
    x[:d],
    x[d : 2 * d],
    x[2 * d],
    x[2 * d + 1],
    x[2 * d + 2 : 3 * d + 2],
    x[3 * d + 2 :],
    1,
    0.1,
    1,
    only_obj=True,
)

print(x, obj) """
