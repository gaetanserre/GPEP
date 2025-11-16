# GPEP — Generalized Performance Estimation Problems

GPEP is an experimental Python framework for computer-assisted worst-case analyses
of first-order optimization methods. It is inspired by the [PEPit](https://github.com/PerformanceEstimation/PEPit) approach and
paper below, but adopts a more general symbolic expression representation
allowing arbitrary (nonlinear) expressions at the cost of relying on global
optimizers rather than convex solvers.

## Reference
B. Goujaud, C. Moucer, F. Glineur, J. Hendrickx, A. Taylor, A. Dieuleveut.
"PEPit: computer-assisted worst-case analyses of first-order optimization methods in Python."
Math. Prog. Comp. 16, 337–367 (2024). https://doi.org/10.1007/s12532-024-00259-7

## Installation
Clone the repository and install via pip:

```bash
git clone https://github.com/gaetanserre/GPEP.git
cd GPEP
pip install .
```

## Key ideas and differences vs PEPit
- **PEPit** restricts expressions to linear combinations of some symbolic values (see [[B. Goujaud et. al., 2024 – _Remark 1_]](https://link.springer.com/article/10.1007/s12532-024-00259-7)). That restriction enables reformulation as a semidefinite program and therefore the use of convex solvers with theoretical guarantees (exact worst-case bounds under the model).

- **GPEP** removes the linearity restriction: expressions in GPEP can be arbitrary symbolic compositions of variables, constants and operators. This permits a richer modeling language, but it generally prevents reduction to convex programs.

- Because of the above, GPEP uses global (nonconvex) optimization solvers to search for worst-case instances. As constrained optimization problems are generally not directly supported by many global optimizers, we reformulate constraints using Lagrangian penalties. The default solver is [CMA-ES](https://github.com/CMA-ES/libcmaes); other global optimizers can be plugged in.

- Trade-offs:
  - Pros: more flexible symbolic modeling; can express non-linear relationships.
  - Cons: slower (global search) and no formal guarantees of global optimality. Nevertheless, for some [standard examples](https://pepit.readthedocs.io/en/0.4.0/examples.html) the numerical results match PEPit closely.

<p align="center">
  <img src="assets/gd_gpep_pepit.svg" alt="gd_gpep_pepit" style="max-width:100%;height:auto;"/>
</p>

<p align="center">
Evolution of the performance of two steps of gradient descent with step-size γ on 1-smooth 0.1-strongly convex functions (see <a href="assets/compare_pepit.ipynb">compare_pepit.ipynb</a>).
</p>

## Symbolic expression representation
- Core abstraction: an `GPEP.Expression` wraps a base variable or constant and a list of operator nodes. Operators (`Add`, `Sub`, `Dot`, `Norm`, `Exp`, `Log`, ...) implement two methods:
  - `eval(value)` — apply the operator to a numeric/array value.
  - `str(expr)` — pretty-print the operator applied to an inner expression.

- Expressions are built lazily via Python operator overloading (e.g. `x + y`, `g.dot(x - y)`, `g.norm()**2`). Overloads return new `Expression` objects that append operator nodes to the `op_list`.

- Evaluation: calling `expr.eval()` evaluates the base variable/constant and applies the operator sequence in order, producing a concrete numeric result.

- String rendering: `__str__` folds operator `str()` calls to obtain a human-readable symbolic form.

## Function and PEP roles
- `GPEP.Function` manages sampled points, proxy variables for function values and gradients, and registers expressions encountered while simulating the algorithm. It exposes methods to produce interpolation constraints (one-point and two-point) that encode the functional assumptions being used (smoothness, convexity, Lipschitz gradient, etc.).

- Several `Function` subclasses implement common models:
  - `SmoothFunction`, `SmoothConvexFunction`, `ConvexFunction`, `ConvexLipschitzFunction`, `SmoothStronglyConvexFunction`
  - These generate the appropriate interpolation constraints (see [[Taylor et al., 2027]](https://arxiv.org/abs/1512.07516)).

- `GPEP.GPEP` orchestrates assembling all interpolation constraints, initial conditions, and a performance metric. It then converts the abstract constraints into a finite-dimensional optimization problem over proxy variables and solves it numerically using a chosen global optimizer.

## Solvers
- Default: CMA-ES via the [`GOB`](https://github.com/gaetanserre/GOB) package.
- You may substitute other global optimizers supported by `gob.optimizers` or implement your own.
- Because the problem is nonconvex in general (arbitrary expressions), the solver may return locally optimal solutions; treat results as numerical evidence rather than formal proofs.

## Example usage
Estimate the optimal performance of the [SBS](https://proceedings.mlr.press/v258/serre25a.html) method on smooth strongly convex functions.

```python
from GPEP import GPEP, emax
from GPEP.functions import SmoothStronglyConvexFunction

f = SmoothStronglyConvexFunction(L=3, mu=0.1)

xs = f.get_stationary_point()
fs = f(xs)
sigma = 0.1  # kernel bandwidth
n = 2  # number of particles
t = 1  # number of iterations

x0s = [f.gen_initial_point() for _ in range(n)]
kernel = lambda x, y: (-(x - y).norm() ** 2 / (2 * sigma**2)).exp()
dkernel = lambda x, y: (x - y) * kernel(x, y) / sigma**2

x = [x0s[i] for i in range(n)]
for _ in range(t):
    updates = [0] * n
    for i in range(n):
        s = 0
        for j in range(n):
            s += -f.grad(x[j]) * kernel(x[i], x[j]) + dkernel(x[i], x[j])
        updates[i] = s / n
    for i in range(n):
        x[i] = x[i] + (1 / 3) * updates[i]

pep = GPEP(f)

# Initial conditions: all starting points within distance 1 from a stationary point
dists = [(x0 - xs).norm() ** 2 for x0 in x0s]
pep.set_initial_condition(emax(dists) <= 1)

# Performance metric: largest distance to the image of a stationary point
for x0 in x:
    pep.set_metric(f(x0) - fs)
pep.solve()
```

Example output:
```
Obj= 0.7402156166826082 Constraints= [-1.33673401e-04 -1.42346604e-07 -1.25881515e-07 -5.57515870e-08
 -7.24200370e-01 -6.09628122e-01 -1.74985274e-07 -6.14423586e-01
 -7.24537620e-01 -6.26604690e-01 -6.30184432e-01 -1.10369773e-08
 -3.64938676e-01 -6.45185944e-08 -3.02703336e-01 -3.05173359e-01
 -3.52329877e-01 -3.05481968e-01 -8.61723646e-08 -3.07242702e-01
 -6.47282964e-08]
```

## Notes
- This project is experimental and intended for research / prototyping.
- If you need formal worst-case certificates, prefer the convex-restricted approach implemented in [PEPit](https://github.com/PerformanceEstimation/PEPit).

