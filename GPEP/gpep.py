#
# Created in 2024 by Gaëtan Serré
#

"""
Performance Estimation Problem (PEP) driver for GPEP.

This module provides the PEP class which orchestrates the construction and
numerical solution of a performance estimation problem derived from a Function
instance. It collects interpolation constraints, initial conditions and a
performance metric, then solves a finite-dimensional optimization problem
to estimate worst-case performance.

The implementation uses gob.optimizers for numerical minimization and expects
the Function instance to expose points, values and gradient proxies.
"""

from .expression import emin
import numpy as np
from gob.optimizers import CMA_ES
from gob.benchmarks import create_bounds


class GPEP:
    """GPEP problem manager.

    Parameters
    ----------
    f : Function
        Function instance providing sampled points, value and gradient proxies.

    Attributes
    ----------
    f : Function
        The managed Function instance.
    initial_conditions : list
        List of constraint objects representing initial problem conditions.
    metric : list
        List of Expression objects describing the performance metric to maximize/minimize.
    """

    def __init__(self, f):
        """Initialize the PEP driver.

        Parameters
        ----------
        f : Function
            Function instance for which to build the PEP.
        """
        self.f = f
        self.initial_conditions = []
        self.metric = []

    def set_initial_condition(self, constraint):
        """Add an initial condition constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint object describing an initial condition to include in the PEP.
        """
        self.initial_conditions.append(constraint)

    def set_metric(self, metric):
        """Set the performance metric to be evaluated.

        Parameters
        ----------
        metric : Expression
            Expression describing the performance metric (to be aggregated by emin).
        """
        self.metric.append(metric)

    def solve(self, opt=None, verbose=0):
        """Assemble and solve the finite-dimensional optimization representing the PEP.

        Parameters
        ----------
        opt : callable or None, optional
            Optimizer factory or None to use default CMA-ES optimizer.

            For a custom optimizer, the callable should accept bounds (dimension × (min, max))as input and return an optimizer instance with a minimize() method.
        verbose : int, optional
            Verbosity flag (print progress when non-zero).

        Returns
        -------
        tuple
            (x_opt, objective_value) where x_opt is the optimizer solution and objective_value
            is the evaluated objective at x_opt.

        Notes
        -----
        The method converts the abstract interpolation constraints into numeric constraints
        using the current proxy Variable values and minimizes a penalized objective via the
        chosen optimizer.
        """
        if verbose:
            print("Solving PEP...")
            print(self.f)

        nb_points = len(self.f.points)
        nb_values = len(self.f.values)
        nb_grads = len(self.f.grads)

        def F(x, only_obj=False, verbose=False):
            x = np.array(x)
            self.f.set_points(x[: nb_points * d].reshape(nb_points, d))
            thresh = nb_points * d
            self.f.set_grads(x[thresh : thresh + nb_grads * d].reshape(nb_grads, d))
            thresh += nb_grads * d
            self.f.set_values(x[thresh:])

            self.f.set_stat_grads(d)

            obj = -emin(self.metric).eval()

            abstract_constraints = self.f.create_interpolation_constraints()
            constraints = np.zeros(
                len(abstract_constraints) + len(self.initial_conditions)
            )
            for i, constraint in enumerate(
                abstract_constraints + self.initial_conditions
            ):
                constraints[i] = constraint.c_eval()
            if verbose:
                print("Obj=", -obj, "Constraints=", constraints)

            if only_obj:
                return -obj

            lambda_ = np.zeros_like(constraints)
            lambda_[constraints > 0] = 1e15 * max(1, np.abs(obj))

            return obj + np.sum(lambda_ * constraints)

        d = nb_points + nb_grads

        n_comp = nb_points * d + nb_grads * d + nb_values
        l, u = -10, 10
        bounds = create_bounds(n_comp, l, u)
        if opt is None:
            opt = CMA_ES(bounds, n_eval=250_000, sigma0=10)
        else:
            opt = opt(bounds)
        res = opt.minimize(F)
        return res[0], F(res[0], verbose=True, only_obj=True)

    def print_info(self):
        """Print a human-readable summary of current proxies and values.

        The method evaluates and prints points, expressions, values, gradients and
        their norms for debugging and reporting purposes.
        """
        f = self.f
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
            grads[k] = v.eval()
        grad_norms = {}
        for k, v in f.grads.items():
            grad_norms[k] = v.norm().eval()
        stat_grads = {}
        for k, v in f.stat_grads.items():
            stat_grads[k] = v.eval()
        stat_grads_norm = {}
        for k, v in f.stat_grads.items():
            stat_grads_norm[k] = v.norm().eval()
        print(f"\n\nPoints: {points}")
        print(f"Expr: {expr}")
        print(f"Values: {values}")
        print(f"Grads: {grads}")
        print(f"Grad norms: {grad_norms}")
        print(f"Stat grads: {stat_grads}")
        print(f"Stat grad norms: {stat_grads_norm}")
