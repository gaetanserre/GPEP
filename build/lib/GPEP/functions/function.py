#
# Created in 2024 by Gaëtan Serré
#

"""
Function utilities for GPEP.

This module provides the Function class used to manage sampled points, their
function and gradient proxies, and to generate interpolation constraints for
the solver layer.
"""

from ..variable import Variable
import numpy as np


class Function:
    """
    Container for function interpolation data and related metadata.

    Parameters
    ----------
    name : str
        Short descriptive name for the function instance.

    Attributes
    ----------
    point_counter : int
        Counter used to name generated point variables.
    points : dict
        Mapping point id -> Variable (input points).
    values : dict
        Mapping point id -> Variable (function value proxies).
    grads : dict
        Mapping point id -> Variable (gradient proxies for non-stationary points).
    stat_grads : dict
        Mapping point id -> Variable (gradient proxies for stationary points).
    expr : dict
        Mapping expression id -> expression object for expressions tracked by the Function.
    hash_to_id : dict
        Mapping hash(expression) -> expression id used to reuse proxies for identical expressions.
    """

    def __init__(self, name):
        """
        Initialize a Function container.

        Parameters
        ----------
        name : str
            Name of the function for identification/logging.
        """
        self.name = name

        self.point_counter = 0
        self.points = {}
        self.values = {}
        self.grads = {}
        self.stat_grads = {}

        self.expr_counter = 0
        self.expr = {}
        self.hash_to_id = {}

    def __str__(self):
        """
        Return a short human-readable summary of the Function.

        Returns
        -------
        str
            Multi-line summary listing initial and stationary points and interpolation constraints.
        """
        n_stat_points = self.get_nb_stat_points()
        n_initial_points = self.get_nb_init_points()

        s = f"{self.name} Function:\n"
        if n_initial_points == 0:
            s_init_points = "\tNo initial point(s)\n"
        else:
            s_init_points = f"\t{n_initial_points} initial point(s):\n"
        if n_stat_points == 0:
            s_stats_points = "\tNo stationary point(s)\n"
        else:
            s_stats_points = f"\t{n_stat_points} stationary point(s):\n"
        for v in self.points.values():
            if v.id in self.grads:
                s_init_points += (
                    f" \t\t• {v}: Value {self.values[v.id]} - Grad {self.grads[v.id]}\n"
                )
            elif v.id in self.stat_grads:
                s_stats_points += f" \t\t• {v}: Value {self.values[v.id]} - Zero grad {self.stat_grads[v.id]}\n"

        constraints = self.create_interpolation_constraints()
        if len(constraints) == 0:
            s_constraints = "\tNo interpolation constraint(s)\n"
        else:
            s_constraints = f"\t{len(constraints)} interpolation constraint(s):\n"
        for c in constraints:
            s_constraints += f" \t\t• {c}\n"

        return s + s_init_points + s_stats_points + s_constraints

    def get_nb_stat_points(self):
        """
        Return the number of stationary points tracked.

        Returns
        -------
        int
            Number of stationary (zero-gradient) points.
        """
        return len(self.stat_grads)

    def get_nb_init_points(self):
        """
        Return the number of initial (non-stationary) points tracked.

        Returns
        -------
        int
            Number of initial points.
        """
        return len(self.points) - self.get_nb_stat_points()

    def add_point_(self, v):
        """
        Register a new sampled point and create proxies for its value and gradient.

        Parameters
        ----------
        v : Variable
            Point variable to register.

        Returns
        -------
        (Variable, Variable)
            Tuple (f_v, g_v) where f_v is the value proxy and g_v the gradient proxy.
        """
        self.points[v.id] = v
        fv = Variable(f"f_{v.id}")
        gv = Variable(f"g_{v.id}")
        self.values[v.id] = fv
        self.grads[v.id] = gv
        return fv, gv

    def add_expr_(self, e):
        """
        Register an arbitrary expression and create proxies for its value and gradient.

        Parameters
        ----------
        e : Expression
            Expression to register.

        Returns
        -------
        (Variable, Variable)
            Tuple (f_e, g_e) proxies for the expression.
        """
        self.expr[f"e{self.expr_counter}"] = e
        fe = Variable(f"f_e{self.expr_counter}")
        ge = Variable(f"g_e{self.expr_counter}")
        self.values[f"e{self.expr_counter}"] = fe
        self.grads[f"e{self.expr_counter}"] = ge
        self.hash_to_id[hash(e)] = f"e{self.expr_counter}"
        self.expr_counter += 1
        return fe, ge

    def __call__(self, v):
        """
        Return the value proxy corresponding to a point or expression.

        Parameters
        ----------
        v : Variable or Expression
            Input variable or expression.

        Returns
        -------
        Variable
            The value proxy Variable associated with v.
        """
        if isinstance(v, Variable) and v.id in self.points:
            return self.values[v.id]
        else:
            if hash(v) in self.hash_to_id:
                id = self.hash_to_id[hash(v)]
                return self.values[id]
            else:
                fv, _ = self.add_expr_(v)
                return fv

    def grad(self, v):
        """
        Return the gradient proxy corresponding to a point or expression, creating one if needed.

        Parameters
        ----------
        v : Variable or Expression
            Input variable or expression.

        Returns
        -------
        Variable
            The gradient proxy Variable associated with v.
        """
        if isinstance(v, Variable):
            if v.id in self.grads:
                return self.grads[v.id]
            elif v.id in self.stat_grads:
                return self.stat_grads[v.id]
        else:
            if hash(v) in self.hash_to_id:
                id = self.hash_to_id[hash(v)]
                return self.grads[id]
            else:
                _, gv = self.add_expr_(v)
                return gv

    def gen_initial_point(self):
        """
        Generate and register a new initial point variable.

        Returns
        -------
        Variable
            The newly created point variable.
        """
        v = Variable(f"x{self.point_counter}")
        self.add_point_(v)
        self.point_counter += 1
        return v

    def get_stationary_point(self):
        """
        Generate and register a new stationary point (zero-gradient proxy).

        Returns
        -------
        Variable
            The newly created stationary point variable.
        """
        v = Variable(f"x{self.point_counter}")
        self.points[v.id] = v
        fv = Variable(f"f_{v.id}")
        gv = Variable(f"g_{v.id}")
        self.values[v.id] = fv
        self.stat_grads[v.id] = gv
        self.point_counter += 1
        return v

    def set_points(self, points):
        """
        Assign numeric values to tracked input points.

        Parameters
        ----------
        points : iterable
            Values to assign in the order of self.points keys.
        """
        for i, k in enumerate(self.points.keys()):
            self.points[k].set_value(points[i])

    def set_values(self, values):
        """
        Assign numeric values to value proxies.

        Parameters
        ----------
        values : iterable
            Values assigned to value proxy Variables in the order of self.values keys.
        """
        for i, k in enumerate(self.values.keys()):
            self.values[k].var.set_value(values[i])

    def set_grads(self, values):
        """
        Assign numeric vectors to gradient proxies.

        Parameters
        ----------
        values : iterable
            Gradient values assigned to gradient proxy Variables.
        """
        for i, k in enumerate(self.grads.keys()):
            self.grads[k].var.set_value(values[i])

    def set_stat_grads(self, d):
        """
        Initialize stationary gradient proxies to zero vectors of dimension d.

        Parameters
        ----------
        d : int
            Dimension of the zero gradient vectors.
        """
        for k in self.stat_grads.keys():
            self.stat_grads[k].var.set_value(np.zeros(d))

    @staticmethod
    def merge_dicts(d1, d2):
        """
        Return a shallow merge of two dictionaries; d2 overwrites d1 on conflicts.

        Parameters
        ----------
        d1, d2 : dict
            Dictionaries to merge.

        Returns
        -------
        dict
            Shallow copy of merged dictionaries.
        """
        d = d1.copy()
        d.update(d2)
        return d

    def gen_1_point_constraint(self, x, f, g):
        pass

    def gen_2_points_constraint(self, x1, x2, f1, f2, g1, g2):
        pass

    def create_interpolation_constraints(self):
        """
        Create interpolation constraints for all tracked points and expressions.

        The method combines registered points and expressions and generates one-point
        and two-point interpolation constraints using gen_1_point_constraint and
        gen_2_points_constraint. It returns the list of constructed constraints.

        Returns
        -------
        list
            List of constraints representing interpolation relations.
        """
        points = self.merge_dicts(self.points, self.expr)
        grads = self.merge_dicts(self.grads, self.stat_grads)
        constraints = []
        for k1, x1 in points.items():
            f1 = self.values[k1]
            g1 = grads[k1]
            c1 = self.gen_1_point_constraint(x1, f1, g1)
            if c1 is not None:
                constraints.append(c1)
            for k2, x2 in points.items():
                if k1 == k2:
                    continue
                constraints.append(
                    self.gen_2_points_constraint(
                        x1,
                        x2,
                        f1,
                        self.values[k2],
                        g1,
                        grads[k2],
                    )
                )
        return constraints
