from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_optimized_gradient(L, n, wrapper="cvxpy", solver=None, verbose=1):
    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then Define the starting point of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the optimized gradient method (OGM) method
    theta_new = 1
    x_new = x0
    y = x0
    for i in range(n):
        x_old = x_new
        x_new = y - 1 / L * func.gradient(y)
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

    # Set the performance metric to the function value accuracy
    problem.set_performance_metric(func(y) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * theta_new**2)

    # Print conclusion if required
    if verbose != -1:
        print(
            "*** Example file: worst-case performance of optimized gradient method ***"
        )
        print(
            "\tPEPit guarantee:\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2".format(pepit_tau)
        )
        print(
            "\tTheoretical guarantee:\t f(y_n)-f_* <= {:.6} ||x_0 - x_*||^2".format(
                theoretical_tau
            )
        )

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_optimized_gradient(
        L=3, n=3, wrapper="cvxpy", solver=None, verbose=1
    )
