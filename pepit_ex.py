from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, SmoothConvexFunction


def wc_gradient_descent(L, mu, gamma, n, wrapper="cvxpy", solver=None, verbose=1):

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()
    y0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    x = x0
    for _ in range(n):
        x = x - gamma * func.gradient(x)

    # Set the performance metric to the function values accuracy
    problem.set_performance_metric((func(x) - fs))

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    print(x0.eval())
    print(xs.eval())
    print(func(x0).eval())
    print(func(x).eval())
    print(func(xs).eval())
    print(func.gradient(x0).eval())
    print(func.gradient(x).eval())
    print(func.gradient(xs).eval())

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * (2 * n * L * gamma + 1))

    # Print conclusion if required
    """ if verbose != -1:
        print(
            "*** Example file: worst-case performance of gradient descent with fixed step-sizes ***"
        )
        print(
            "\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2".format(pepit_tau)
        )
        print(
            "\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2".format(
                theoretical_tau
            )
        )
 """
    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 3
    pepit_tau, theoretical_tau = wc_gradient_descent(
        L=L, mu=1, gamma=1 / L, n=1, wrapper="cvxpy", solver=None, verbose=1
    )
