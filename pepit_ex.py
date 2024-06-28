from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_gradient_descent_lyapunov_1(
    L, gamma, n, wrapper="cvxpy", solver=None, verbose=1
):

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding gradient and function value gn and fn
    xn = problem.set_initial_point()
    gn, fn = func.oracle(xn)

    # Run the GD at iteration (n+1)
    xnp1 = xn - gamma * gn
    gnp1, fnp1 = func.oracle(xnp1)

    # Compute the Lyapunov function at iteration n and at iteration n+1
    init_lyapunov = n * (fn - fs) + L / 2 * (xn - xs) ** 2
    final_lyapunov = (n + 1) * (fnp1 - fs) + L / 2 * (xnp1 - xs) ** 2

    # Set the performance metric to the difference between the initial and the final Lyapunov
    problem.set_performance_metric(final_lyapunov - init_lyapunov)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    if gamma == 1 / L:
        theoretical_tau = 0.0
    else:
        theoretical_tau = None

    # Print conclusion if required
    if verbose != -1:
        print(
            "*** Example file:"
            " worst-case performance of gradient descent with fixed step-size for a given Lyapunov function***"
        )
        print("\tPEPit guarantee:\t" "V_(n+1) - V_(n) <= {:.6}".format(pepit_tau))
        if gamma == 1 / L:
            print(
                "\tTheoretical guarantee:\t"
                "V_(n+1) - V_(n) <= {:.6}".format(theoretical_tau)
            )

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)

    print(f"Dimension of the optimal solution: {xn.eval().shape}")
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    L = 1
    pepit_tau, theoretical_tau = wc_gradient_descent_lyapunov_1(
        L=L, gamma=1 / L, n=10, wrapper="cvxpy", solver=None, verbose=1
    )
