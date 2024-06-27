from math import sqrt

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction


def wc_optimized_gradient_for_gradient(L, n, wrapper="cvxpy", solver=None, verbose=1):

    # Instantiate PEP
    problem = PEP()

    # Declare a smooth convex function
    func = problem.declare_function(SmoothConvexFunction, L=L)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func(xs)

    # Then define x0 the starting point of the algorithm and its function value f(x_0)
    x0 = problem.set_initial_point()
    f0 = func(x0)

    # Set the initial constraint that is f(x_0) - f(x_*)
    problem.set_initial_condition(f0 - fs <= 1)

    # Compute scalar sequence of \tilde{theta}_t
    theta_tilde = [
        1
    ]  # compute \tilde{theta}_{t} from \tilde{theta}_{t+1} (sequence in reverse order)
    for i in range(n):
        if i < n - 1:
            theta_tilde.append((1 + sqrt(4 * theta_tilde[i] ** 2 + 1)) / 2)
        else:
            theta_tilde.append((1 + sqrt(8 * theta_tilde[i] ** 2 + 1)) / 2)
    theta_tilde.reverse()

    print(theta_tilde)

    # Run n steps of the optimized gradient method for gradient (OGM-G) method
    x = x0
    y_new = x0
    x_grad = func.gradient(x)

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
        x_grad = func.gradient(x)

    # Set the performance metric to the gradient norm
    problem.set_performance_metric(x_grad**2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = 2 * L / (theta_tilde[0] ** 2)

    # Print conclusion if required
    if verbose != -1:
        print(
            "*** Example file: worst-case performance of optimized gradient method for gradient ***"
        )
        print(
            "\tPEP-it guarantee:\t ||f'(x_n)||^2 <= {:.6} (f(x_0) - f_*)".format(
                pepit_tau
            )
        )
        print(
            "\tTheoretical guarantee:\t ||f'(x_n)||^2 <= {:.6} (f(x_0) - f_*)".format(
                theoretical_tau
            )
        )

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)

    print(f"Dimension of the optimal value: {x0.eval().shape}")
    print((f0 - fs).eval())
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_optimized_gradient_for_gradient(
        L=3, n=2, wrapper="cvxpy", solver=None, verbose=1
    )
