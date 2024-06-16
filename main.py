from pkg.functions import *
from pkg import PEP

gamma = 1 / 3
f = SmoothStronglyConvexFunction(3, 1)
x0 = f.gen_initial_point()
xs = f.get_stationary_point()
x = x0
for _ in range(1):
    x = x - gamma * f.grad(x)
pep = PEP(f)
pep.set_metric((f(xs) - f(x)).abs())
print(pep.solve(5, backend="Python"))
""" y = f(xs)
pep = PEP(f)
pep.solve(2)
print(f(x0).eval()) """
