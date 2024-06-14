from pkg.functions import *
from pkg import PEP

gamma = 1
f = SmoothStronglyConvexFunction(1, 0.5)
x0 = f.gen_initial_point()
y0 = f.gen_initial_point()
x = x0
y = y0
for _ in range(4):
    x = x - gamma * f.grad(x)
    y = y - gamma * f.grad(y)
pep = PEP(f)
pep.set_metric((x - y).norm() ** 2)
print(pep.solve(5, backend="Python"))
""" y = f(xs)
pep = PEP(f)
pep.solve(2)
print(f(x0).eval()) """
