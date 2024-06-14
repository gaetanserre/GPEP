from pkg.expression import Expression
from pkg.variable import Variable
from pkg.functions import SmoothStronglyConvexFunction
from pkg import PEP
import numpy as np

gamma = 1
f = SmoothStronglyConvexFunction(1, 0.1)
x0 = f.gen_initial_point()
y0 = f.gen_initial_point()
x = x0 - gamma * f.grad(x0)
y = y0 - gamma * f.grad(y0)
pep = PEP(f)
pep.set_metric((x - y).norm() ** 2)
print(pep.solve(7))
""" y = f(xs)
pep = PEP(f)
pep.solve(2)
print(f(x0).eval()) """
