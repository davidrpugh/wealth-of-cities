"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-09-25

"""
import numpy as np
import sympy as sym

import model

# define parameters
f, beta, phi, tau, theta = sym.var('f, beta, phi, tau, theta')

# define variables
M, P, W, Y = sym.var('M, P, W, Y')

# extract data on total population
L = model.total_population[0]

# define model equations using derivation from paper (impose f=1; P[0]=1)
eqn1 = (1 / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * (W / P)**-theta * (Y / P) - phi * f
eqn2 = beta * L - (M / phi) * (theta / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * (W / P)**-theta * (Y / P)
eqn3 = Y - beta * L * W

equilibrium_system = sym.Matrix([eqn1, eqn2, eqn3])
equilibrium_jacobian = equilibrium_system.jacobian([Y, W, M])

args = (P, Y, W, M, f, beta, phi, tau, theta)
numeric_equilibrium_system = sym.lambdify(args, equilibrium_system,
                                          modules=[{'ImmutableMatrix': np.array}, "numpy"])
numeric_equilibrium_jacobian = sym.lambdify(args, equilibrium_jacobian,
                                            modules=[{'ImmutableMatrix': np.array}, "numpy"])

# compute analytic solution (returns a list of dicts) and extract variables
soln, = sym.solve([eqn1, eqn2, eqn3], Y, W, M, dict=True)

analytic_nominal_gdp = soln[Y]
analytic_nominal_wage = soln[W]
analytic_num_firms = soln[M]
