"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-09-15

"""
import numpy as np
import sympy as sym

import model

# define parameters
S, f, phi, theta = sym.var('S, f, phi, theta')

# define variables
P, Y, W, M = sym.var('P, Y, W, M')

# define model equations using derivation from paper (impose f=1; P[0]=1)
eqn0 = 1 - 1
eqn1 = phi * f - (1 / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * (W / P)**-theta * (Y / P)
eqn2 = S - (M / phi) * (theta / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * (W / P)**-theta * (Y / P)
eqn3 = Y - S * W

equilibrium_system = sym.Matrix([eqn0, eqn1, eqn2, eqn3])
equilibrium_jacobian = equilibrium_system.jacobian([P, Y, W, M])

args = (P, Y, W, M, f, phi, theta, S)
numeric_equilibrium_system = sym.lambdify(args, equilibrium_system,
                                          modules=[{'ImmutableMatrix': np.array}, "numpy"])
numeric_equilibrium_jacobian = sym.lambdify(args, equilibrium_jacobian,
                                            modules=[{'ImmutableMatrix': np.array}, "numpy"])

# compute analytic solution (returns a list of dicts) and extract variables
#soln, = sym.solve([eqn0, eqn1, eqn2], Y, W, M, dict=True)

#analytic_nominal_gdp = soln[Y]
#analytic_nominal_wage = soln[W]
#analytic_num_firms = soln[M]
