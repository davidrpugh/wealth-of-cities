"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-10-01

"""
import numpy as np
import sympy as sym

import model

# define parameters
f, beta, phi, tau, theta = sym.var('f, beta, phi, tau, theta')

# define variables
L, M, P, W, Y = sym.var('L, M, P, W, Y')

# define model equations using derivation from paper
eqn1 = (1 / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * (W / P)**-theta * (Y / P) - phi * f
eqn2 = beta * L - (M / phi) * (theta / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * (W / P)**-theta * (Y / P)
eqn3 = Y - beta * L * W

# compute analytic solution (returns a list of dicts) and extract variables
soln, = sym.solve([eqn1, eqn2, eqn3], Y, W, M, dict=True)

# wrap the analytic solutions
args = (P, L, f, beta, phi, tau, theta)
modules = [{'ImmutableMatrix': np.array}, "numpy"]
numeric_nominal_gdp = sym.lambdify(args, soln[Y], modules)
numeric_nominal_wage = sym.lambdify(args, soln[W], modules)
numeric_num_firms = sym.lambdify(args, soln[M], modules)


def get_initial_guess(num_cities, f, beta, phi, tau, theta):
    """Returns an initial guess for a given number of cities."""

    P0 = np.repeat(1.0, num_cities-1)
    Y0 = numeric_nominal_gdp(1.0, model.total_population[:num_cities],
                             f, beta, phi, tau, theta)
    W0 = numeric_nominal_wage(1.0, model.total_population[:num_cities],
                              f, beta, phi, tau, theta)
    M0 = numeric_num_firms(1.0, model.total_population[:num_cities:],
                           f, beta, phi, tau, theta)
    initial_guess = np.hstack((P0, Y0, W0, M0))
    return initial_guess
