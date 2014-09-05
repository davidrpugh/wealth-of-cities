"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-09-05

"""
import sympy as sym

# define parameters
S, phi, theta = sym.var('S, phi, theta')

# define variables
Y, W, M = sym.var('Y, W, M')

# define model equations (impose f=1; P[0]=1)
eqn0 = Y - S * W
eqn1 = phi - (1 / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * W**-theta * Y
eqn2 = S - (M / phi) * (theta / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta * W**-theta * Y

# compute analytic solution (returns a list of dicts) and extract variables
soln, = sym.solve([eqn0, eqn1, eqn2], Y, W, M, dict=True)

analytic_nominal_gdp = soln['Y']
analytic_nominal_wage = soln['W']
analytic_num_firms = soln['M']
