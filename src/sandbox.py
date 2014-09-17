import numpy as np
from scipy import optimize

import model

# define the number of cities
num_cities = 1

# define parameters
f, phi = 1.0, 5.0
theta = np.repeat(1.05, num_cities)

# define an initial guess
P0 = np.ones(num_cities)
Y0 = np.ones(num_cities)
W0 = np.ones(num_cities)
M0 = np.ones(num_cities)
initial_guess = np.hstack((Y0, W0, M0))


def equilibrium_system(X, f, phi, theta):
    P = np.append(np.ones(1.0), X[:num_cities])
    Y = X[num_cities:2 * num_cities]
    W = X[2 * num_cities:3 * num_cities]
    M = X[-num_cities:]

    out = model._numeric_system(P, Y, W, M, f, phi, theta).ravel()
    return out


def equilibrium_jacobian(X, f, phi, theta):
    P = np.append(np.ones(1.0), X[:num_cities])
    Y = X[num_cities:2 * num_cities]
    W = X[2 * num_cities:3 * num_cities]
    M = X[-num_cities:]

    out = model._numeric_jacobian(P, Y, W, M, f, phi, theta)
    return out


result = optimize.root(equilibrium_system,
                       x0=initial_guess,
                       args=(f, phi, theta),
                       jac=equilibrium_jacobian,
                       method='hybr',
                       tol=1e-12
                       )
