import numpy as np
from scipy import optimize

import model

# define the number of cities
num_cities = model.num_cities

# define parameters
f, beta, phi, tau = 1.0, 1.0, 1.0, 1.0
theta = np.repeat(1.5, num_cities)

# define an initial guess
P0 = np.ones(num_cities-1)
Y0 = np.ones(num_cities)
W0 = Y0 / (beta * model.total_population[:num_cities])
M0 = np.repeat(0.75, num_cities)
initial_guess = np.hstack((P0, Y0, W0, M0))


def equilibrium_system(X, f, beta, phi, tau, theta):
    """
    System of non-linear equations defining the model equilibrium.

    Parameters
    ----------
    X : ndarray
    f : float
        Firms fixed cost of production.
    beta : float
        Scaling factor for aggregate labor supply in a city.
    phi : float
    tau : float
        Iceberg trade costs.
    theta : ndarray
        Elasticity of substitutio between varieties of the consumption good.

    Returns
    -------
    residual : ndarray
        Equilibrium values of the endogenous variables will make the residual
        zero everywhere.

    """
    P = np.append(np.ones(1.0), X[:num_cities-1])
    Y = X[num_cities-1:2 * num_cities-1]
    W = X[2 * num_cities-1:3 * num_cities-1]
    M = X[3 * num_cities-1:]
    residual = model.numeric_system(P, Y, W, M, f, beta, phi, tau, theta).ravel()
    return residual


def equilibrium_jacobian(X, f, beta, phi, tau, theta):
    """
    Jacobian matrix of partial derivatives for the system of non-linear
    equations defining the model equilibrium.

    Parameters
    ----------
    X : ndarray
    f : float
        Firms fixed cost of production.
    beta : float
        Scaling factor for aggregate labor supply in a city.
    phi : float
    tau : float
        Iceberg trade costs.
    theta : ndarray
        Elasticity of substitutio between varieties of the consumption good.

    Returns
    -------
    jac : ndarray
        Jacobian matrix of partial derivatives.

    """
    P = np.append(np.ones(1.0), X[:num_cities-1])
    Y = X[num_cities-1:2 * num_cities-1]
    W = X[2 * num_cities-1:3 * num_cities-1]
    M = X[3 * num_cities-1:]

    jac = model.numeric_jacobian(P, Y, W, M, f, beta, phi, tau, theta)
    return jac


# solve for the model equilibrium
result = optimize.root(equilibrium_system,
                       x0=initial_guess,
                       args=(f, beta, phi, tau, theta),
                       jac=equilibrium_jacobian,
                       method='krylov',
                       tol=1e-6,
                       )
