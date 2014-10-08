import numpy as np
from scipy import optimize

import master_data
import model
from test_model import get_initial_guess

# load the csv file containing the geocoordinates
data = master_data.major_xs(2010)
clean_data = data.drop([998, 48260])  # drop MSAs with bad geo coords

# use same number of cities
num_cities = model.num_cities


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
        Elasticity of substitution between varieties of the consumption good.

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


def _solve_model(f, beta, phi, tau, theta):
    """Solve for the equilibrium of the model given parameters."""

    initial_guess = get_initial_guess(num_cities, f, beta, phi, tau, theta)

    result = optimize.root(equilibrium_system,
                           x0=initial_guess,
                           args=(f, beta, phi, tau, theta),
                           jac=equilibrium_jacobian,
                           method='hybr',
                           tol=1e-6,
                           )

    if result.success:
        return result.x
    else:
        raise ValueError


def nlls_objective(params):
    """Objective function for non-linear least squares parameter estimation."""
    actual_GDP = clean_data['GDP_MP']
    predicted_GDP = _solve_model(1.0, **params)
    residual = actual_GDP - predicted_GDP
    obj = np.sum(residual**2)
    return obj
