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


class Estimator(object):

    def __init__(self, solver, data):
        """
        Create and instance of the Estimator class.

        Parameters
        ----------
        solver : solvers.Solver
            An instance of the solvers.Solver class.
        data : pandas.DataFrame
            An instance of the pandas.DataFrame class.

        """
        self.data = data
        self.solver = solver

    def objective(self, params):
        raise NotImplementedError

    def estimate(self, method, **kwargs):
        """
        Estimate the model given some data.

        Parameters
        ----------
        method : str
            Valid method used to find the minimize the objective function. See
            scipy.optimize.minimize for a complete list of valid methods.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The solution represented as a OptimizeResult object. Important
            attributes are: x the solution array, success a Boolean flag
            indicating if the algorithm exited successfully and message which
            describes the cause of the termination.

        """
        result = optimize.minimize(self.objective,
                                   x0=self.initial_guess,
                                   method=method,
                                   **kwargs)
        return result


class NLLS(Estimator):

    def objective(self, params, **kwargs):
        """Quick pass at a weighted NLLS objective function."""
        actual_GDP = self.data['GDP_MP']
        predicted_GDP = self.solver.solve(**kwargs)
        residual = actual_GDP - predicted_GDP
        obj = np.sum(residual**2)
        return obj
