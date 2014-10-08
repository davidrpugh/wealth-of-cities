import numpy as np
from scipy import optimize


class Solver(object):

    def __init__(self, model):
        """
        Create and instance of the Solver class.

        Parameters
        ----------
        model : model.model
            Instance of the model.Model class that you wish to solve.

        """
        self.model = model

    def system(self, X):
        """
        System of non-linear equations defining the model equilibrium.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        residual : numpy.ndarray
            Value of the model residual given current values of endogenous
            variables and parameters.

        """
        P = np.append(np.ones(1.0), X[:self.model.N-1])
        Y = X[self.model.N-1:2 * self.model.N-1]
        W = X[2 * self.model.N-1:3 * self.model.N-1]
        M = X[3 * self.model.N-1:]
        residual = self.model._numeric_system(P, Y, W, M, **self.model.params)
        return residual.ravel()

    def jacobian(self, X):
        """
        Jacobian matrix of partial derivatives for the system of non-linear
        equations defining the model equilibrium.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        jac : numpy.ndarray
            Jacobian matrix of partial derivatives.

        """
        P = np.append(np.ones(1.0), X[:self.model.N-1])
        Y = X[self.model.N-1:2 * self.model.N-1]
        W = X[2 * self.model.N-1:3 * self.model.N-1]
        M = X[3 * self.model.N-1:]

        jac = self.model._numeric_jacobian(P, Y, W, M, **self.model.params)

        return jac

    def solve(self, initial_guess, method, **kwargs):
        """
        Solve the system of non-linear equations describing the equilibrium.

        Parameters
        ----------
        initial_guess : numpy.ndarray
        method : str

        Returns
        -------
        result :

        """
        # solve for the model equilibrium
        result = optimize.root(self.system,
                               x0=initial_guess,
                               jac=self.jacobian,
                               method=method,
                               **kwargs
                               )
        return result


if __name__ == '__main__':

    from cruft import Model
    import master_data
    from test_model import get_initial_guess

    # grab data on physical distances
    physical_distances = np.load('../data/google/normed_vincenty_distance.npy')

    # compute the effective labor supply
    raw_data = master_data.panel.minor_xs(2010)
    clean_data = raw_data.sort('GDP_MP', ascending=False).drop([998, 48260])
    population = clean_data['POP_MI'].values

    # define some number of cities
    N = 1

    # define some parameters
    params = {'f': 1.0, 'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
              'theta': np.repeat(10.0, N)}

    model = Model(number_cities=N,
                  params=params,
                  physical_distances=physical_distances,
                  population=population)

    solver = Solver(model)

    initial_guess = get_initial_guess(N, **model.params)

    result = solver.solve(initial_guess, method='hybr')

    print("Solution converged? {}".format(result.success))
    print("Equilibrium nominal price levels:\n{}".format(result.x[:N-1]))
    print("Equilibrium nominal GDP:\n{}".format(result.x[N-1:2 * N-1]))
    print("Equilibrium nominal wages:\n{}".format(result.x[2 * N-1:3 * N-1]))
    print("Equilibrium number of firms:\n{}".format(result.x[3 * N-1:]))
