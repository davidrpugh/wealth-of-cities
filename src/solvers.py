import numpy as np
from scipy import optimize
import sympy as sym

# define parameters
f, beta, phi, tau, theta = sym.var('f, beta, phi, tau, theta')

# define variables
L, M, P, W, Y = sym.var('L, M, P, W, Y')


class InitialGuess(object):

    modules = modules = [{'ImmutableMatrix': np.array}, "numpy"]

    def __init__(self, solver):
        """
        Create an instance of the InitialGuess class.

        Parameters
        ----------
        solver : solvers.Solver
            An instance of the solvers.Solver class.

        """
        self.solver = solver

        # initialize cached values
        self.__numeric_gdp = None
        self.__numeric_num_firms = None
        self.__numeric_wage = None
        self.__symbolic_solution = None

    @property
    def _args(self):
        """
        Tuple of arguments to pass to functions used for numeric evaluation of
        symbolic solution.

        :getter: Return the current arguments
        :type: tuple

        """
        return (P, L, f, beta, phi, tau, theta)

    @property
    def _numeric_gdp(self):
        """
        Vectorized function for evaluating solution for nominal GDP.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_gdp is None:
            self.__numeric_gdp = sym.lambdify(self._args,
                                              self._symbolic_solution[Y],
                                              self.modules)
        return self.__numeric_gdp

    @property
    def _numeric_wage(self):
        """
        Vectorized function for evaluating solution for nominal wage.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_wage is None:
            self.__numeric_wage = sym.lambdify(self._args,
                                               self._symbolic_solution[W],
                                               self.modules)
        return self.__numeric_wage

    @property
    def _numeric_num_firms(self):
        """
        Vectorized function for evaluating solution for nominal number of firms.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_num_firms is None:
            self.__numeric_num_firms = sym.lambdify(self._args,
                                                    self._symbolic_solution[M],
                                                    self.modules)
        return self.__numeric_num_firms

    @property
    def _symbolic_equations(self):
        """
        List of symbolic equations defining a model with a single city.

        :getter: Return the current list of model equations.
        :type: list

        """
        # define model equations using derivation from paper
        eqn1 = ((1 / (theta - 1)) * ((theta / (theta - 1)) * (1 / phi))**-theta *
                (W / P)**-theta * (Y / P) - phi * f)
        eqn2 = (beta * L - (M / phi) * (theta / (theta - 1)) * ((theta / (theta - 1)) *
                (1 / phi))**-theta * (W / P)**-theta * (Y / P))
        eqn3 = Y - beta * L * W

        return [eqn1, eqn2, eqn3]

    @property
    def _symbolic_solution(self):
        """
        Dictionary of symbolic expressions for analytic solution to the single
        city model.

        :getter: Return the analytic solution to the model as a dictionary.
        :type: dict

        """
        if self.__symbolic_solution is None:
            self.__symbolic_solution, = sym.solve(self._symbolic_equations,
                                                  Y, W, M, dict=True)
        return self.__symbolic_solution

    @property
    def guess(self):
        """
        The initial guess for the model equilibrium.

        :getter: Return current initial guess.
        :type: numpy.ndarray

        """
        # extract model data
        num_cities = self.solver.model.N
        params = self.solver.model.params
        population = self.solver.model.population

        P0 = np.repeat(1.0, num_cities-1)
        Y0 = self._numeric_gdp(1.0, population[:num_cities], **params)
        W0 = self._numeric_wage(1.0, population[:num_cities], **params)
        M0 = self._numeric_num_firms(1.0, population[:num_cities], **params)

        return np.hstack((P0, Y0, W0, M0))


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

        # create an instance of the InitialGuess class
        self.initial_guess = InitialGuess(self)

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

    def solve(self, method, **kwargs):
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
                               x0=self.initial_guess.guess,
                               jac=self.jacobian,
                               method=method,
                               **kwargs
                               )
        return result


if __name__ == '__main__':

    from cruft import Model
    import master_data

    # grab data on physical distances
    physical_distances = np.load('../data/google/normed_vincenty_distance.npy')

    # compute the effective labor supply
    raw_data = master_data.panel.minor_xs(2010)
    clean_data = raw_data.sort('GDP_MP', ascending=False).drop([998, 48260])
    population = clean_data['POP_MI'].values

    # define some number of cities
    N = 10

    # define some parameters
    params = {'f': 1.0, 'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
              'theta': np.repeat(10.0, N)}

    model = Model(number_cities=N,
                  params=params,
                  physical_distances=physical_distances,
                  population=population)

    solver = Solver(model)

    result = solver.solve(method='hybr', tol=1e-6)

    print("Solution converged? {}".format(result.success))
    print("Equilibrium nominal price levels:\n{}".format(result.x[:N-1]))
    print("Equilibrium nominal GDP:\n{}".format(result.x[N-1:2 * N-1]))
    print("Equilibrium nominal wages:\n{}".format(result.x[2 * N-1:3 * N-1]))
    print("Equilibrium number of firms:\n{}".format(result.x[3 * N-1:]))
