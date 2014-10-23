import nose

import numpy as np

import master_data
import models
import solvers

# grab data on physical distances
physical_distances = np.load('../data/google/normed_vincenty_distance.npy')

# compute the effective labor supply
raw_data = master_data.panel.minor_xs(2010)
clean_data = raw_data.sort('GDP_MP', ascending=False).drop([998, 48260])
population = clean_data['POP_MI'].values

# define some parameters
N = 380
params = {'f': 1.0, 'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
          'theta': np.repeat(10.0, N)}

model = models.Model(params=params,
                     physical_distances=physical_distances,
                     population=population)


def test_initial_guesses():
    """Compare results using IslandsGuess vs HotStartGuess."""
    # define some number of cities
    N = np.random.randint(1, 25)

    # create an initial guess
    islands = solvers.IslandsGuess(model)
    islands.number_cities = N

    solver = solvers.Solver(model)
    islands_result = solver.solve(islands.guess, method='hybr', tol=1e-12,
                                  with_jacobian=True)

    # create an initial guess
    hot_start = solvers.HotStartGuess(model)
    hot_start.number_cities = N
    hot_start.solver_kwargs = {'method': 'hybr',
                               'tol': 1e-12,
                               'with_jacobian': True}

    np.testing.assert_almost_equal(islands_result.x, hot_start.guess,
                                   err_msg="Number of cities: {}".format(N))


def test_jacobians():
    """Testing results using finite difference and symbolic jacobians."""
    # define some number of cities
    N = np.random.randint(1, 25)

    # check that solutions are the same for approx and exact jacobian
    solver = solvers.Solver(model)
    initial_guess = solvers.IslandsGuess(model)
    initial_guess.number_cities = N
    approx_jac = solver.solve(initial_guess.guess, method='hybr', tol=1e-12,
                              with_jacobian=False, options={'eps': 1e-15})

    exact_jac = solver.solve(initial_guess.guess, method='hybr', tol=1e-12,
                             with_jacobian=True)

    np.testing.assert_almost_equal(approx_jac.x, exact_jac.x,
                                   err_msg="Number of cities: {}".format(N))


def test_not_implemented_methods():
    """Testing unimplemented methods of InitialGuess class."""
    with nose.tools.assert_raises(NotImplementedError):
        initial = solvers.InitialGuess(model)
        initial.guess
