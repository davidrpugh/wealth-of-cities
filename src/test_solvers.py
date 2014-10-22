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


def test_initial_guesses():
    """Compare results using IslandsGuess and HotStartGuess initial guesses."""
    # define some number of cities
    N = 15

    # define some parameters
    params = {'f': 1.0, 'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
              'theta': np.repeat(10.0, N)}

    mod = models.Model(number_cities=N,
                       params=params,
                       physical_distances=physical_distances,
                       population=population)

    # create an initial guess
    islands = solvers.IslandsGuess(mod)
    islands.N = N

    solver = solvers.Solver(mod)
    islands_result = solver.solve(islands.guess, method='hybr', tol=1e-12,
                                  with_jacobian=True)

    # create an initial guess
    hot_start = solvers.HotStartGuess(mod)
    hot_start.N = N
    hot_start.solver_kwargs = {'method': 'hybr',
                               'tol': 1e-12,
                               'with_jacobian': True}

    np.testing.assert_almost_equal(islands_result.x, hot_start.guess)
