"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-10-08

"""
import nose
import numpy as np

from model import Model
import master_data
import solvers

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

solver = solvers.Solver(model)

# specify some parameter values
fixed_costs = np.logspace(-2, 2, 7)
scaling_factors = np.logspace(-2, 2, 7)
productivities = np.logspace(-2, 2, 7)
iceberg_costs = np.logspace(-2, 2, 7)
elasticities = np.logspace(3e-1, 2, 7)


def test_residual():
    """Testing model residual."""
    for fixed_cost in fixed_costs:
        for scaling_factor in scaling_factors:
            for productivity in productivities:
                for iceberg_cost in iceberg_costs:
                    for elasticity in elasticities:

                        tmp_params = {'f': fixed_cost,
                                      'beta': scaling_factor,
                                      'phi': productivity,
                                      'tau': iceberg_cost,
                                      'theta': np.array([elasticity])
                                      }
                        model.params = tmp_params

                        expected_residual = np.zeros(3)
                        X0 = solver.initial_guess.guess
                        actual_residual = solver.system(X0)

                        np.testing.assert_almost_equal(expected_residual,
                                                       actual_residual,
                                                       verbose=True)


def test_balance_trade():
    """Testing that exports balance imports."""
    for h in range(model.N):
        actual_trade_balance = model.goods_market_clearing(h)
        expected_trade_balance = 0

        nose.tools.assert_equals(actual_trade_balance, expected_trade_balance)


def test_validate_num_cities():
    """Testing validation method for num_cities attribute."""
    # num_cities must be an integer...
    invalid_num_cities = 1.0

    with nose.tools.assert_raises(AttributeError):
        Model(number_cities=invalid_num_cities,
              params=params,
              physical_distances=physical_distances,
              population=population)

    # ...greater or equal to 1
    invalid_num_cities = 0

    with nose.tools.assert_raises(AttributeError):
        Model(number_cities=invalid_num_cities,
              params=params,
              physical_distances=physical_distances,
              population=population)


def test_validate_params():
    """Testing validation method for params attribute."""
    # params must be a dict
    invalid_params = (1.0, 1.31, 1.0 / 1.31, 0.05, np.repeat(10.0, N))

    with nose.tools.assert_raises(AttributeError):
        Model(number_cities=10,
              params=invalid_params,
              physical_distances=physical_distances,
              population=population)

    # ...and provide values for all required params
    invalid_params = {'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
                      'theta': np.repeat(10.0, N)}

    with nose.tools.assert_raises(AttributeError):
        Model(number_cities=15,
              params=invalid_params,
              physical_distances=physical_distances,
              population=population)
