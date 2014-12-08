"""
Test suite for the models.py module.

@author : David R. Pugh
@date : 2014-10-21

"""
import nose
import numpy as np

from models import Model
import master_data
import solvers

# grab data on physical distances
physical_distances = np.load('../data/google/normed_vincenty_distance.npy')

# compute the effective labor supply
raw_data = master_data.panel.minor_xs(2010)
clean_data = raw_data.sort('GDP_MP', ascending=False).drop([998, 48260])
population = clean_data['POP_MI'].values

# specify some parameter values
fixed_costs = np.logspace(-2, 2, 2)
scaling_factors = np.logspace(-2, 2, 2)
productivities = np.logspace(-2, 2, 2)
iceberg_costs = np.logspace(-2, 2, 2)
elasticities = np.logspace(3e-1, 2, 2)


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
                                      'elasticity_substitution': np.array([elasticity])
                                      }

                        tmp_model = Model(params=tmp_params,
                                          physical_distances=physical_distances,
                                          population=population)

                        tmp_solver = solvers.Solver(tmp_model)
                        tmp_initial = solvers.IslandsGuess(tmp_model)
                        tmp_initial.number_cities = 1

                        # conduct the test
                        expected_residual = np.zeros(3)
                        actual_residual = tmp_solver.system(tmp_initial.guess)

                        mesg = "Model params: {}".format(tmp_params)
                        np.testing.assert_almost_equal(expected_residual,
                                                       actual_residual,
                                                       err_msg=mesg)


def test_balance_trade():
    """Testing that exports balance imports."""
    # define some parameters
    params = {'f': 1.0, 'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
              'elasticity_substitution': np.repeat(10.0, 1)}
    model = Model(params=params,
                  physical_distances=physical_distances,
                  population=population)
    model.number_cities = 1

    for h in range(model.number_cities):
        actual_trade_balance = model.goods_market_clearing(h)
        expected_trade_balance = 0

        nose.tools.assert_equals(actual_trade_balance, expected_trade_balance)


def test_validate_num_cities():
    """Testing validation method for num_cities attribute."""
    # num_cities must be an integer...
    invalid_num_cities = 1.0

    valid_params = {'f': 1.0, 'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
                    'elasticity_substitution': np.repeat(10.0, 1)}

    with nose.tools.assert_raises(AttributeError):
        model = Model(params=valid_params,
                      physical_distances=physical_distances,
                      population=population)
        model.number_cities = invalid_num_cities

    # ...greater or equal to 1
    invalid_num_cities = 0

    with nose.tools.assert_raises(AttributeError):
        model = Model(params=valid_params,
                      physical_distances=physical_distances,
                      population=population)
        model.number_cities = invalid_num_cities


def test_validate_params():
    """Testing validation method for params attribute."""
    # params must be a dict
    invalid_params = (1.0, 1.31, 1.0 / 1.31, 0.05, np.repeat(10.0, 1))

    with nose.tools.assert_raises(AttributeError):
        Model(params=invalid_params,
              physical_distances=physical_distances,
              population=population)

    # ...and provide values for all required params
    invalid_params = {'beta': 1.31, 'phi': 1.0 / 1.31, 'tau': 0.05,
                      'elasticity_substitution': np.repeat(10.0, 1)}

    with nose.tools.assert_raises(AttributeError):
        Model(params=invalid_params,
              physical_distances=physical_distances,
              population=population)
