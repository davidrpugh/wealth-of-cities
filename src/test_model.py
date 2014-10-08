"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-10-02

"""
import nose
import numpy as np

from initial_guess import get_initial_guess
import model
import physical_distance
import sandbox

# specify some parameter values
fixed_costs = np.logspace(-2, 2, 5)
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

                        expected_residual = np.zeros(3)
                        X0 = get_initial_guess(1, **tmp_params)
                        actual_residual = sandbox.equilibrium_system(X0, **tmp_params)

                        np.testing.assert_almost_equal(expected_residual,
                                                       actual_residual,
                                                       verbose=True)


def test_balance_trade():
    """Testing that exports balance imports."""
    for h in range(model.num_cities):
        actual_trade_balance = model.goods_market_clearing(h)
        expected_trade_balance = 0

        nose.tools.assert_equals(actual_trade_balance, expected_trade_balance)


def test_data_alignment():
    """Testing alignment of population and physical distance data."""
    condition = model.clean_data.index == physical_distance.geo_coords.index
    nose.tools.assert_true(condition.all())
