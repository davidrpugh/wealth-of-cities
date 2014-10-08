"""
Code for generating the symbolic equations which define the equilibrium of our
model.

@author : David R. Pugh
@date : 2014-10-06

"""
import numpy as np
import sympy as sym

import master_data

# define parameters
f, beta, phi, tau = sym.var('f, beta, phi, tau')
elasticity_substitution = sym.DeferredVector('theta')

# compute the economic distance
physical_distance = np.load('../data/google/normed_vincenty_distance.npy')
# economic_distance = sym.MatrixSymbol('delta', num_cities, num_cities)

# compute the effective labor supply
raw_data = master_data.panel.minor_xs(2010)
clean_data = raw_data.sort('GDP_MP', ascending=False).drop([998, 48260])
total_population = clean_data['POP_MI'].values
effective_labor_supply = sym.Matrix([beta * total_population])
# total_labor_supply = sym.DeferredVector('S')

# define variables
nominal_gdp = sym.DeferredVector('Y')
nominal_price_level = sym.DeferredVector('P')
nominal_wage = sym.DeferredVector('W')
num_firms = sym.DeferredVector('M')


class Model(object):

    def __init__(self, number_cities, params, physical_distances):
        """
        Create an instance of the Model class.

        Parameters
        ----------
        number_cities : int
            Number of cities in the economy.
        params : dict
            Dictionary of model parameters.
        physical_distances : numpy.ndarray (shape=(N,N))
            Square array of pairwise measures pf physical distance between
            cities.

        """
        self.N = number_cities
        self.params = params
        self.physical_distances

    @property
    def economic_distances(self):
        """
        Square matrix of pairwise measures of economic distance between cities.

        :getter: Return the matrix of economic distances.
        :type: sympy.Basic

        """
        return np.exp(self.physical_distances)**tau

    @property
    def N(self):
        """
        Number of cities in the economy.

        :getter: Return the current number of cities.
        :setter: Set a new number of cities.
        :type: int

        """
        return self._N

    @N.setter
    def N(self, value):
        """Set a new number of cities."""
        self._N = self._validate_number_cities(value)

    @property
    def physical_distances(self):
        """
        Square array of pairwise measures pf physical distance between cities.

        :getter: Return the current array of physical distances.
        :setter: Set a new array of physical distances.
        :type: numpy.ndarray

        """
        return self._physical_distances[:self.N, :self.N]

    @physical_distances.setter
    def physical_distances(self, array):
        """Set a new array of physical distances."""
        self._physical_distance = array

    @property
    def params(self):
        """
        Dictionary of model parameters.

        :getter: Return the current parameter dictionary.
        :setter: Set a new parameter dictionary.
        :type: dict

        """
        return self._params

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        self._params = self._validate_params(value)

    @classmethod
    def _validate_number_cities(cls, value):
        if not isinstance(params, int):
            mesg = "Model.N attribute must have type int and not {}"
            raise AttributeError(mesg.format(value.__class__))
        elif value < 1:
            mesg = "Model.N attribute must be greater than or equal to 1."
            raise AttributeError(mesg)
        else:
            return value

    @classmethod
    def _validate_params(cls, params):
        required_params = ['f', 'beta', 'phi', 'theta', 'tau']
        if not isinstance(params, dict):
            mesg = "Model.params attribute must have type dict and not {}"
            raise AttributeError(mesg.format(params.__class__))
        elif not set(required_params) < set(params.keys()):
            mesg = "Parameter dictionary must specify values for each of {}"
            raise AttributeError(mesg.format(required_params))
        else:
            return params

    @classmethod
    def goods_market_clearing(cls, h):
        """Exports must balance imports for city h."""
        return cls.total_exports(h) - cls.total_imports(h)

    @classmethod
    def labor_market_clearing(cls, h):
        """Labor market clearing condition for city h."""
        return effective_labor_supply[h] - cls.total_labor_demand(h)

    def labor_productivity(self, h, j):
        """Productivity of labor in city h when producing good j."""
        return phi / self.economic_distances[h, j]

    @classmethod
    def marginal_costs(cls, h, j):
        """Marginal costs of production of good j in city h."""
        return nominal_wage[h] / cls.labor_productivity(h, j)

    @staticmethod
    def mark_up(j):
        """Markup over marginal costs of production for good j."""
        return (elasticity_substitution[j] / (elasticity_substitution[j] - 1))

    @classmethod
    def optimal_price(cls, h, j):
        """Optimal price of good j sold in city h."""
        return cls.mark_up(j) * cls.marginal_costs(h, j)

    @classmethod
    def quantity_demand(cls, price, j):
        """Quantity demanded of a good in city j depends negatively on its price."""
        return cls.relative_price(price, j)**(-elasticity_substitution[j]) * cls.real_gdp(j)

    @staticmethod
    def real_gdp(i):
        """Real gross domestic product of city i."""
        return nominal_gdp[i] / nominal_price_level[i]

    @staticmethod
    def relative_price(price, j):
        """Relative price of a good in city j."""
        return price / nominal_price_level[j]

    @staticmethod
    def resource_constraint(h):
        """Nominal GDP in city h must equal nominal income in city h."""
        return nominal_gdp[h] - effective_labor_supply[h] * nominal_wage[h]

    @staticmethod
    def revenue(price, quantity):
        """Revenue from producing a certain quantity at a given price."""
        return price * quantity

    @classmethod
    def total_cost(cls, h):
        """Total cost of production for a firm in city h."""
        return cls.total_variable_cost(h) + cls.total_fixed_cost(h)

    def total_exports(self, h):
        """Total exports of various goods from city h."""
        individual_exports = []
        for j in range(self.N):
            p_star = self.optimal_price(h, j)
            q_star = self.quantity_demand(p_star, j)
            total_revenue_h = num_firms[h] * self.revenue(p_star, q_star)
            individual_exports.append(total_revenue_h)

        return sum(individual_exports)

    @staticmethod
    def total_fixed_cost(h):
        """Total fixed cost of production for a firm in city h."""
        return f * nominal_wage[h]

    @staticmethod
    def total_fixed_labor_demand(h):
        """Total fixed labor demand for firms in city h."""
        return num_firms[h] * f

    def total_imports(self, h):
        """Total imports of various goods into city h."""
        individual_imports = []
        for j in range(self.N):
            p_star = self.optimal_price(j, h)
            q_star = self.quantity_demand(p_star, h)
            total_revenue_j = num_firms[j] * self.revenue(p_star, q_star)
            individual_imports.append(total_revenue_j)

        return sum(individual_imports)

    @classmethod
    def total_labor_demand(cls, h):
        """Total demand for labor for firms in city h."""
        return cls.total_variable_labor_demand(h) + cls.total_fixed_labor_demand(h)

    @classmethod
    def total_profits(cls, h):
        """Total profits for a firm in city h."""
        return cls.total_revenue(h) - cls.total_cost(h)

    def total_revenue(self, h):
        """Total revenue for a firm producing in city h."""
        individual_revenues = []
        for j in range(self.N):
            p_star = self.optimal_price(h, j)
            q_star = self.quantity_demand(p_star, j)
            individual_revenues.append(self.revenue(p_star, q_star))

        return sum(individual_revenues)

    def total_variable_cost(self, h):
        """Total variable costs of production for a firm in city h."""
        individual_variable_costs = []
        for j in range(self.N):
            p_star = self.optimal_price(h, j)
            q_star = self.quantity_demand(p_star, j)
            individual_variable_costs.append(self.variable_cost(q_star, h, j))

        return sum(individual_variable_costs)

    def total_variable_labor_demand(self, h):
        """Total variable labor demand for firms in city h."""
        individual_labor_demands = []
        for j in range(self.N):
            p_star = self.optimal_price(h, j)
            q_star = self.quantity_demand(p_star, j)
            individual_labor_demands.append(self.variable_labor_demand(q_star, h, j))

        return num_firms[h] * sum(individual_labor_demands)

    @classmethod
    def variable_cost(cls, quantity, h, j):
        """
        Variable cost of a firm in city h to produce a given quantity of good
        for sale in city j.

        """
        return cls.variable_labor_demand(quantity, h, j) * nominal_wage[h]

    @classmethod
    def variable_labor_demand(cls, quantity, h, j):
        """
        Variable labor demand by firm in city h to produce a given quantity of
        good for sale in city j.

        """
        return quantity / cls.labor_productivity(h, j)

### construct the system of non-linear equilibrium conditions and its jacobian

# normalize P[0] = 1.0 (so only P[1]...P[num_cities-1] are unknowns)
endog_vars = ([nominal_price_level[h] for h in range(1, num_cities)] +
              [nominal_gdp[h] for h in range(num_cities)] +
              [nominal_wage[h] for h in range(num_cities)] +
              [num_firms[h] for h in range(num_cities)])

# drop one equation as a result of normalization
equations = ([goods_market_clearing(h) for h in range(1, num_cities)] +
             [total_profits(h) for h in range(num_cities)] +
             [labor_market_clearing(h) for h in range(num_cities)] +
             [resource_constraint(h) for h in range(num_cities)])

symbolic_system = sym.Matrix(equations)
#symbolic_jacobian = symbolic_system.jacobian(endog_vars)

# wrap the symbolic equilibrium system and jacobian
vector_vars = (nominal_price_level, nominal_gdp, nominal_wage, num_firms)
params = (f, beta, phi, tau, elasticity_substitution)
args = vector_vars + params

numeric_system = sym.lambdify(args, symbolic_system,
                              modules=[{'ImmutableMatrix': np.array}, "numpy"])

#numeric_jacobian = sym.lambdify(args, symbolic_jacobian,
#                                modules=[{'ImmutableMatrix': np.array}, "numpy"])
