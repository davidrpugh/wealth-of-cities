"""
Main classes representing the model.

@author : David R. Pugh
@date : 2014-10-21

"""
import numpy as np
import sympy as sym

# define parameters
f, beta, phi, tau = sym.var('f, beta, phi, tau')
elasticity_substitution = sym.DeferredVector('theta')

# define variables
nominal_gdp = sym.DeferredVector('Y')
nominal_price_level = sym.DeferredVector('P')
nominal_wage = sym.DeferredVector('W')
num_firms = sym.DeferredVector('M')


class Model(object):

    # initialize the cached values
    __symbolic_equations = None
    __symbolic_jacobian = None
    __symbolic_system = None
    __symbolic_variables = None

    def __init__(self, number_cities, params, physical_distances, population):
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
        population : numpy.ndarray (shape=(N,))
            Array of total population for each city.

        """
        self.N = number_cities
        self.params = params
        self.physical_distances = physical_distances
        self.population = population

    @property
    def _symbolic_args(self):
        """
        Arguments to pass to functions used for numeric evaluation of model.

        :getter: Return the current arguments
        :type: tuple

        """
        variables = (nominal_price_level, nominal_gdp, nominal_wage, num_firms)
        params = (f, beta, phi, tau, elasticity_substitution)
        return variables + params

    @property
    def _symbolic_equations(self):
        """
        List of symbolic equations defining the model.

        :getter: Return the current list of model equations.
        :type: list

        """
        if self.__symbolic_equations is None:
            # drop one equation as a result of normalization
            eqns = ([self.goods_market_clearing(h) for h in range(1, self.N)] +
                    [self.total_profits(h) for h in range(self.N)] +
                    [self.labor_market_clearing(h) for h in range(self.N)] +
                    [self.resource_constraint(h) for h in range(self.N)])
            self.__symbolic_equations = eqns
        return self.__symbolic_equations

    @property
    def _symbolic_jacobian(self):
        """
        Symbolic representation of Jacobian matrix of partial derivatives.

        :getter: Return the current Jacobian matrix.
        :type: sympy.Matrix

        """
        if self.__symbolic_jacobian is None:
            jac = self._symbolic_system.jacobian(self._symbolic_variables)
            self.__symbolic_jacobian = jac
        return self.__symbolic_jacobian

    @property
    def _symbolic_system(self):
        """
        Matrix representation of symbolic model equations.

        :getter: Return the current model equations as a symbolic Matrix.
        :type: sympy.Matrix

        """
        return sym.Matrix(self._symbolic_equations)

    @property
    def _symbolic_variables(self):
        """
        List of symbolic endogenous variables.

        :getter: Return the current list of endogenous variables.
        :type: list

        """
        if self.__symbolic_variables is None:
            # normalize P[0] = 1.0 (only P[1]...P[num_cities-1] are unknowns)
            variables = ([nominal_price_level[h] for h in range(1, self.N)] +
                         [nominal_gdp[h] for h in range(self.N)] +
                         [nominal_wage[h] for h in range(self.N)] +
                         [num_firms[h] for h in range(self.N)])
            self.__symbolic_variables = variables
        return self.__symbolic_variables

    @property
    def economic_distances(self):
        """
        Square matrix of pairwise measures of economic distance between cities.

        :getter: Return the matrix of economic distances.
        :type: sympy.Basic

        """
        return np.exp(self.physical_distances)**tau

    @property
    def effective_labor_supply(self):
        """
        Effective labor supply is a constant multple of total population.

        :getter: Return the current effective labor supply.
        :type: sympy.Matrix

        """
        return sym.Matrix([beta * self.population])

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

        # don't forget to clear cache!
        self._clear_cache()

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
        self._physical_distances = array

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

    def _clear_cache(self):
        """Clear all cached values."""
        self.__symbolic_equations = None
        self.__symbolic_jacobian = None
        self.__symbolic_system = None
        self.__symbolic_variables = None

    @classmethod
    def _validate_number_cities(cls, value):
        """Validate number of cities, N, attribute."""
        if not isinstance(value, int):
            mesg = "Model.N attribute must have type int and not {}"
            raise AttributeError(mesg.format(value.__class__))
        elif value < 1:
            mesg = "Model.N attribute must be greater than or equal to 1."
            raise AttributeError(mesg)
        else:
            return value

    @classmethod
    def _validate_params(cls, params):
        """Validate params attribute."""
        required_params = ['f', 'beta', 'phi', 'tau']
        if not isinstance(params, dict):
            mesg = "Model.params attribute must have type dict and not {}"
            raise AttributeError(mesg.format(params.__class__))
        elif not set(required_params) <= set(params.keys()):
            mesg = "Parameter dictionary must specify values for each of {}"
            raise AttributeError(mesg.format(required_params))
        else:
            return params

    def goods_market_clearing(self, h):
        """Exports must balance imports for city h."""
        return self.total_exports(h) - self.total_imports(h)

    def labor_market_clearing(self, h):
        """Labor market clearing condition for city h."""
        return self.effective_labor_supply[h] - self.total_labor_demand(h)

    def labor_productivity(self, h, j):
        """Productivity of labor in city h when producing good j."""
        return phi / self.economic_distances[h, j]

    def marginal_costs(self, h, j):
        """Marginal costs of production of good j in city h."""
        return nominal_wage[h] / self.labor_productivity(h, j)

    @staticmethod
    def mark_up(j):
        """Markup over marginal costs of production for good j."""
        return (elasticity_substitution[j] / (elasticity_substitution[j] - 1))

    def optimal_price(self, h, j):
        """Optimal price of good j sold in city h."""
        return self.mark_up(j) * self.marginal_costs(h, j)

    @classmethod
    def quantity_demand(cls, price, j):
        """Quantity demand of a good is negative function of price."""
        return cls.relative_price(price, j)**(-elasticity_substitution[j]) * cls.real_gdp(j)

    @staticmethod
    def real_gdp(i):
        """Real gross domestic product of city i."""
        return nominal_gdp[i] / nominal_price_level[i]

    @staticmethod
    def relative_price(price, j):
        """Relative price of a good in city j."""
        return price / nominal_price_level[j]

    def resource_constraint(self, h):
        """Nominal GDP in city h must equal nominal income in city h."""
        constraint = (nominal_gdp[h] -
                      self.effective_labor_supply[h] * nominal_wage[h])
        return constraint

    @staticmethod
    def revenue(price, quantity):
        """Revenue from producing a certain quantity at a given price."""
        return price * quantity

    def total_cost(self, h):
        """Total cost of production for a firm in city h."""
        return self.total_variable_cost(h) + self.total_fixed_cost(h)

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

    def total_labor_demand(self, h):
        """Total demand for labor for firms in city h."""
        total_demand = (self.total_variable_labor_demand(h) +
                        self.total_fixed_labor_demand(h))
        return total_demand

    def total_profits(self, h):
        """Total profits for a firm in city h."""
        return self.total_revenue(h) - self.total_cost(h)

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
            variable_demand_h = self.variable_labor_demand(q_star, h, j)
            individual_labor_demands.append(variable_demand_h)

        return num_firms[h] * sum(individual_labor_demands)

    def variable_cost(self, quantity, h, j):
        """
        Variable cost of a firm in city h to produce a given quantity of good
        for sale in city j.

        """
        return self.variable_labor_demand(quantity, h, j) * nominal_wage[h]

    def variable_labor_demand(self, quantity, h, j):
        """
        Variable labor demand by firm in city h to produce a given quantity of
        good for sale in city j.

        """
        return quantity / self.labor_productivity(h, j)


class SingleCityModel(Model):

    # initialize cached values
    __numeric_gdp = None
    __numeric_num_firms = None
    __numeric_wage = None
    __symbolic_solution = None

    _modules = [{'ImmutableMatrix': np.array}, "numpy"]

    def __init__(self, params, physical_distances, population):
        """
        Create an instance of the SingleCityModel class.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.
        physical_distances : numpy.ndarray (shape=(N,N))
            Square array of pairwise measures pf physical distance between
            cities.
        population : numpy.ndarray (shape=(N,))
            Array of total population for each city.

        """
        super(SingleCityModel, self).__init__(1, params, physical_distances, population)

    @property
    def solution(self):
        """
        Equilbrium values of nominal GDP, nominal wages, and number of firms.

        :getter: Return current solution.
        :type: numpy.ndarray

        """
        P0 = np.ones(1.0)
        Y0 = self._numeric_gdp(P0, **self.params)
        W0 = self._numeric_wage(P0, **self.params)
        M0 = self._numeric_num_firms(P0, **self.params)

        return np.hstack((P0, Y0, W0, M0))

    @property
    def _numeric_gdp(self):
        """
        Vectorized function for evaluating solution for nominal GDP.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_gdp is None:
            Y = nominal_gdp[0]
            self.__numeric_gdp = sym.lambdify(self._symbolic_args,
                                              self._symbolic_solution[Y],
                                              self._modules)
        return self.__numeric_gdp

    @property
    def _numeric_wage(self):
        """
        Vectorized function for evaluating solution for nominal wage.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_wage is None:
            W = nominal_wage[0]
            self.__numeric_wage = sym.lambdify(self._symbolic_args,
                                               self._symbolic_solution[W],
                                               self._modules)
        return self.__numeric_wage

    @property
    def _numeric_num_firms(self):
        """
        Vectorized function for evaluating solution for number of firms.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_num_firms is None:
            M = num_firms[0]
            self.__numeric_num_firms = sym.lambdify(self._symbolic_args,
                                                    self._symbolic_solution[M],
                                                    self._modules)
        return self.__numeric_num_firms

    @property
    def _symbolic_args(self):
        """
        Arguments to pass to functions used for numeric evaluation of model.

        :getter: Return the current arguments
        :type: tuple

        """
        variables = (nominal_price_level,)
        params = (f, beta, phi, tau, elasticity_substitution)
        return variables + params

    @property
    def _symbolic_solution(self):
        """
        Dictionary of symbolic expressions for analytic solution to the model.

        :getter: Return the analytic solution to the model as a dictionary.
        :type: dict

        """
        if self.__symbolic_solution is None:
            self.__symbolic_solution, = sym.solve(self._symbolic_equations,
                                                  self._symbolic_variables,
                                                  dict=True)
        return self.__symbolic_solution
