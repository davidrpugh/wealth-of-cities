"""
Code for generating the symbolic equations which define the equilibrium of our
model.

@author : David R. Pugh
@date : 2014-09-25

"""
import numpy as np
import sympy as sym

import master_data
from physical_distance import normed_vincenty_distance

# define the number of cities
num_cities = 15

# define parameters
f, beta, phi, tau = sym.var('f, beta, phi, tau')
elasticity_substitution = sym.DeferredVector('theta')

# compute the economic distance
physical_distance = normed_vincenty_distance
economic_distance = np.exp(physical_distance[:num_cities, :num_cities])**tau
# economic_distance = sym.MatrixSymbol('delta', num_cities, num_cities)

# compute the effective labor supply
total_population = master_data.panel['POP_MI'][2010].values
effective_labor_supply = sym.Matrix([beta * total_population[:num_cities]])
# total_labor_supply = sym.DeferredVector('S')


# define variables
nominal_gdp = sym.DeferredVector('Y')
nominal_price_level = sym.DeferredVector('P')
nominal_wage = sym.DeferredVector('W')
num_firms = sym.DeferredVector('M')


def goods_market_clearing(h):
    """Exports must balance imports for city h."""
    return total_exports(h) - total_imports(h)


def labor_market_clearing(h):
    """Labor market clearing condition for city h."""
    return effective_labor_supply[h] - total_labor_demand(h)


def labor_productivity(h, j):
    """Productivity of labor in city h when producing good j."""
    return phi / economic_distance[h, j]


def marginal_costs(h, j):
    """Marginal costs of production of good j in city h."""
    return nominal_wage[h] / labor_productivity(h, j)


def mark_up(j):
    """Markup over marginal costs of production for good j."""
    return (elasticity_substitution[j] / (elasticity_substitution[j] - 1))


def optimal_price(h, j):
    """Optimal price of good j sold in city h."""
    return mark_up(j) * marginal_costs(h, j)


def quantity_demand(price, j):
    """Quantity demanded of a good in city j depends negatively on its price."""
    return relative_price(price, j)**(-elasticity_substitution[j]) * real_gdp(j)


def real_gdp(i):
    """Real gross domestic product of city i."""
    return nominal_gdp[i] / nominal_price_level[i]


def relative_price(price, j):
    """Relative price of a good in city j."""
    return price / nominal_price_level[j]


def resource_constraint(h):
    """Nominal GDP in city h must equal nominal income in city h."""
    return nominal_gdp[h] - effective_labor_supply[h] * nominal_wage[h]


def revenue(price, quantity):
    """Revenue from producing a certain quantity at a given price."""
    return price * quantity


def total_cost(h):
    """Total cost of production for a firm in city h."""
    return total_variable_cost(h) + total_fixed_cost(h)


def total_exports(h):
    """Total exports of various goods from city h."""
    individual_exports = []
    for j in range(num_cities):
        p_star = optimal_price(h, j)
        q_star = quantity_demand(p_star, j)
        individual_exports.append(num_firms[h] * revenue(p_star, q_star))

    return sum(individual_exports)


def total_fixed_cost(h):
    """Total fixed cost of production for a firm in city h."""
    return f * nominal_wage[h]


def total_fixed_labor_demand(h):
    """Total fixed labor demand for firms in city h."""
    return num_firms[h] * f


def total_imports(h):
    """Total imports of various goods into city h."""
    individual_imports = []
    for j in range(num_cities):
        p_star = optimal_price(j, h)
        q_star = quantity_demand(p_star, h)
        individual_imports.append(num_firms[j] * revenue(p_star, q_star))

    return sum(individual_imports)


def total_labor_demand(h):
    """Total demand for labor for firms in city h."""
    return total_variable_labor_demand(h) + total_fixed_labor_demand(h)


def total_profits(h):
    """Total profits for a firm in city h."""
    return total_revenue(h) - total_cost(h)


def total_revenue(h):
    """Total revenue for a firm producing in city h."""
    individual_revenues = []
    for j in range(num_cities):
        p_star = optimal_price(h, j)
        q_star = quantity_demand(p_star, j)
        individual_revenues.append(revenue(p_star, q_star))

    return sum(individual_revenues)


def total_variable_cost(h):
    """Total variable costs of production for a firm in city h."""
    individual_variable_costs = []
    for j in range(num_cities):
        p_star = optimal_price(h, j)
        q_star = quantity_demand(p_star, j)
        individual_variable_costs.append(variable_cost(q_star, h, j))

    return sum(individual_variable_costs)


def total_variable_labor_demand(h):
    """Total variable labor demand for firms in city h."""
    individual_labor_demands = []
    for j in range(num_cities):
        p_star = optimal_price(h, j)
        q_star = quantity_demand(p_star, j)
        tmp_demand = num_firms[h] * variable_labor_demand(q_star, h, j)
        individual_labor_demands.append(tmp_demand)

    return sum(individual_labor_demands)


def variable_cost(quantity, h, j):
    """
    Variable cost of a firm in city h to produce a given quantity of good for
    sale in city j.

    """
    return variable_labor_demand(quantity, h, j) * nominal_wage[h]


def variable_labor_demand(quantity, h, j):
    """
    Variable labor demand by firm in city h to produce a given quantity of good
    for sale in city j.

    """
    return quantity / labor_productivity(h, j)

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
symbolic_jacobian = symbolic_system.jacobian(endog_vars)

# wrap the symbolic equilibrium system and jacobian
vector_vars = (nominal_price_level, nominal_gdp, nominal_wage, num_firms)
params = (f, beta, phi, tau, elasticity_substitution)
args = vector_vars + params

numeric_system = sym.lambdify(args, symbolic_system,
                              modules=[{'ImmutableMatrix': np.array}, "numpy"])
numeric_jacobian = sym.lambdify(args, symbolic_jacobian,
                                modules=[{'ImmutableMatrix': np.array}, "numpy"])
