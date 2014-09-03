import sympy as sym


# define the number of cities
num_cities = 10

# define parameters
phi = sym.var('phi')
elasticity_substitution = sym.DeferredVector('theta')
economic_distance = sym.MatrixSymbol('delta', num_cities, num_cities)

# define variables
nominal_gdp = sym.DeferredVector('Y')
price_level = sym.DeferredVector('P')
nominal_wage = sym.DeferredVector('W')


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
    return nominal_gdp[i] / price_level[i]


def real_wage(i):
    """Real wage in city i."""
    return nominal_wage[i] / price_level[i]


def relative_price(price, j):
    """Relative price of a good in city j."""
    return price / price_level[j]


def relative_price_level(i, j):
    """Relative price level between cities i and j."""
    return price_level[i] / price_level[j]


def revenue(price, quantity):
    """Revenue from producing a certain quantity at a given price."""
    return price * quantity


def total_revenue(h):
    """Total revenue for a firm producing in city h."""
    individual_revenues = []
    for j in range(num_cities):
        p_star = optimal_price(h, j)
        q_star = quantity_demand(optimal_price(h, j), j)
        individual_revenues.append(revenue(p_star, q_star))

    return sum(individual_revenues)
