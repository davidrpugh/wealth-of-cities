import sympy as sym


# define the number of cities
N = 10

# define parameters
phi = sym.var('phi')
elasticity_substitution = sym.DeferredVector('theta')
economic_distance = sym.MatrixSymbol('delta', N, N)

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


def price(h, j):
    """Price of good j sold in city h."""
    return mark_up(j) * marginal_costs(h, j)


def real_gdp(i):
    """Real gross domestic product of city i."""
    return nominal_gdp[i] / price_level[i]


def real_wage(i):
    """Real wage in city i."""
    return nominal_wage[i] / price_level[i]


def relative_prices(i, j):
    """Relative price level between cities i and j."""
    return price_level[i] / price_level[j]    
