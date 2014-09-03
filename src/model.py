import sympy as sym


# define parameters
elasticity_substitution = sym.DeferredVector('theta')

# define variables
nominal_gdp = sym.DeferredVector('Y')
price_level = sym.DeferredVector('P')
nominal_wage = sym.DeferredVector('W')


def real_gdp(i):
    """Real gross domestic product of city i."""
    return nominal_gdp[i] / price_level[i]


def real_wage(i):
    """Real wage in city i."""
    return nominal_wage[i] / price_level[i]


def relative_prices(i, j):
    """Relative price level between cities i and j."""
    return price_level[i] / price_level[j]
