import sympy as sym

# define variables
nominal_gdp = sym.DeferredVector('Y')
price_level = sym.DeferredVector('P')


def real_gdp(i):
    """Real gross domestic product of city i."""
    return nominal_gdp[i] / price_level[i]


