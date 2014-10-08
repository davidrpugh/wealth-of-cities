class Equilibrium(object):

    equations = ?


class Market(object):

    equilibrium = Equilibrium()


class Model(object):

    cities = [City() for city in range(num_cities)]

    equilibrium = Equilibrium()

    market = Market()