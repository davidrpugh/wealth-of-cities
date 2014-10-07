from geopy.distance import great_circle, vincenty
import pandas as pd
import numpy as np


def compute_physical_distance(data):
    """
    Compute measures of physical distance given geographical coordinates for
    U.S. metropolitan statistical areas (MSAs).

    Parameters
    ----------
    data : DataFrame (shape = (N, 2))
        Pandas DataFrame containing geographical coordinate data for MSAs.

    Returns
    -------
    great_circle_distance, vincenty_distance : tuple
        Two ndarrays with shape (N, N) containing different measurements of
        physical distance between MSAs.

    """
    N = data.shape[0]
    great_circle_distance = np.array(np.zeros((N, N)))
    vincenty_distance = np.array(np.zeros((N, N)))

    # inefficient given that distance is symmetric!
    for i, geo_fips_1 in enumerate(data.index):
        for j, geo_fips_2 in enumerate(data.index):
            tmp_coords_1 = tuple(data.ix[geo_fips_1])
            tmp_coords_2 = tuple(data.ix[geo_fips_2])

            tmp_metric = great_circle(tmp_coords_1, tmp_coords_2)
            great_circle_distance[i, j] = tmp_metric.kilometers

            tmp_metric = vincenty(tmp_coords_1, tmp_coords_2)
            vincenty_distance[i, j] = tmp_metric.kilometers

    return great_circle_distance, vincenty_distance


# load the csv file containing the geocoordinates
data = pd.read_csv('../data/master.csv', index_col='GeoFips')

# extract relevant subset of data and sort descending on nominal GDP
subset = data.loc[(data.Code == 'GDP_MP') & (data.TimePeriod == 2010)]
sorted_subset = subset.sort('DataValue', ascending=False)
geo_coords = sorted_subset[['lat', 'lng']].drop([998, 48260])

# compute the physical distance matrices
physical_distance_matrices = compute_physical_distance(geo_coords)
great_circle_distance, vincenty_distance = physical_distance_matrices

# normalize the distance measures (for numeric purposes)
normed_vincenty_distance = vincenty_distance / vincenty_distance.max()
normed_great_circle_distance = great_circle_distance / great_circle_distance.max()

# save the resulting arrays to disk
with open('../data/google/normed_vincenty_distance.npy', 'w') as results:
    np.save(results, normed_vincenty_distance)

with open('../data/google/normed_great_circle_distance.npy', 'w') as results:
    np.save(results, normed_great_circle_distance)
