from geopy.distance import great_circle, vincenty
import pandas as pd
import numpy as np


def compute_physical_distance(geo_coords):
    """
    Compute measures of physical distance given geo coordinates.

    Parameters
    ----------
    geo_coords : DataFrame
        Pandas DataFrame containing geo coordinate data for U.S. metropolitan
        statistical areas (MSAs).

    Returns
    -------
    great_circle_distance, vincenty_distance : tuple
        Two (N, N) matrices containing different measurements of physical
        distance between MSAs.

    """
    N = geo_coords.shape[0]
    great_circle_distance = np.array(np.zeros((N, N)))
    vincenty_distance = np.array(np.zeros((N, N)))

    # inefficient given that distance is symmetric!
    for i, geo_fips_1 in enumerate(geo_coords.index):
        for j, geo_fips_2 in enumerate(geo_coords.index):
            tmp_geo_coords_1 = tuple(geo_coords.ix[geo_fips_1])
            tmp_geo_coords_2 = tuple(geo_coords.ix[geo_fips_2])

            tmp_metric = great_circle(tmp_geo_coords_1, tmp_geo_coords_2)
            great_circle_distance[i, j] = tmp_metric.kilometers

            tmp_metric = vincenty(tmp_geo_coords_1, tmp_geo_coords_2)
            vincenty_distance[i, j] = tmp_metric.kilometers

    return great_circle_distance, vincenty_distance


# load the csv file containing the geocoordinates
geo_coords = pd.read_csv('../data/google/geocoordinates.csv', index_col=0)

# compute the physical distance matrices
physical_distance_matrices = compute_physical_distance(geo_coords)
great_circle_distance, vincenty_distance = physical_distance_matrices

# normalize the distance measures (for numeric purposes)
normed_vincenty_distance = vincenty_distance / vincenty_distance.max()
normed_great_circle_distance = great_circle_distance / great_circle_distance.max()
