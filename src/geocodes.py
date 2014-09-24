import geopy
from geopy.distance import great_circle, vincenty
import numpy as np
import pandas as pd

import bea

# load the place names from the BEA data
data = bea.data_frame[['GeoName', 'GeoFips']].drop_duplicates()

geolocator = geopy.geocoders.GoogleV3(timeout=10)


def get_geo_coords(data, geolocator):
    """
    Return a Pandas DataFrame storing the latitude and longitude coords.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing GeoName and GeoFips columns.
    geolocator : object
        GeoPy geolocator object used to fetch the latitude and longitude
        coordinates.

    Returns
    -------
    df : DataFrame
        Pandas DataFrame containing the latitude and longitude coordinates
        indexed by GeoFips code.

    """
    geo_coords = {}
    for i, geo_name in enumerate(data['GeoName']):
        try:
            tmp_idx = data.iloc[i]['GeoFips']
            tmp_loc = geolocator.geocode(geo_name)
            geo_coords[tmp_idx] = {'lat': tmp_loc.latitude,
                                   'lng': tmp_loc.longitude}
        except AttributeError:
            print("Can't find " + geo_name + "!")

    df = pd.DataFrame.from_dict(geo_coords, orient='index')
    return df


# this loop should create two ndarrays storing measures of pair-wise physical distances
geo_coords = get_geo_coords(data, geolocator)

N = geo_coords.shape[0]
great_circle_distance = np.array(np.zeros((N, N)))
vincenty_distance = np.array(np.zeros((N, N)))

for i, geo_fips_1 in enumerate(geo_coords):
    for j, geo_fips_2 in enumerate(geo_coords):
        tmp_metric = great_circle(geo_coords[city_i], geo_coords[city_j])
        great_circle_distance[i, j] = tmp_metric.kilometers

        tmp_metric = vincenty(geo_coords[city_i], geo_coords[city_j])
        vincenty_distance[i, j] = tmp_metric.kilometers