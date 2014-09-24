import geopy
from geopy.distance import great_circle, vincenty
import numpy as np
import pandas as pd

import bea

# load the place names from the BEA data
data = bea.data_frame[['GeoName', 'GeoFips']].drop_duplicates()

geolocator = geopy.geocoders.GoogleV3(timeout=10)


def get_geo_coords(places, geolocator):
    """
    Return a Pandas DataFrame storing the latitude and longitude coords.

    Parameters
    ----------
    places : Series
        Pandas Series containing place names.
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
    for i, city_i in enumerate(data['GeoName']):
        try:
            tmp_idx = data.iloc[i]['GeoFips']
            tmp_loc = geolocator.geocode(city_i)
            geo_coords[tmp_idx] = {'lat': tmp_loc.latitude,
                                   'lng': tmp_loc.longitude}
        except AttributeError:
            print("Can't find " + city_i + "!")

    df = pd.DataFrame.from_dict(geo_coords, orient='index')
    return df


# this loop should create two ndarrays storing measures of pair-wise physical distances
N = len(geo_coords.keys())
great_circle_distance = np.array(np.zeros((N, N)))
vincenty_distance = np.array(np.zeros((N, N)))

for i, city_i in enumerate(geo_coords.keys()):
    for j, city_j in enumerate(geo_coords.keys()):
        tmp_metric = great_circle(geo_coords[city_i], geo_coords[city_j])
        great_circle_distance[i, j] = tmp_metric.kilometers

        tmp_metric = vincenty(geo_coords[city_i], geo_coords[city_j])
        vincenty_distance[i, j] = tmp_metric.kilometers