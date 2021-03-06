import geopy
import pandas as pd

import fetch_bea_data


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
            clean_geo_name = geo_name[:-32]  # drop (Metropolitan Statistical Area)
            tmp_idx = data.iloc[i]['GeoFips']
            tmp_loc = geolocator.geocode(clean_geo_name)
            geo_coords[i] = {'GeoFips': tmp_idx,
                             'lat': tmp_loc.latitude,
                             'lng': tmp_loc.longitude}
        except AttributeError:
            print("Can't find " + geo_name + "!")

    df = pd.DataFrame.from_dict(geo_coords, orient='index')
    return df

# load the place names from the BEA data
data = fetch_bea_data.dataframe[['GeoName', 'GeoFips']].drop_duplicates()

# define a geolocator
geolocator = geopy.geocoders.GoogleV3(timeout=10)

# grab and save the geo_coordinates data
geo_coords = get_geo_coords(data, geolocator)
geo_coords.to_csv('../data/google/geocoordinates.csv')
