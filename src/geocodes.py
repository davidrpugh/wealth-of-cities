import geopy
from geopy.distance import great_circle, vincenty
import numpy as np

import bea

data = bea.data_frame[['GeoName', 'GeoFips']].drop_duplicates()
geolocator = geopy.geocoders.GoogleV3(timeout=10)

geo_coords = {}
for i, city_i in enumerate(data['GeoName']):
    try:
        tmp_idx = data.iloc[i]['GeoFips']
        tmp_loc = geolocator.geocode(city_i)
        geo_coords[tmp_idx] = (tmp_loc.latitude, tmp_loc.longitude)
    except AttributeError:
        print "Can't find " + city_i + "!"


N = len(geo_coords.keys())
great_circle_distance = np.array(np.zeros((N, N)))
vincenty_distance = np.array(np.zeros((N, N)))

for i, city_i in enumerate(geo_coords.keys()):
    for j, city_j in enumerate(geo_coords.keys()):
        tmp_metric = great_circle(geo_coords[city_i], geo_coords[city_j])
        great_circle_distance[i, j] = tmp_metric.kilometers

        tmp_metric = vincenty(geo_coords[city_i], geo_coords[city_j])
        vincenty_distance[i, j] = tmp_metric.kilometers