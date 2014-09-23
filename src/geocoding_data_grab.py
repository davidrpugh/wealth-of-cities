"""
Grabs latitude and longitude data from Google Geocoding API.

author : David R. Pugh
date : 2014-09-23

"""
import math
import json
import requests

import bea

MY_API_KEY = 'AIzaSyDEq0MIyTYoks-3gjtjy95FE7hHzhnLi7g'
GOOGLE_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
METRO_NAMES = bea.data_frame['GeoName'].unique()


def _extract_latitude_coords(json):
    """Extract the latitude coordinate from raw json data."""
    return tmp_data['results'][0]['geometry']['location']['lat']


def _extract_longitude_coords(json):
    """Extract the latitude coordinate from raw json data."""
    return tmp_data['results'][0]['geometry']['location']['lng']


# this may be inefficient but avoids Google's API throttle!
geocodes = {}

for h, city_h in enumerate(METRO_NAMES[1:]):

    tmp_query = {'address': city_h,
                 'key': MY_API_KEY,
                 }
    tmp_response = requests.get(url=GOOGLE_BASE_URL, params=tmp_query)
    tmp_data = json.loads(tmp_response.content)

    if tmp_data['status'] == 'OK':
        tmp_lat = _extract_latitude_coords(tmp_data)
        tmp_lng = _extract_longitude_coords(tmp_data)
        geocodes[city_h] = {'lat': tmp_lat, 'lng': tmp_lng}

    else:
        break

    print('Done with ' + city_h + '!')


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees).

    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km
