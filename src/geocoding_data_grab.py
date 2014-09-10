"""
Apparently I can increase our quota by registering a billing credit card. All
I need to do is get our quota to 100,000 elements per day and we ar golden. We 
only need to grab the data once!

"""
import json
import numpy as np
import pandas as pd
import requests

MY_API_KEY = 'AIzaSyDEq0MIyTYoks-3gjtjy95FE7hHzhnLi7g'
GOOGLE_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json'

metro_names = ['San Francisco-Oakland, MSA']

# this may be inefficient but avoids Google's API throttle!
for h, city_h in enumerate(metro_names):

    tmp_query = {'address': city_h,
                 'key': MY_API_KEY,
                 }

    tmp_response = requests.get(url=GOOGLE_BASE_URL, params=tmp_query)
    tmp_data = json.loads(tmp_response.content)

    print('Done with ' + city_h + '!')
