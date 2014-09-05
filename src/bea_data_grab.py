"""
This Python script downloads the relevant JSON data from the Bureau of Economic
Analysis (BEA) data API, re-formats the JSON data into a nice csv file, and
then saves the csv file to disk for later use.

If interested in the gory details of the BEA data API, I suggest you read the
user manual:

    http://www.bea.gov/api/_pdf/bea_web_service_api_user_guide.pdf

Currently only Metropolitan Statistical Area (MSA) data is available via the
BEA data API.  I have put in a request with a contact at BEA for the
Micropolitan Statistical Area data be made available as soon as possible.

@author : David R. Pugh
@date : 2014-09-03

"""
import pandas as pd
import json
import requests

# to access the BEA data we need an API key
MY_API_KEY = '6F508DBE-6979-4E85-A335-8DAE6187FC0D'

# the BEA uniform reference identifier serves as the base URL
BEA_BASE_URL = 'http://www.bea.gov/api/data?'

# we are interested in grabbing the following data series:
key_codes = ['POP_MI',     # Total MSA population
             'GDP_MP',     # Nominal GDP
             'RGDP_MP',    # Real GDP
             'PCRGDP_MP',  # Per capita real GDP
             'TPI_MI',     # Total personal income
             'PCPI_MI',    # Per capita personal income
             'DIR_MI',     # Dividends, interest, and rent
             'PCTR_MI',    # Personal current transfer receipts
             'WS_MI',      # Wage and salary dispursements
             'SUPP_MI',    # Supplements to wages and salary
             'PROP_MI',    # Proprietors income
             ]

data_frames = []

for key_code in key_codes:

    tmp_query = {'UserID': MY_API_KEY,
                 'method': 'GetData',
                 'datasetname': 'RegionalData',
                 'KeyCode': key_code}

    # connect to the BEA data API
    tmp_response = requests.get(url=BEA_BASE_URL, params=tmp_query)

    # load the JSON data as a Python dict
    tmp_raw_data = json.loads(tmp_response.content)

    # key into the actual data
    tmp_data = tmp_raw_data['BEAAPI']['Results']['Data']

    # store it as a Pandas DataFrame
    tmp_data = pd.DataFrame(tmp_data)

    data_frames.append(tmp_data)

    print('Done grabbing the {} data!'.format(key_code))

# combine the data frames into a single master data frame...
combined_data = pd.concat(data_frames)

# for some reason BEA now has duplicate entries for GeoName!
check_for_dups = ['CL_UNIT', 'Code', 'DataValue', 'GeoFips', 'NoteRef',
                  'TimePeriod', 'UNIT_MULT']
bea_metro_data = combined_data.drop_duplicates(subset=check_for_dups)

# ...and then write it to a csv file again!
bea_metro_data.to_csv('../data/raw_bea_metro_data.csv')
