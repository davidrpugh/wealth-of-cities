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


def download_bea_data(base_url, api_key, key_codes, mesg=False):
    """Downloads data from the BEA data API."""
    data_frames = []

    for key_code in key_codes:

        tmp_data = _download_data_series(base_url, api_key, key_code)
        data_frames.append(tmp_data)

        if mesg:
            print('Done grabbing the {} data!'.format(key_code))
        else:
            pass

    # combine the data frames into a single DataFrame
    combined_data = pd.concat(data_frames)

    return combined_data


def _download_data_series(base_url, api_key, key_code):
    """Download specific data series from the BEA data API."""
    tmp_query = {'UserID': api_key,
                 'method': 'GetData',
                 'datasetname': 'RegionalData',
                 'KeyCode': key_code}

    tmp_response = requests.get(url=base_url, params=tmp_query)
    tmp_raw_data = json.loads(tmp_response.content)
    tmp_data = tmp_raw_data['BEAAPI']['Results']['Data']
    tmp_data = pd.DataFrame(tmp_data)

    return tmp_data


def remove_duplicate_rows(raw_bea_data):
    """Removes duplicate data entries."""
    check_for_dups = ['CL_UNIT', 'Code', 'DataValue', 'GeoFips', 'NoteRef',
                      'TimePeriod', 'UNIT_MULT']  # BEA has duplicate GeoNames!
    clean_data = raw_bea_data.drop_duplicates(subset=check_for_dups)
    return clean_data


def main():
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

    raw_data = download_bea_data(BEA_BASE_URL, MY_API_KEY, key_codes, True)
    clean_data = remove_duplicate_rows(raw_data)

    # save to disk
    clean_data.to_csv('../data/raw_bea_metro_data.csv')

    return clean_data

if __name__ == '__main__':
    bea_metro_data = main()
