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
import glob
import json
import numpy as np
import pandas as pd
import requests


def clean_dataframe(df):
    """Clean the combined BEA combined DataFrame."""
    remove_duplicate_rows(df)
    remove_unused_cols(df)


def combine_json_files_to_dataframe(json_files):
    """Combines the raw BEA json files into a single Pandas DataFrame."""
    dfs = []
    for json_file in json_files:
        tmp_df = json_file_to_dataframe(json_file)
        dfs.append(tmp_df)

    combined_df = pd.concat(dfs)
    return combined_df


def create_new_variables(panel):
    """Create some additional variables of interest."""

    # per capita nominal GDP (thousands of USD)
    panel['PCGDP_MP'] = panel['GDP_MP'] / panel['POP_MI']

    # per capita wages (thousands of USD)
    panel['PCWS_MI'] = panel['WS_MI'] / panel['POP_MI']

    # total employee compensation including pensions, etc (billions of USD)
    panel['COE_MI'] = panel['WS_MI'] + panel['SUPP_MI']

    # per capita employee compensation (thousands of USD)
    panel['PCCOE_MI'] = panel['COE_MI'] / panel['POP_MI']

    # per capita dividends, interest, and rent (thousands of USD)
    panel['PCDIR_MI'] = panel['DIR_MI'] / panel['POP_MI']


def dataframe_to_panel(df):
    """Convert BEA combined DataFrame to Pandas Panel object."""
    hierarchical_df = df.set_index(['Code', 'TimePeriod', 'GeoFips'])
    unstacked_df = hierarchical_df .DataValue.unstack('Code')
    panel = unstacked_df.to_panel()
    return panel


def download_raw_json_data(path, base_url, api_key, key_code):
    """Download raw JSON data for given key_code from the BEA data API."""
    tmp_query = {'UserID': api_key,
                 'method': 'GetData',
                 'datasetname': 'RegionalData',
                 'KeyCode': key_code}
    tmp_response = requests.get(url=base_url, params=tmp_query)

    with open(path + key_code + '.json', 'w') as tmp_file:
        tmp_file.write(tmp_response.content)


def json_file_to_dataframe(json_file):
    """Load raw BEA JSON file and convert to a Pandas DataFrame."""
    raw_json_data = json.load(open(json_file))
    tmp_json_data = raw_json_data['BEAAPI']['Results']['Data']
    tmp_data = pd.DataFrame(tmp_json_data, dtype=np.int64)
    return tmp_data


def remove_duplicate_rows(df):
    """Remove duplicate rows from the BEA combined BEA DataFrame."""
    check_for_dups = ['CL_UNIT', 'Code', 'DataValue', 'GeoFips', 'TimePeriod',
                      'UNIT_MULT']  # BEA has duplicate GeoNames!
    df.drop_duplicates(subset=check_for_dups, inplace=True)


def remove_unused_cols(df):
    """Remove unused columns from the BEA combined BEA DataFrame."""
    # incorporate the unit multiplier before dropping UNIT_MULT!
    df.loc[:, 'DataValue'] = df['DataValue'] * 10**df['UNIT_MULT']
    df.drop(['NoteRef', 'UNIT_MULT'], axis=1, inplace=True)


def rescale_variables(panel):
    """Rescale variables to units more appropriate for numerical work."""
    # default BEA unit is USD, the natural unit is billions of USD
    to_billions_USD = ['DIR_MI', 'GDP_MP', 'PCTR_MI', 'PROP_MI', 'RGDP_MP',
                       'SUPP_MI', 'TPI_MI', 'WS_MI']
    # default BEA unit is persons, the natural unit is millions of persons
    to_millions_persons = ['POP_MI']

    # default BEA unit is USD, the natural unit is thousands of USD
    to_thousands_USD = ['PCPI_MI', 'PCRGDP_MP']

    for item in panel.items:
        if item in to_billions_USD:
            panel[item] /= 1e9
        elif item in to_millions_persons:
            panel[item] /= 1e6
        elif item in to_thousands_USD:
            panel[item] /= 1e3
        else:
            pass


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

    # grab the raw BEA data
    for key_code in key_codes:
        download_raw_json_data('../data/bea/', BEA_BASE_URL, MY_API_KEY, key_code)

    # combine into a data frame
    json_files = glob.glob('../data/bea/*.json')
    combined_df = combine_json_files_to_dataframe(json_files)

    # clean the combined dataframe (done inplace!)
    clean_dataframe(combined_df)

    # save to disk
    combined_df.to_csv('../data/bea/raw_bea_metro_data.csv')

    # convert to panel data object
    panel = dataframe_to_panel(combined_df)

    # rescale old variables and create new ones (done inplace!)
    rescale_variables(panel)
    create_new_variables(panel)

    return panel


if __name__ == '__main__':
    panel = main()
