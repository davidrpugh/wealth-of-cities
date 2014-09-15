"""
Notes
-----

The grab_multiple_key_codes function should be incorporated into pyBEA.

"""
import pandas as pd

import pybea


def clean_dataframe(df):
    """Clean the combined BEA combined DataFrame."""
    drop_duplicate_rows(df)
    drop_unused_cols(df)


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
    """Convert combined BEA DataFrame to Pandas Panel object."""
    hierarchical_df = df.set_index(['Code', 'TimePeriod', 'GeoFips'])
    unstacked_df = hierarchical_df .DataValue.unstack('Code')
    panel = unstacked_df.to_panel()
    return panel


def drop_duplicate_rows(df):
    """Remove duplicate rows from the BEA combined BEA DataFrame."""
    check_for_dups = ['CL_UNIT', 'Code', 'DataValue', 'GeoFips', 'TimePeriod',
                      'UNIT_MULT']  # BEA has duplicate GeoNames!
    df.drop_duplicates(subset=check_for_dups, inplace=True)


def drop_unused_cols(df):
    """Remove unused columns from the BEA combined BEA DataFrame."""
    # incorporate the unit multiplier before dropping UNIT_MULT!
    df.loc[:, 'DataValue'] = df['DataValue'] * 10**df['UNIT_MULT']
    df.drop(['NoteRef', 'UNIT_MULT'], axis=1, inplace=True)


def grab_multiple_key_codes(user_id, key_codes, geo_fips, year):
    """Combines data on multiple key codes into a single Pandas DataFrame."""
    dfs = []
    for key_code in key_codes:
        tmp_df = pybea.get_data(UserID=user_id,
                                DataSetName='RegionalData',
                                KeyCode=key_code,
                                GeoFips=geo_fips,
                                Year=year)
        dfs.append(tmp_df)
        print("Done with " + key_code + "!")

    combined_df = pd.concat(dfs)

    return combined_df


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

    combined_df = grab_multiple_key_codes(MY_API_KEY, key_codes, 'MSA', '2010')

    # clean the combined dataframe (done inplace!)
    clean_dataframe(combined_df)

    # save to disk
    combined_df.to_csv('../data/bea/raw_bea_metro_data.csv')

    # convert to panel data object
    panel = dataframe_to_panel(combined_df)

    # rescale old variables and then create new ones (done inplace!)
    rescale_variables(panel)
    create_new_variables(panel)

    return panel

if __name__ == '__main__':
    bea_metro_panel = main()