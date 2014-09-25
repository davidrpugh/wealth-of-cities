"""
This script combines the Bureau of Economic Analysis (BEA) data on U.S.
metropolitan statistical areas (MSAs) with the geographical coordinates
data from Google Geocoding API.

@author : David R. Pugh
@date : 2014-09-25

"""
import pandas as pd


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
    hierarchical_df = df.set_index(['Code', df.index, 'TimePeriod'])
    unstacked_df = hierarchical_df.DataValue.unstack('Code')
    panel = unstacked_df.to_panel()
    return panel


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

# load the csv file containing the bea data
bea_df = pd.read_csv('../data/bea/raw_bea_metro_data.csv',
                     index_col='GeoFips',
                     usecols=['CL_UNIT', 'Code', 'DataValue', 'GeoFips', 'TimePeriod'],
                     )

# load the csv file containing the geocoordinates
geo_coords_df = pd.read_csv('../data/google/geocoordinates.csv',
                            index_col='GeoFips',
                            usecols=['GeoFips', 'lat', 'lng'],
                            )

# combine the two dataframes and save
dataframe = pd.merge(bea_df, geo_coords_df, left_index=True, right_index=True)
dataframe.to_csv('../data/master.csv')

# convert to a Panel object
panel = dataframe_to_panel(dataframe)

# rescale, and add additional variables (in place!)
rescale_variables(panel)
create_new_variables(panel)
panel.to_pickle('../data/master.pkl')
