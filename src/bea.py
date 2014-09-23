"""
Fetches relevant data on Metropolitan Statistical Areas (MSAs) from the Bureau
of Economic Analysis (BEA) data API.

@author : David R. Pugh
@date : 2014-09-23

"""
import pybea


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
    unstacked_df = hierarchical_df.DataValue.unstack('Code')
    panel = unstacked_df.to_panel()
    return panel


def drop_unused_cols(df):
    """Remove unused columns from the BEA combined BEA DataFrame."""
    # incorporate the unit multiplier before dropping UNIT_MULT!
    df.loc[:, 'DataValue'] = df['DataValue'] * 10**df['UNIT_MULT']
    return df.drop(['NoteRef', 'UNIT_MULT'], axis=1)


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


# we are interested in the following variables...
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

# ...in the following years
years = ['2000', '2005', '2010']

# fectch the data from the BEA data api...
raw_data_frame = pybea.get_data(DataSetName='RegionalData',
                                KeyCodes=key_codes,
                                GeoFips='MSA',
                                Year=years)

# ...clean it and the save a copy to disk!
data_frame = drop_unused_cols(raw_data_frame)
data_frame.to_csv('../data/bea/raw_bea_metro_data.csv')

# Convert to a Panel object, rescale, and add additional variables (in place!)
panel = dataframe_to_panel(data_frame)
rescale_variables(panel)
create_new_variables(panel)
