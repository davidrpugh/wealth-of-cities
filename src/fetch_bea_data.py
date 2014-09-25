"""
Fetches relevant data on Metropolitan Statistical Areas (MSAs) from the Bureau
of Economic Analysis (BEA) data API.

@author : David R. Pugh
@date : 2014-09-23

"""
import pybea


def drop_unused_cols(df):
    """Remove unused columns from the BEA combined BEA DataFrame."""
    # incorporate the unit multiplier before dropping UNIT_MULT!
    df.loc[:, 'DataValue'] = df['DataValue'] * 10**df['UNIT_MULT']
    return df.drop(['NoteRef', 'UNIT_MULT'], axis=1)


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
