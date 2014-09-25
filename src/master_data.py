"""
This script combines the Bureau of Economic Analysis (BEA) data on U.S.
metropolitan statistical areas (MSAs) with the geographical coordinates
data from Google Geocoding API.

@author : David R. Pugh
@date : 2014-09-25

"""
import pandas as pd

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
