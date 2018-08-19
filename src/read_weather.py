import pandas as pd
import numpy as np

def read_weather_data(path):
    """ Takes in a path to a yearly data_frame,
        puts out a clean data frame
    """
    df = pd.read_csv(path, header=None, index_col=False,
                    names=['station_id',
                            'measurement_date',
                            'measurement_type',
                            'measurement_flag',
                            'quality_flag',
                            'source_flag',
                            'observation_time'],
                           parse_dates=['measurement_date'])
    weather_data_subset = df[df.measurement_type.isin(['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'])][['station_id', 'measurement_date', 'measurement_type', 'measurement_flag']]
    return weather_data_subset

def read_metadata_txt(path):
    """Takes in the path to the metadata
    returns a dataframe of meta data
    """
    df = pd.read_csv(path,
                           sep='\s+',  # Fields are separated by one or more spaces
                           usecols=[0, 1, 2, 3],  # Grab only the first 4 columns
                           na_values=[-999.9],  # Missing elevation is noted as -999.9
                           header=None,
                           names=['station_id', 'latitude', 'longitude', 'elevation'])
    return df
