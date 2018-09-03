import pandas as pd
import numpy as np
from collections import defaultdict

def read_weather_data(path):
    """ Takes in a path to a yearly .csv,
        returns a data frame with subset data
        PRECIP, SNOW, TMAX, TMIN
    """
    df = pd.read_csv(path, compression='infer', header=None, index_col=False,
                    names=['station_id',
                            'measurement_date',
                            'measurement_type',
                            'measurement_flag',
                            'quality_flag',
                            'source_flag',
                            'observation_time'],
                           parse_dates=['measurement_date'])
    df=df[pd.isna(df['quality_flag'])]
    weather_data_subset = df[df.measurement_type.isin(['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN'])][['station_id', 'measurement_date', 'measurement_type', 'measurement_flag']]
    return weather_data_subset

def read_metadata_txt(path):
    """Takes in the path to the metadata
    Returns Data frame with:
    STATION ID, LATITUDE, LONGITUDE, ELEVATION
    """
    df = pd.read_csv(path,
                           sep='\s+',  # Fields are separated by one or more spaces
                           usecols=[0, 1, 2, 3, 4],  # Grab only the first 4 columns
                           na_values=[-999.9],  # Missing elevation is noted as -999.9
                           header=None,
                           names=['station_id', 'latitude', 'longitude', 'elevation', 'state'])
    return df

def make_clean_csv(panda_df, dest_path_name):
    """Takes in a pandas df, a dest_path, and a name for file
    """
    panda_df.to_csv(dest_path_name)
    return True

def station_id_lookup(df):
    """Takes in a data frame
    returns a dictionary with the keys as station_id,
    lat, long, elevation, and state as values.
    """
    station_dict= defaultdict()
    values = df.values
    for row in values:
        stationid = row[0]
        data= row[1:]
        station_dict[stationid]=data
    return station_dict

def make_coordinate_array(tile_name='h09v05'):
    """Takes in a tile name, and returns the lat long
    for every index in numpy array
    """
    if tile_name== 'h09v05':
        xoff= -117.4740487
        y_off= 39.9958333
        a= 0.008868148103055
        b= 0
        d= 0
        e=-0.008868148103054807
    else:
        raise ValueError('No information for that tile... yet')
    rows= 2830
    columns= 1127
    lat_long=np.zeros(shape=(rows,columns))
    lat_long_list=[]
    for row in  range(0,rows):
        for col in  range(0,colms):
            lat_long_list.append(pixel2coord(col,row))
    return lat_long_list    
