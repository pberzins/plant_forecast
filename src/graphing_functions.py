import pandas as pd
import numpy as np

def get_average_per_state(meta_data,weather_data, state):
    """Takes in the meta_data_df, and weather_data, and a state,
        returns averages per state on a julian day basis
    """
    one_state=meta_data[meta_data['state']==state]

    state_set = set(one_state['station_id'])
    weather_set=set(weather_data['station_id'])
    intersection_list = list(weather_set& state_set)

    all_stations_in_state= weather_data[weather_data['station_id'].isin(intersection_list)]


    pivoted = pd.pivot_table(all_stations_in_state,index=['station_id','measurement_date'], columns='measurement_type', values='measurement_flag')
    grouped_by_day= pivoted.groupby('measurement_date').mean()

    return get_julian_day_column(grouped_by_day)

def get_julian_day_column(df):
    """Takes in a data frame,
    adds a julian day column
    """
    df['julian']= df.index.to_julian_date()
    return df
