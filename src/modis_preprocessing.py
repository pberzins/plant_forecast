import numpy as np
import pandas as pd
import gdal
import calendar
import os
import numpy.ma as ma
import time
import datetime
import psycopg2 as pg2


def cast_array_to_csv(timestamp_array, ndvi_array, product, tile):
    """INPUT:
    Timestamp is a 2d array,
    NDVI is a 3d array
    """
    df = pd.DataFrame()
    df['capture_date'] = timestamp_array
    df['product'] = product
    df['tile_id'] = tile
    df['ndvi'] = ndvi_array

    df['capture_date'] = pd.to_datetime(df['capture_date'])
    df['product'] = df['product'].astype(str)
    df['tile_id'] = df['tile_id'].astype(str)
    df['ndvi'] = df['ndvi']
    return df


def cast_csv_to_postgres(path_to_file, db_name, table_name, loc='localhost'):
    """Takes in a data frame with columns:
    capture_date = date time object
    product = i.e. MOD13A2
    tile_id = i.e. h09v05
    ndvi = 2d numpy array
    """
    conn = pg2.connect(dbname=db_name, host=loc)
    cur = conn.cursor()
    conn.autocommit = True
    make_table_command = f"""CREATE TABLE {table_name}
                                (index int,
                                capture_date date,
                                product text,
                                tile_id text,
                                ndvi integer[][]);"""

    upload_data_command = f"""COPY {table_name}
                                FROM '{path_to_file}'
                                csv header;"""

    cur.execute(make_table_command)
    cur.execute(upload_data_command)

    #run_time= time.time()-start
    # print(f'Uploaded MODIS Data: {f to SQL in about {run_time} seconds')
    conn.close()
    return None


def get_weather_from_sql(start_date=2013, end_date=2014, meta_data=None, state='NM'):
    """Takes in a database, and gets something out of it
    """
    one_state = meta_data[meta_data['state'] == state]
    clause = tuple(one_state['station_id'].values)

    conn = pg2.connect(dbname='weather', host='localhost')
    cur = conn.cursor()

    date_list = list(range(start_date, end_date))
    empty_list = []
    for year in date_list:
        start = time.time()
        print(f'Getting weather data from year: {year}')
        table_name = 'w_'+str(year)[-2:]
        cur.execute(f'SELECT * from {table_name};')

        data = cur.fetchall()
        #wdf= pd.DataFrame(data, columns=['index','station_id','measurement_date','measurement_type', 'measurement_flag'])
        #wdf = wdf.set_index('measurement_date')
        #wdf.drop(columns=['index'], inplace=True)
        empty_list.append(np.array(data))
        end = time.time()-start
        print(f'Weather for year {year} collectd in {end} seconds')
    conn.close()
    return np.concatenate(np.array(empty_list))


def prepare_weather_data_for_merge(df):
    wdf = pd.DataFrame(df, columns=[
                       'index', 'station_id', 'measurement_date', 'measurement_type', 'measurement_flag'])
    wdf = wdf.set_index('measurement_date')
    wdf.drop(columns=['index'], inplace=True)
    wdf.index = pd.to_datetime(wdf.index)
    wdf['measurement_flag'] = wdf['measurement_flag'].astype(float)
    return wdf


def get_average_per_state(meta_data, weather_data, state):
    """Takes in the meta_data_df, and weather_data, and a state,
        returns averages per state on a julian day basis
    """
    one_state = meta_data[meta_data['state'] == state]

    state_set = set(one_state['station_id'])
    weather_set = set(weather_data['station_id'])
    intersection_list = list(weather_set & state_set)

    print(
        f'There are {len(intersection_list)} stations in the state of {state}')

    all_stations_in_state = weather_data[weather_data['station_id'].isin(
        intersection_list)]
    pivoted = pd.pivot_table(all_stations_in_state, index=[
                             'station_id', 'measurement_date'],
                             columns='measurement_type', values='measurement_flag')
    grouped_by_day = pivoted.groupby('measurement_date').mean()

    return grouped_by_day


def make_coordinate_array(data, geom):
    lat_list = []
    long_list = []
    counter = 0
    for row in range(0, len(sample_data)):
        for column in range(0, len(sample_data[0])):
            start = time.time()
            tupel = pf.pixel2coord(row, column, geom)
            lat_list.append(tupel)
            print(f'calculated in {time.time()-start} seconds!')
            counter += 1
    latitude_array = np.array(lat_list)
    return latitude_array


if __name__ == "__main__":
    pass
