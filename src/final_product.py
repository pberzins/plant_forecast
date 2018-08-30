import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

import psycopg2 as pg2
import sqlalchemy
import s3fs

import julian
import datetime
import calendar
import time

import gdal
import os

import src.modis_preprocessing as mpre
import src.read_weather as rw

class PlantForecast():
    """ EXAMPLE:
        pf = PlantForecast()
        pf.load_metadata()
        pf.load_ndvi()
        pf.load_weather()
        pf.merge_modis_weather()

        train_df, test_df = pf.train_test_split_by_year([2015,2016,2017])

        ***-__-__-__-***

        Model what you want to find!
        ***-__-__-__-***

        Graph Results!

        ***
    """

    def __init__(self,tiff_files_path='/Users/Berzyy/plant_forecast/data/modis_co/tiff_files/',
                meta_data_path='data/weather/meta_data/ghcnd-stations.txt',
                db_name='weather',
                host='localhost'):
                """INPUTS:
                tiff_files_path: Path to where you have two folders of tiff files:
                    -One folder should be called "NDVI", and the other "Quality"
                meta_data_path: Path to where ghcnd-stations.txt is downloaded
                db_name= The name of the PostGRESQL database
                host= Who is hosting the server
                """
                self.tiff_path= tiff_files_path
                self.meta_data_path= meta_data_path
                self.db_name = db_name
                self.host = host

    def load_metadata(self):
        """Loads the ghcnd-stations.txt into a Pandas DataFrame.
        OUTPUT:
        A Pandas Data frame with columns:
        station_id|latitude|longitude|elevation|state
        Returns self
        """
        df = pd.read_csv(self.meta_data_path,
                               sep='\s+',
                               usecols=[0, 1, 2, 3, 4],
                               na_values=[-999.9],  # Missing elevation is noted as -999.9
                               header=None,
                               names=['station_id', 'latitude', 'longitude', 'elevation', 'state'])
        self.meta_data = df

        return self

    def load_ndvi(self,preloaded=False, preloaded_path='/Users/Berzyy/plant_forecast/data/weather/2000_2017_ndvi.csv'):
        """INPUT: self
            OUTPUT:
            A Pandas DataFrame with columns:
            Date| NDVI
        """
        if preloaded == True and preloaded_path!= None:
            df= pd.read_csv(preloaded_path)
            df = df.set_index('measurement_date')
            df.index = pd.to_datetime(df.index)
            self.ndvi = df
            return self

        else:
            self.ndvi= self.modis_powerhouse(self.tiff_path)
            return self

    def load_weather(self, preloaded=False, preloaded_path='/Users/Berzyy/plant_forecast/data/weather/2000_2017_nm.csv'):
        """INPUTS:
            preloaded: True or False, if there is a preloaded CSV
            preloaded_path: Path to CSV with columns:
            measurement_date|PRCP|SNOW|SNWD|TMAX|TMIN
            If both are none, it will pull from PostGreSQL
            OUTPUT:
            A Pandas Data Frame named self.weather
        """
        if preloaded == True and preloaded_path!= None:
            df= pd.read_csv(preloaded_path)
            df = df.set_index('measurement_date')
            df.index = pd.to_datetime(df.index)
            self.weather= df
            return self

        else:
            conn = pg2.connect(dbname=self.db_name, host=self.host)
            cur = conn.cursor()

            conn.autocommit=True

            get_weather_in_tile_command = """SELECT station_id
                                            FROM station_metadata
                                            WHERE ST_Contains(ST_GeomFromText('POLYGON(
                                            (-117.4740487 39.9958333,
                                            -104.23 39.75,
                                            -92.3771896 30.0014304,
                                            -103.7  30.0014304,
                                            -117.4740487 39.9958333))',4326),geom);"""

            cur.execute(get_weather_in_tile_command)
            data = list(cur.fetchall())
            stations = tuple(map(' '.join, data))

            table_list= ['w_00','w_01','w_02','w_03','w_04', 'w_05', 'w_06','w_07','w_08','w_09','w_10', 'w_11','w_12','w_13','w_14','w_15','w_16','w_17']

            data_frame_list= []
            for e in table_list:
                start = time.time()

                get_weather_command= f"""SELECT * FROM {e} WHERE station_id in {stations} limit 100;"""
                cur.execute(get_weather_command)
                weather = cur.fetchall()

                df = self.prepare_weather_data_for_merge(weather)
                df = self.pivot_weather_data_frame(df)

                data_frame_list.append(df)
                print(f"Got data from table: {e} in about {time.time()-start} seconds!")
            conn.close()
            df = pd.concat(data_frame_list)
            self.weather=df
            return self

    def merge_modis_weather(self, longterm=100):
        """OUTPUT:
        Merges NDVI data and WeatherData into Pandas Dataframe with columns:
        measurement_date|PRCP|SNOW|SNWD|TMAX|TMIN|NDVI
        """
        df = self.time_delta_merge(self.ndvi,self.weather, longterm)
        #print(df.columns)
        df['intercept']=1
        df.dropna(inplace=True)
        self.combined= df
        return self

    def train_test_split_by_year(self,test_years=[2017],train_years=list(range(2000,2010))):
        """INPUT:
        test_year= list of years held out of fitting of model
        OUTPUT:
        Training DataFrame and Testing DataFrame
        """
        test_df=self.combined[self.combined.index.year.isin(test_years)]
        train_df=self.combined[self.combined.index.year.isin(train_years)]
        self.test = test_df
        self.train= train_df
        return train_df, test_df


    def time_delta_merge(self,ndvi_df, weather_df,longterm=100):
        """Takes in two data frames,
        ndvi_df columns: "measurement_date|ndvi"
        weather_df columns: "meaurment_date|PRCP|SNOW|SNWD|TMAX|TMIN"
        Averages all weather from the past 16 days for every date that there is
        a satellite image, including that day.
        """
        satellite_data = ndvi_df.index.values
        ndvi_weather_aggregate_list =[]
        #print(longterm)
        for e in satellite_data[1:]:
            ndvi_value_for_date= ndvi_df[ndvi_df.index==e]['ndvi'].values

            rng = pd.date_range(end=e, periods=16, freq='D')
            subset = weather_df[weather_df.index.isin(rng)]
            #print(weather_df[weather_df.index==e])
            mean= subset.mean().values
            #return mean

            precip_range= pd.date_range(end=e, periods=longterm, freq='D')
            precip_subset= weather_df[weather_df.index.isin(precip_range)]

            long_mean=precip_subset.mean().values
            long_sum=precip_subset.sum().values

            datum = np.array([e,mean[0],mean[1],mean[2],mean[3],mean[4],
                                long_mean[0],long_mean[1],long_mean[2],
                                long_mean[3],long_mean[4],ndvi_value_for_date[0]])

                                #,long_sum[0],long_sum[1],long_sum[2],ndvi_value_for_date[0]])

            ndvi_weather_aggregate_list.append(datum)

        data= np.array(ndvi_weather_aggregate_list)[:,1:]
        indi= np.array(ndvi_weather_aggregate_list)[:,0]

        df = pd.DataFrame(data=data, index=indi,
                            columns=['PRCP','SNOW','SNOWD','TMAX','TMIN',
                                    'LT_precip','LT_snow','LT_snowd', 'LT_tmax',
                                    'LT_tmin','NDVI'])
                                    #'s_precip','s_snow','s_snowd','NDVI'])

        return df

    def pivot_weather_data_frame(self,df):
        """INPUTS: Self, df
        OUTPUTS:
        A Pandas DataFrame with columns:
        PRCP|SNOW|SNWD|TMAX|TMIN with values in corresponding "measurment_flag" as floats
        """
        pivoted = pd.pivot_table(df,index=['station_id','measurement_date'], columns='measurement_type', values='measurement_flag')
        grouped_by_day= pivoted.groupby('measurement_date').mean()

        #return get_julian_day_column(grouped_by_day)
        return grouped_by_day

    def prepare_weather_data_for_merge(self,df):
        """INPUTS:
        Self, and a DataFrame, an SQL QUERY
        OUTPUTS:
        A Data frame with index of 'measurement_date',

        """
        wdf= pd.DataFrame(df, columns=['index','station_id','measurement_date','measurement_type', 'measurement_flag'])
        wdf = wdf.set_index('measurement_date')
        wdf.drop(columns=['index'], inplace=True)
        wdf.index = pd.to_datetime(wdf.index)
        wdf['measurement_flag']=wdf['measurement_flag'].astype(float)
        return wdf

    def modis_powerhouse(self,path):
        """Takes in a path to a folder where there are two folders:
            1.) ndvi_tiff
            2.) quality_tiff
        Takes in this folder and casts NDVI values and Quality Values into 2d arrays
        """
        quality_folder_path= path+ 'quality_tiff/'
        ndvi_folder_path= path+ 'ndvi_tiff/'
        file_list= os.listdir(ndvi_folder_path)

        ndvi_file_set=  set(list(f for f in os.listdir(ndvi_folder_path) if f.endswith('.' + 'tif')))
        quality_file_set= set(list(f for f in os.listdir(quality_folder_path) if f.endswith('.' + 'tif')))

        cross_checked = sorted(ndvi_file_set&quality_file_set)
        array_list = []
        for f in cross_checked:
            start= time.time()
            product= f[:7]
            year= f[9:13]
            julian_day= f[13:16]
            tile= f[17:23]

            ndvi_file= ndvi_folder_path+f
            quality_file= quality_folder_path+f

            ndvi = gdal.Open(ndvi_file)
            n_band = ndvi.GetRasterBand(1)
            n_arr = n_band.ReadAsArray()
            ndvi= None

            quality = gdal.Open(quality_file)
            q_band = quality.GetRasterBand(1)
            q_arr = q_band.ReadAsArray()
            quality= None

            data=self.quality_screen(q_arr, n_arr)

            av=data[data!=-3000].mean()
            date_time=self.JulianDate_to_MMDDYYY(int(year),int(julian_day))

            date = np.full(data.shape, date_time)

            tile_array = np.array([date[0][0], av])
            array_list.append(tile_array)
            print(f'Compiled Matrix for {date_time} in {time.time()-start} seconds')



        df = pd.DataFrame(np.array(np.array(array_list)), columns=['measurement_date','ndvi'])
        df = df.set_index('measurement_date')
        df.index = pd.to_datetime(df.index)

        return df

    def quality_screen(self,quality, ndvi):
        """INPUTS
            quality= 2D array
            ndvi = 2d array
        OUTPUTS:
            A matrix with all quality flags != 1 set to -3000 (no_fill value)
        """
        ndvi[quality!=0]= -3000
        return ndvi

    def JulianDate_to_MMDDYYY(self,y,jd):
        """INPUTS:
        Takes in a year and Julian Date of year
        OUTPUS:
        Returns a date time object
        """
        month = 1
        day = 0
        while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
            jd = jd - calendar.monthrange(y,month)[1]
            month = month + 1
        return datetime.date(y, month, jd)
