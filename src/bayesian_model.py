from src.final_product import PlantForecast
import src.graphing_tools as gtools

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os
import time
import gdal

import matplotlib.pyplot as plt
import matplotlib

import sampyl as smp
from sampyl import np


if __name__ == '__main__':
    pf = PlantForecast()
    pf.load_metadata()
    pf.load_ndvi(preloaded=True)
    pf.load_weather(preloaded=True)
    pf.merge_modis_weather(longterm=365)

    train_df, test_df = pf.train_test_split_by_year(
        test_years=[2015, 2016, 2017], train_years=list(range(2000, 2015)))
    X_train = train_df[['PRCP', 'SNOW', 'SNOWD', 'TMAX', 'TMIN',
                        'LT_precip', 'LT_snow', 'LT_snowd', 'LT_tmax', 'LT_tmin', 'intercept']]
    y_train = train_df[['NDVI']].values.reshape(-1,)

    X_test = test_df[['PRCP', 'SNOW', 'SNOWD', 'TMAX', 'TMIN', 'LT_precip',
                      'LT_snow', 'LT_snowd', 'LT_tmax', 'LT_tmin', 'intercept']]
    y_test = test_df[['NDVI']].values.reshape(-1,)

    lin = None
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_score = lin.score(X_test, y_test)
    lin_mse = mean_squared_error(y_test, lin_pred)
    co = lin.coef_
