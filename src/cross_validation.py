from src.final_product import PlantForecast
import time

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import os
import src.modis_preprocessing as mpre
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import time

def test_for_longterm_mean(high=400):

    gbr_mse_list=[]
    rf_mse_list=[]
    ridge_mse_list=[]

    gbr_score_list=[]
    rf_score_list=[]
    ridge_score_list=[]

    range_list=[]

    pf=PlantForecast()
    pf=pf.load_metadata()
    pf=pf.load_ndvi(preloaded=True)
    pf=pf.load_weather(preloaded=True)

    for e in range(1,high,10):
        start=time.time()
        pf.merge_modis_weather(longterm=e)

        train_df, test_df=pf.train_test_split_by_year()

        X_train= train_df[['PRCP','SNOW','SNOWD','TMAX','TMIN','LT_precip','LT_snow','LT_snowd', 'LT_tmax','LT_tmin','intercept']]
        y_train = train_df[['NDVI']].values.reshape(-1,)

        X_test= test_df[['PRCP','SNOW','SNOWD','TMAX','TMIN','LT_precip','LT_snow','LT_snowd', 'LT_tmax','LT_tmin','intercept']]
        y_test = test_df[['NDVI']].values.reshape(-1,)

        #model = GradientBoostingRegressor(loss='ls', n_estimators=10000,
                                  #learning_rate=0.001,max_depth=100,subsample=0.8)

        gbr = None
        gbr = GradientBoostingRegressor(loss='ls', n_estimators=10000,
                                  learning_rate=0.1,max_depth=100,subsample=0.8)
        gbr.fit(X_train,y_train)
        gbr_pred=gbr.predict(X_test)
        gbr_score=gbr.score(X_test,y_test)
        gbr_mse = mean_squared_error(y_test, gbr_pred)

        rf = None
        rf= RandomForestRegressor()
        rf.fit(X_train,y_train)
        rf_pred= rf.predict(X_test)
        rf_score=rf.score(X_test,y_test)
        rf_mse = mean_squared_error(y_test, rf_pred)

        ridge = None
        ridge= LinearRegression()
        ridge.fit(X_train,y_train)
        ridge_pred= ridge.predict(X_test)
        ridge_score=ridge.score(X_test,y_test)
        ridge_mse = mean_squared_error(y_test, ridge_pred)



        gbr_mse_list.append(gbr_mse)
        rf_mse_list.append(rf_mse)
        ridge_mse_list.append(ridge_mse)

        gbr_score_list.append(gbr_score)
        rf_score_list.append(rf_score)
        ridge_score_list.append(ridge_score)

        range_list.append(e)
        model = None
        print(f'Checked average for {e} days in {time.time()-start} seconds with score {ridge_score}, and mse: {ridge_mse}')
    return range_list, gbr_mse_list, rf_mse_list, ridge_mse_list, gbr_score_list, rf_score_list, ridge_score_list
