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

def test_for_longterm_mean(high=150):
    score_list=[]
    range_list=[]
    mse_list=[]
    pf=PlantForecast()
    pf=pf.load_metadata()
    pf=pf.load_ndvi(preloaded=True)
    pf=pf.load_weather(preloaded=True)
    for e in range(40,high,10):
        start=time.time()
        pf.merge_modis_weather(longterm=e)
        train_df, test_df=pf.train_test_split_by_year(test_years=list(range(2010,2018)))

        X_train = train_df[['PRCP','SNOW','SNOWD','TMAX','TMIN','LT_precip']]
        y_train = train_df[['NDVI']].values

        X_test = test_df[['PRCP','SNOW','SNOWD','TMAX','TMIN','LT_precip']]
        y_test = test_df[['NDVI']].values

        model = GradientBoostingRegressor(loss='ls', n_estimators=10000,
                                  learning_rate=0.1,max_depth=100,subsample=0.8)
        #model= Ridge()
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        score = model.score(X_test,y_test)
        mse = mean_squared_error(y_test, y_pred)

        mse_list.append(mse)
        score_list.append(score)
        range_list.append(e)
        model = None
        print(f'Checked average for {e} days in {time.time()-start} seconds with score {score}, and mse: {mse}')
    return score_list, range_list, mse_list
