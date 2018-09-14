import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.final_product import PlantForecast
import src.modis_preprocessing as mpre
import src.cross_validation as cv


def plot_over_time(X_train, X_test, y_train, y_test):
    """Takes in a split dataframe and graphs it as a time series
    """
    gbr = None
    gbr = GradientBoostingRegressor(loss='ls', n_estimators=10000,
                                    learning_rate=0.001, max_depth=5, subsample=0.5)
    print('Fitting Gradient Boosted Model')
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)
    gbr_score = gbr.score(X_test, y_test)
    gbr_mse = mean_squared_error(y_test, gbr_pred)

    rf = None
    rf = RandomForestRegressor()
    print('Fitting Random Forest Model')
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    rf_mse = mean_squared_error(y_test, rf_pred)

    lin = None
    lin = LinearRegression()
    print('Fitting Linear Regression Model')
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_score = lin.score(X_test, y_test)
    lin_mse = mean_squared_error(y_test, lin_pred)

    ridge = None
    ridge = Ridge(alpha=0.1)
    print('Fitting Ridge Regression Model')
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_score = ridge.score(X_test, y_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)

    lasso = None
    lasso = Lasso(alpha=.5)
    print('Fitting Lasso Regression Model')
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_score = lasso.score(X_test, y_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(X_test.index, y_test, label='Actual NDVI',
            color='Black', linewidth=3)
    ax.plot(X_test.index, gbr_pred,
            label=f'Gradient Boosted r2={gbr_score:.2f} mse={gbr_mse:.2f}', color='C0')
    ax.plot(X_test.index, rf_pred,
            label=f'Random Forest r2={rf_score:.2f} mse={rf_mse:.2f}', color='C1')
    ax.plot(X_test.index, lin_pred,
            label=f'Linear Regression r2={lin_score:.2f} mse={lin_mse:.2f}', color='C2')
    #ax.plot(X_test.index, ridge_pred, label=f'Ridge Regression r2={ridge_score:.2f} mse={ridge_mse:.2f}', color='red')
    #ax.plot(X_test.index, lasso_pred, label=f'Lasso Regression Predicted NDVI r2:{lasso_score}', color='blue')

    #ax2 = ax.twinx()
    #ax2.plot(X_test.index, X_test['TMAX'], color='Red',linestyle=':')
    #ax2.plot(X_test.index, X_test['TMIN'], color='teal',linestyle=':')

    #ax3= ax.twinx()
    #ax3.bar(X_test.index, -X_test['PRCP'].astype(float), 3,label='precipitation',color='blue')
    #ax3.bar(X_test.index, -X_test['SNOW'].astype(float), 3,label='snow',color='yellow')
    # ax3.set_ylim(-200,0)

    ax.set_title(
        'Linear Model, Gradient Boosted, Random Forest, and Ridge Regression, \n Trained 2000-2015, Tested 2015-2017')
    ax.set_ylabel('NDVI scaled 0.0001')
    ax.set_xlabel('Date')
    #ax.fill_between(X_test.index, y_test, 0)
    # ax.set_ylim(1800,4800)
    ax.legend(loc=0, prop={'size': 16})
    # ax2.legend(loc=1)
    # ax3.legend(loc=2)

    plt.show()

    return None


def graph_lag_mse(range_list, gbr_mse_list, rf_mse_list, ridge_mse_list, gbr_score_list, rf_score_list, ridge_score_list):
    """
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range_list, gbr_mse_list, label='Gradient Boosted MSE')
    ax.plot(range_list, rf_mse_list, label='Random Forest MSE')
    ax.plot(range_list, ridge_mse_list, label='Linear Regression MSE')

    ax.set_title(
        'Gradient Boosted, Random Forest, and Linear Regression \n MSE in relation to size of rolling average window')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Number of Lagged Days')
    ax.legend(prop={'size': 14})

    plt.show()

    return None
    