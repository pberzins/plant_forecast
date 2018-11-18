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
