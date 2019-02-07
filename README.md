# Remote Sensing from MODIS Satellite to Predict Vegetative Health

Here is a link to my Google Slides Presentation: [link](https://docs.google.com/presentation/d/1hh4mL5_KHlkKRD4tlpuhtyhKs_0CP3Mx-1P6-ZX81nA/edit?usp=sharing)

## Goals of the Project
This project attempts to use remote sensing techniques to predict the future vegetatative health of an area in respect to changing weather. Farmers, Gardeners, and Fire Management officials would be able to use the local weather forecast and predict how a specified region's vegetation will change.

## Data Understanding
How does one measure Vegetative Health?:

**NDVI:** NDVI which stands for Normalized Differenced Vegetation Index is a measure of the amount of "green" in a substance using infrared wavelengths. More information can be found here: [NDVI Wiki](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index)

**Photosynthesis:** Photosynthesis is the production of glucose (sugar) and water from Carbon Dioxide and and Water. More information can be found here: [Photosynthesis Wiki](https://en.wikipedia.org/wiki/Photosynthesis)


## Data Preparation
__This project requires GDAL!__
GDAL can be very difficult to install if python packages are not organized in a particular way. In the case of this project, I had to uninstall and reinstall my Anaconda Environment to get it to work. This is no small task, good luck! [GDAL](https://www.gdal.org/)

## Satellite Data:
Satellite Data is obtained with the PyModis Module which allows you to bulk download MODIS tiles and products available from NASA. For download and documentation of PyModis: [PyModis](http://www.pymodis.org/)

The Naming System of MODIS Products will be described below:

**Example Filename:**

### MOD13A2.A2006001.h08v05.006.2015113045801.hdf

**MOD13A2** : Product Short Name

**A2006001**: Year and Julian Date of acquistion (i.e the 1st Julian day of 2006)

**h08v05**: The tile in which to download.

**006** : Collection Version

**2015113045801**: Julian Date of Production

**hdf** : Data Format

more information on this specific product can be found here: [MOD13A2 Product Information](https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13a2_v006)

### Converting .hdf files into .tiff files
hdf files are very cumbersome to work with, especially since their projection is Sinusoidal. Many people are more accustomed to thinking in terms of latitude and longitude (Geographical). For this project I used a an application called: HDF-EOS To GeoTIFF Conversion Tool (HEG). HEG is very useful for converting HDF's to Geographic projections. Can be downloaded here: [HEG Download](https://newsroom.gsfc.nasa.gov/sdptoolkit/HEG/HEGHome.html)

## Weather Data:

Historical Weather data is very easy to download and manipulate! Weather was downloaded from Global Historical Climate Network Daily (GHCND), which can be downloaded either directly from the FTP server or it can be downloaded using one of the provided data tools. I recommend getting started with their tool if you are not sure exactly what information you are looking for.

More information can be found here: [GHCND Info](https://www.ncdc.noaa.gov/ghcn-daily-description)

__Meta Data:__
The meta data for each station is very important because it gives the Latitude, Longitude, and Elevation. This Meta Data is uploaded to the FTP server and can be found here: [meta_data](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt)

## Using PostGIS and PostGRESQL:
PostGIS is an extension of PostGRESQL which allows the ability for the user to do spatial queries on Tables in SQL. Since the Meta Data provides LAT, LONG parameters, it is very easy to turn those points of information into geometry points in PostGIS:
Documentation can be found here: [PostGIS](https://postgis.net/)

## QGIS to Visualize Results:
QGIS is its own beast, if you have ever used ARCGis and been frustrated how clunky it is, QGIS is the tool for you! QGIS allows you connect directly to a PostGIS database and visualize results. This project does not teach you much about QGIS but it is important to know that it exists, and you will not regret knowing your way around this application.
More information can be found here: [QGIS](https://qgis.org/en/site/)
      

![Picture](https://raw.githubusercontent.com/pberzins/plant_forecast/master/pictures/tile_america.png)

## Modeling
Modeling was done using SciKit Learn to build a variety of models in order to compare when and where specific models are appropriate measures.

### Data for Modeling looks like something like this:

| Date        | TMAX         | TMIN         | SNOW | SNOWD | PRECIP | NDVI     |
|-------------|--------------|--------------|------|-------|--------|----------|
| date_object | 0.1 Degree C | 0.1 Degree C | mm   | mm    | mm     | unitless |

### Graph of MSE with Changing LAG Duration: 


![MSE Graph](https://raw.githubusercontent.com/pberzins/plant_forecast/master/pictures/MSE_per_lag.png)


## TLDR; What you need to succeed:

You should start in the .ipynb file **PlantForecast_demo.pynb** to get yourself accustomed to the structure of the **PlantForecast Class** located in **final_product.py** and see and what it does.


**Python:** from Anaconda **AND** Python.org (very important for QGIS) [Conda](https://anaconda.org/anaconda/python), [Python.org](https://www.python.org/downloads/release/python-366/)

**GDAL:** I will say this again, do not underestimate how hard this can be to download. [GDAL](https://www.gdal.org/)

**PostGRESQL and PostGIS:** [SQL](https://www.postgresql.org/) and [PostGIS](https://postgis.net/)  

**QGIS:** Follow the directions for download to the **T**, if you have conda python but not python.org python, it will mess up and you will not be able to get it to work. [QGIS](https://qgis.org/en/site/)

**HEG:** You will need to use HEG to warp the sinusoidal projection of the HDF files into Geographic Projection (Latitutude, Longitude) it can be downloaded here: [HEG](https://newsroom.gsfc.nasa.gov/sdptoolkit/HEG/HEGHome.html)

## More Resources:

Weather Data: https://www.ncdc.noaa.gov/ghcn-daily-description

Satellite Data: https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13a2_v006

QGIS: https://www.qgis.org/en/site/

MODIS Tile info : https://modis-land.gsfc.nasa.gov/MODLAND_grid.html
