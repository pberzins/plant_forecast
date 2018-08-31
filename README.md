# NDVI forecaster from Historical Weather data and MODIS satellite

## Business Understanding
This project attempts to use remote sensing techniques to predict the future vegetatative health of an area in respect to changing weather. Farmers, Gardeners, and Fire Management officials would be able to use the local weather forecast and predict how a specified region's vegetation will change. 

## Data Understanding 
How does one measure Vegetative Health?:

**NDVI:** NDVI which stands for Normalized Differenced Vegetation Index is a measure of the amount of "green" in a substance using infrared wavelengths. More information can be found here: https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index

**Photosynthesis:** Photosynthesis is the production of glucose (sugar) and water from Carbon Dioxide and and Water. More information can be found here: https://en.wikipedia.org/wiki/Photosynthesis


## Data Preparation
__This project requires GDAL!__
GDAL can be very difficult to install if python packages are not organized in a particular way. In the case of this project, I had to uninstall and reinstall my Anaconda Environment to get it to work. This is no small task, good luck! 

## Satellite Data:
Satellite Data is obtained with the PyModis Module which allows you to bulk download MODIS tiles and products available from NASA. For download and documentation of PyModis: http://www.pymodis.org/

The Naming System of MODIS Products will be described below: 

**Example Filename:**

### MOD13A2.A2006001.h08v05.006.2015113045801.hdf 

**MOD13A2** : Product Short Name

**A2006001**: Year and Julian Date of acquistion (i.e the 1st Julian day of 2006) 

**h08v05**: The tile in which to download. 

**006** : Collection Version

**2015113045801**: Julian Date of Production

**hdf** : Data Format 

more information on this specific product can be found here: https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13a2_v006

### Converting .hdf files into .tiff files 
hdf files are very cumbersome to work with, especially since their projection is Sinusudial. Many people are more accustomed to thinking in terms of latitude and longitude (Geographical). For this project I used a an application called: HDF-EOS To GeoTIFF Conversion Tool (HEG). HEG is very useful for converting HDF's to Geographic projections. Can be downloaded here: https://newsroom.gsfc.nasa.gov/sdptoolkit/HEG/HEGHome.html

## Weather Data:




## Modeling

## Evaluation 


Weather Data: https://www.ncdc.noaa.gov/ghcn-daily-description

Satellite Data: https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13a2_v006

QGIS: https://www.qgis.org/en/site/
