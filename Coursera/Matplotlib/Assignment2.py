
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d100/1b10750994e98f9fe5828966bf365e6f39293ca7b239de2bfadfaa56.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **Colombes, Hauts-de-Seine, France**, and the stations the data comes from are shown on the map below.

# In[10]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import datetime
import numpy as np
get_ipython().magic('matplotlib notebook')

def plotTemperatures(binsize, hashid):
    temperatures = pd.read_csv('data/C2A2_data/BinnedCsvs_d{}/{}.csv'.format(binsize, hashid))
    
    temperatures = (temperatures[temperatures['Date'] !='2010-11-06'])
    temperatures['Date'] = pd.to_datetime(temperatures['Date'])
    #temperatures[temperatures['Date'].)
    
    Feb29_mask = temperatures['Date'].map(lambda x: (x.day, x.month)) == (29,2)
    temperatures = temperatures[~ Feb29_mask]
    
    #temperatures['DateIndex'] = pd.DatetimeIndex(temperatures['Date']).dayofyear
    #temperatures['DateIndex'] = temperatures['DateIndex'].map(lambda x: pd.Timedelta(days=x)) 
    
    temperatures['Data_Value'] = temperatures['Data_Value'] * 0.1
    date_min = datetime.date(2005, 1, 1)
    date_max = datetime.date(2014, 12, 31)
    date_2015_begin = datetime.date(2015, 1, 1)
    date_2015_end = datetime.date(2015, 12, 31)
    df_2005_2014 = temperatures[(temperatures['Date'] >= date_min) & (temperatures['Date'] <= date_max)]
    df_2005_2014['Date'] = df_2005_2014['Date'].map(lambda x: x.replace(year=2015))
    
    df_2015 = temperatures[(temperatures['Date'] >= date_2015_begin) & (temperatures['Date'] <= date_2015_end)]
    dates = [d for d in df_2015['Date']]
    
    max_temp_by_date = df_2005_2014[df_2005_2014['Element']=='TMAX'].groupby(['Date']).max()
    min_temp_by_date = df_2005_2014[df_2005_2014['Element']=='TMIN'].groupby(['Date']).min() 
    #print((max_temp_by_date['Data_Value'].resample('DY').max()))
    #print(df_2005_2014.groupby(by=[df_2005_2014['Date'].year]))  
    
    max_2015 = df_2015[df_2015['Element']=='TMAX'].groupby(['Date']).max()
    max_2015 = max_2015[max_2015['Data_Value'] > max_temp_by_date['Data_Value']]
    dates_max = [d for d in max_2015.index]
    
    min_2015 = df_2015[df_2015['Element']=='TMIN'].groupby(['Date']).min()
    min_2015 = min_2015[min_2015['Data_Value'] < min_temp_by_date['Data_Value']]
    dates_min = [d for d in min_2015.index]
    
    
    y2003 = datetime.date(2014, 12, 1)
    y2017 = datetime.date(2016, 1, 31)

    fig, ax = plt.subplots(figsize=(10,6))
    max_temp_by_date.plot(y="Data_Value", kind='line', ax=ax, color='#ff6347', label='TMAX')
    min_temp_by_date.plot(y="Data_Value", kind='line', ax=ax, xlim=(y2003,y2017), color='#1e90ff', label ='TMIN')
    #df_2015.plot(kind='scatter', x='Date', y="Data_Value", ax=ax, color='black', alpha=0.5, xlim=(y2003,y2017))
    ax.fill_between(min_temp_by_date.index, min_temp_by_date['Data_Value'], max_temp_by_date['Data_Value'], 
                    color='#d3d3d3')
    ax.scatter(dates_max, max_2015['Data_Value'], s = 30, color = '#ff0000', label='Record high in 2015')
    ax.scatter(dates_min, min_2015['Data_Value'], s = 30, color = '#0000ff', label='Record low in 2015')
    ax.set_ylabel('Temperature in °C')
    fig.suptitle('(high and low) temperatures of 2005-2014 and broken records in 2015', fontsize=15)
    plt.legend()
plotTemperatures(100,'1b10750994e98f9fe5828966bf365e6f39293ca7b239de2bfadfaa56')



def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))
    station_locations_by_hash = df[df['hash'] == hashid]

    
    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

#leaflet_plot_stations(100,'1b10750994e98f9fe5828966bf365e6f39293ca7b239de2bfadfaa56')



# In[ ]:



