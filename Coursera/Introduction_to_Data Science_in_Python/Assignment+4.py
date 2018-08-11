
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[1]:

import pandas as pd
import numpy as np
import math
from scipy.stats import ttest_ind
import re
import time


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[2]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[3]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    f = open('university_towns.txt', "r")
    lines = list(f)
    
    series = []
    i = 0
    state = ''
    region = ''
    
    while (i < len(lines)):
        line = lines[i].split('[ed')
        
        if (len(line) == 2):
            state = line[0]
        else:
            line = lines[i].split(' (')
            region = line[0].replace('\n','')
            series = series + [[state, region]]
        i = i + 1
    f.close()
    
    #for i in series:
        #print(i)
    data = pd.DataFrame(series, columns=["State", "RegionName"])
    #for i in data['RegionName']:
        #print(+ i + '1')
    return pd.DataFrame(series, columns=["State", "RegionName"])

get_list_of_university_towns()


# In[4]:

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    
    df = pd.read_excel('gdplev.xls', skiprows=4,header = 1)
    quarterly = df[['Unnamed: 4','GDP in billions of current dollars.1','GDP in billions of chained 2009 dollars.1']]
    quarterly = quarterly.drop([0,1]).reset_index().drop(["index"], axis = 1)
    quarterly.columns = ['Quarter','GDP in billions of current dollars','GDP in billions of chained 2009 dollars']
    #quarterly = quarterly[str(quarterly['Quarter']).startswith('20')]
    quarterly['Quarter'] = quarterly['Quarter'].astype(str)
    quarterly = quarterly[list(map(lambda x: x.startswith('20'), quarterly['Quarter']))]
    #print(quarterly)
    for i in quarterly.index[1:]:
        gdp_i = quarterly.loc[i, 'GDP in billions of chained 2009 dollars']
        gdp_i1 = quarterly.loc[i-1, 'GDP in billions of chained 2009 dollars']
        quarterly.loc[i, 'Growth'] = (gdp_i - gdp_i1) / gdp_i1
    #print(quarterly[quarterly['Growth'] < 0][['Quarter','Growth']])
    
    recession = False
    recessionYear = np.NaN
    indices = list(quarterly.index[1:])
    i = 0
    while (i < len(indices)) & (not recession):
        ind = indices[i]
        if((quarterly.loc[ind, 'Growth'] < 0) & (quarterly.loc[ind - 1, 'Growth'] < 0)):
            recession = True
            recessionYear = quarterly.loc[ind - 1, 'Quarter']
        i = i + 1
        
    
    return recessionYear

get_recession_start()


# In[6]:

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    
    
    df = pd.read_excel('gdplev.xls', skiprows=4,header = 1)
    quarterly = df[['Unnamed: 4','GDP in billions of current dollars.1','GDP in billions of chained 2009 dollars.1']]
    quarterly = quarterly.drop([0,1]).reset_index().drop(["index"], axis = 1)
    quarterly.columns = ['Quarter','GDP in billions of current dollars','GDP in billions of chained 2009 dollars']
    #quarterly = quarterly[str(quarterly['Quarter']).startswith('20')]
    quarterly['Quarter'] = quarterly['Quarter'].astype(str)
    quarterly = quarterly[list(map(lambda x: x.startswith('20'), quarterly['Quarter']))]
    
    for i in quarterly.index[1:]:
        gdp_i = quarterly.loc[i, 'GDP in billions of chained 2009 dollars']
        gdp_i1 = quarterly.loc[i-1, 'GDP in billions of chained 2009 dollars']
        quarterly.loc[i, 'Growth'] = (gdp_i - gdp_i1) / gdp_i1

    recessionYear = get_recession_start()
    
    quarterly = quarterly[quarterly['Quarter'] > recessionYear]
    
    growth = False
    growthYear = np.NaN
    indices = list(quarterly.index[1:])
    i = 0
    while (i < len(indices)) & (not growth):
        ind = indices[i]
        if((quarterly.loc[ind, 'Growth'] >= 0) & (quarterly.loc[ind - 1, 'Growth'] >= 0)):
            growth = True
            growthYear = quarterly.loc[ind, 'Quarter']
        i = i + 1
        

    return growthYear
    
get_recession_end()


# In[7]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    
    df = pd.read_excel('gdplev.xls', skiprows=4,header = 1)
    quarterly = df[['Unnamed: 4','GDP in billions of current dollars.1','GDP in billions of chained 2009 dollars.1']]
    quarterly = quarterly.drop([0,1]).reset_index().drop(["index"], axis = 1)
    quarterly.columns = ['Quarter','GDP in billions of current dollars','GDP in billions of chained 2009 dollars']
    #quarterly = quarterly[str(quarterly['Quarter']).startswith('20')]
    quarterly['Quarter'] = quarterly['Quarter'].astype(str)
    quarterly = quarterly[list(map(lambda x: x.startswith('20'), quarterly['Quarter']))]
    
    recessionYear = get_recession_start()
    growthYear = get_recession_end()
    quarterly = quarterly[(quarterly['Quarter'] >= recessionYear) & (quarterly['Quarter'] < growthYear)]
    quarterly = quarterly.sort_values(by = 'GDP in billions of chained 2009 dollars', ascending = True)
    return quarterly.loc[quarterly.index[0],'Quarter']

get_recession_bottom()


# In[8]:

def computeMeanForQuarters(data):
    start = [i for i,x in enumerate(data.columns) if x == '2000-01'][0]
    end = [i for i,x in enumerate(data.columns) if x == '2016-08'][0]
    
    months =  data.columns[start:end + 1]
    quarter = []
    for value in months:
        quarter = quarter + [value]
        if(len(quarter) == 3):
            year = str(quarter[2]).split('-')[0]
            month = str(quarter[2]).split('-')[1]
            if(month == '03'):
                data[year + 'q1'] = data[quarter].mean(axis = 1)
            elif(month == '06'):
                data[year + 'q2'] = data[quarter].mean(axis = 1)
            elif(month == '09'):
                data[year + 'q3'] = data[quarter].mean(axis = 1)
            elif(month == '12'):
                data[year + 'q4'] = data[quarter].mean(axis = 1)

            quarter = []
    
    if len(quarter) > 0:
        year = quarter[len(quarter) - 1].split('-')[0]
        month = quarter[len(quarter) - 1].split('-')[1]
        if(month <= '03'):
            data[year + 'q1'] = data[quarter].mean(axis = 1)
        elif(month <= '06'):
            data[year + 'q2'] = data[quarter].mean(axis = 1)
        elif(month <= '09'):
            data[year + 'q3'] = data[quarter].mean(axis = 1)
        elif(month <= '12'):
            data[year + 'q4'] = data[quarter].mean(axis = 1)

    return data

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    df = pd.read_csv('City_Zhvi_AllHomes.csv')
    start90s = [i for i,x in enumerate(df.columns) if x == '1996-04'][0]
    start = [i for i,x in enumerate(df.columns) if x == '2000-01'][0]
    
    
    df = df.fillna(np.NaN)
    
    
    colToDel = df.columns[start90s:start]
    df = df.drop(list(colToDel) + ['RegionID', 'Metro', 'CountyName', 'SizeRank'], axis=1)
    months =  df.columns[2:]
    
    df = computeMeanForQuarters(df)
    df = df.drop(list(months), axis=1)
    df['State'] = df['State'].apply(lambda value: states[value])
    df = df.set_index(['State','RegionName'])
    return df

convert_housing_data_to_quarters()


# In[37]:

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    startYear = get_recession_start()
    endYear = get_recession_bottom()
    hd = convert_housing_data_to_quarters()
    start = [i for i,x in enumerate(hd.columns) if x == startYear][0]
    end = [i for i,x in enumerate(hd.columns) if x == endYear][0]
    
    
    hd = hd.iloc[ : ,(start-1):(end+1)]

    
    price1 = hd[startYear]
    price2 = hd[endYear]

    hd['Growth'] = (price2 - price1) / price1
    hd['Price_Ratio'] = hd.iloc[:, 0] / hd.iloc[:, -1]
    
    
    
    list_univ = get_list_of_university_towns()
    univ_town_hd = pd.merge(hd, list_univ, how='inner', right_on=['State','RegionName'],
         left_index=True,suffixes=('_x', '_y')).set_index(['State','RegionName'])
    
    df = pd.merge(hd, list_univ, how='left', right_on=['State','RegionName'],
         left_index=True,suffixes=('_x', '_y')).set_index(['State','RegionName'])
    
    non_univ_town_hd = df[~df.index.isin(univ_town_hd.index)]
    
    (s,p) = ttest_ind(univ_town_hd[~univ_town_hd['Growth'].isnull()]['Growth'], non_univ_town_hd[~non_univ_town_hd['Growth'].isnull()]['Growth'])
    
    avg1 = univ_town_hd['Price_Ratio'].mean()
    avg2 = non_univ_town_hd['Price_Ratio'].mean()

    better = None
    if(avg1 < avg2):
        better = 'university town'
    else:
        better = 'non-university town'
    return (p < 0.1, p, better)
run_ttest()


# In[ ]:



