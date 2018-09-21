
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[8]:

import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df = pd.DataFrame(data=df, columns=['text'])
#df.str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')


# In[37]:

def date_sorter():
    import re
    import math
    from datetime import date
    dictionary = {'January': '1', 'February':'2', 'March': '3', 'April': '4', 'May': '5', 'June': '6',
                  'July': '7', 'August':'8', 'September': '9', 'October': '10', 'November': '11', 'December': '12',
                  'Jan': '1', 'Feb':'2', 'Mar': '3', 'Apr': '4', 'May': '5', 'Jun': '6',
                  'Jul': '7', 'Aug':'8', 'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

    months = '(?P<Month>January|February|March|April|May|June|July|August|September|October|November|December'
    months = months + '|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    numerical_order = '(st|nd|rd|th)'
    
    date1 = '(?P<Month>\d?\d)[/-](?P<Day>\d?\d)[/-](?P<Year>\d?\d?\d\d)' 
    date2 = months  + '[\.\s-]+(?P<Day>\d?\d)[,\s-]+' + '(?P<Year>\d\d\d\d)'
    date3 = months + ' (?P<Day>\d?\d)'+ numerical_order +', (?P<Year>\d\d\d\d)' 
    date4 = '(?P<Day>\d?\d?)(\s?)' + months + '[,\s]+(?P<Year>\d\d\d\d)' 
    date5 = '(?P<Day>)(?P<Month>\d?\d)/(?P<Year>\d\d\d\d)' 
    date6 = '(?P<Day>\d?\d)/(?P<Month>\d?\d)/(?P<Year>\d\d\d\d)' 
    date7 = '(?P<Day>)(?P<Month>)(?P<Year>\d\d\d\d)'
    #date7 = '(?P<Day>\d?\d?)(\s)?(?P<Month>\d?\d?)(\s?)(?P<Year>\d\d\d\d)'
    
    date_regex = [date1, date2, date3, date4, date5, date6, date7]
    
    def extractDate(row):
        i = 0
        matched = False
        while ( (i < len(date_regex)) & (not matched)):
            date = re.search(date_regex[i], row['text'])
            if (date != None):
                row['Day'] = date.group('Day')
                row['Month'] = date.group('Month')
                row['Year'] = date.group('Year')
                matched = True
            i = i + 1
        return row
    
    dates_df = df.apply(extractDate, axis=1)#[['Day','Month','Year']]
    
    def clean(row):
        if len(row['Day'])==0:
            row['Day'] = '1'
        if len(row['Month'])==0:
            row['Month'] = '1'
        if len(row['Month'])>=3:
            row['Month'] = dictionary[row['Month']]
        if len(row['Year'])==2:
            row['Year'] = '19' + row['Year']
        
        row['Date'] = date(int(row['Year']), int(row['Month']), int(row['Day']))
        return row

    dates_df = dates_df.apply(clean, axis=1)
    dates_df = dates_df.sort_values(['Date'], ascending=True)
    #for i in dates_df.index:
        #print(dates_df.loc[i,'Month'],'-------',dates_df.loc[i,'text'])
    
    
    '''
    print(re.findall(date1, '04/20/2009; 04/20/09; 4/20/09; 4/3/09; 4-3-09'))
    print(re.findall(date2, 'March 20, 2009; Mar-20-2009; Mar 20, 2009;  Mar. 20, 2009; Mar 20 2009;'))
    print(re.findall(date3, 'Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009'))
    print(re.findall(date4, 'Feb 2009; Sep 2009; Oct 2010'))
    print(re.findall(date5, '12/2008 12/1975 1/1978 06/1973 8/2009'))
    print(re.findall(date6, 'aS/P suicide attempt 2011 Hx of Outpatient Treatment: Yes'))
    '''
    return pd.Series(dates_df.index)

#date_sorter()


# In[ ]:




# In[ ]:



