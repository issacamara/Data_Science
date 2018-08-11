
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 2 - Pandas Introduction
# All questions are weighted the same in this assignment.
# ## Part 1
# The following code loads the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on [All Time Olympic Games Medals](https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table), and does some basic data cleaning. 
# 
# The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals, total # number of games, total # of medals. Use this dataset to answer the questions below.

# In[8]:

import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()
df


# ### Question 0 (Example)
# 
# What is the first country in df?
# 
# *This function should return a Series.*

# In[2]:

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the row for Afghanistan, which is a Series object. The assignment
    # question description will tell you the general format the autograder is expecting
    return df.iloc[0]

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 


# ### Question 1
# Which country has won the most gold medals in summer games?
# 
# *This function should return a single string value.*

# In[3]:

import numpy as np
def answer_one():
    max = np.max(df["Total"])
    return str(df[df['Total'] == max].index[0])

answer_one()


# ### Question 2
# Which country had the biggest difference between their summer and winter gold medal counts?
# 
# *This function should return a single string value.*

# In[4]:

import numpy as np
def answer_two():
    df["Delta Summer-Winter"] = np.absolute(df["Total"] - df["Total.1"])
    max = np.max(df["Delta Summer-Winter"])
    return str(df[df['Delta Summer-Winter'] == max].index[0])

answer_two()


# ### Question 3
# Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count? 
# 
# $$\frac{Summer~Gold - Winter~Gold}{Total~Gold}$$
# 
# Only include countries that have won at least 1 gold in both summer and winter.
# 
# *This function should return a single string value.*

# In[11]:

import numpy as np
def answer_three():
    df["Relative Diff Gold"] = None
    df["Relative Diff Gold"][df["Gold.2"] > 0] = df["Gold"] - df["Gold.1"] / df["Gold.2"]
    #print(df["Relative Diff Gold"])
    max = np.max(df["Relative Diff Gold"])
    return str(df[df['Relative Diff Gold'] == max].index[0])

answer_three()


# ### Question 4
# Write a function that creates a Series called "Points" which is a weighted value where each gold medal (`Gold.2`) counts for 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point. The function should return only the column (a Series object) which you created, with the country names as indices.
# 
# *This function should return a Series named `Points` of length 146*

# In[7]:

import numpy as np
import pandas as pd
def answer_four():
    Points = pd.Series([3, 2, 1], index=['Gold.2', 'Silver.2', 'Bronze.2'])
    df["Points"] = df['Gold.2']*Points['Gold.2'] + df['Silver.2']*Points['Silver.2'] + df['Bronze.2']*Points['Bronze.2']
    return df["Points"]
answer_four()


# ## Part 2
# For the next set of questions, we will be using census data from the [United States Census Bureau](http://www.census.gov). Counties are political and geographic subdivisions of states in the United States. This dataset contains population data for counties and states in the US from 2010 to 2015. [See this document](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2015/co-est2015-alldata.pdf) for a description of the variable names.
# 
# The census dataset (census.csv) should be loaded as census_df. Answer questions using this as appropriate.
# 
# ### Question 5
# Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need this for future questions too...)
# 
# *This function should return a single string value.*

# In[12]:

census_df = pd.read_csv('census.csv')
census_df.head()


# In[13]:

def answer_five():
    counties = census_df[census_df['SUMLEV'] == 50]
    states = counties['STNAME'].unique()
    max = 0
    state = 'Alabama'
    for s in states:
        count = len(counties[counties['STNAME']==s].index)
        if max < count:
            max = count
            state = s
    return state

answer_five()


# ### Question 6
# **Only looking at the three most populous counties for each state**, what are the three most populous states (in order of highest population to lowest population)? Use `CENSUS2010POP`.
# 
# *This function should return a list of string values.*

# In[83]:

def answer_six():
    counties = census_df[census_df['SUMLEV'] == 50]
    counties = counties.sort_values(by=['STNAME','CENSUS2010POP'], ascending=False)
    #print(counties[['STNAME','CENSUS2010POP']])
    dictionary = {}
    for state in (counties['STNAME'].unique()):
        #print(state)
        #print(counties["CENSUS2010POP"][counties["STNAME"] == state].head(n=3).sum())
        #print("-------------------")
        dictionary[state] = counties["CENSUS2010POP"][counties["STNAME"] == state].head(n=3).sum()
    
    series = pd.Series(dictionary)
    data = pd.DataFrame(series)
    data.columns = ['Sum']
    data = data.sort_values(by='Sum', ascending=False)
    #print(data.head()) 
    #print(series)
    #counties = counties.sort_values(by='CENSUS2010POP', ascending=False)
    #returnVal = list(counties['STNAME'].head(n=3))  
    returnVal = list(data.head(n=3).index)
    
    return returnVal

answer_six()


# ### Question 7
# Which county has had the largest absolute change in population within the period 2010-2015? (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)
# 
# e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, then its largest change in the period would be |130-80| = 50.
# 
# *This function should return a single string value.*

# In[43]:

def answer_seven():
    counties = census_df[census_df['SUMLEV'] == 50]
    min_val = map(min,counties['POPESTIMATE2010'],counties['POPESTIMATE2011'],counties['POPESTIMATE2012'],counties['POPESTIMATE2013'],counties['POPESTIMATE2014'],counties['POPESTIMATE2015'])
    max_val = map(max,counties['POPESTIMATE2010'],counties['POPESTIMATE2011'],counties['POPESTIMATE2012'],counties['POPESTIMATE2013'],counties['POPESTIMATE2014'],counties['POPESTIMATE2015'])
    
    min_val_series = pd.Series(min_val, index = counties.index)
    max_val_series = pd.Series(max_val, index = counties.index)
    #print(min_val_series-max_val_series)
    counties["Diff 2010-2015"] = np.absolute(max_val_series - min_val_series)
    #print(counties["Diff 2010-2015"].head())
    
    counties = counties.sort_values(by='Diff 2010-2015', ascending=False)
    return str(counties["CTYNAME"].head(n=1).values[0])

answer_seven()


# ### Question 8
# In this datafile, the United States is broken up into four regions using the "REGION" column. 
# 
# Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
# 
# *This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] and the same index ID as the census_df (sorted ascending by index).*

# In[36]:

def answer_eight():
    counties = census_df[census_df['SUMLEV'] == 50]
    
    return census_df[((census_df['REGION']==1) | (census_df['REGION']==2)) & (census_df['POPESTIMATE2015'] > census_df['POPESTIMATE2014']) & (census_df['CTYNAME'].str.startswith('Washington')) ][['STNAME', 'CTYNAME']]

answer_eight()


# In[ ]:



