
# coding: utf-8

# # Assignment 4
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# This assignment requires that you to find **at least** two datasets on the web which are related, and that you visualize these datasets to answer a question with the broad topic of **economic activity or measures** (see below) for the region of **Colombes, Hauts-de-Seine, France**, or **France** more broadly.
# 
# You can merge these datasets with data from different regions if you like! For instance, you might want to compare **Colombes, Hauts-de-Seine, France** to Ann Arbor, USA. In that case at least one source file must be about **Colombes, Hauts-de-Seine, France**.
# 
# You are welcome to choose datasets at your discretion, but keep in mind **they will be shared with your peers**, so choose appropriate datasets. Sensitive, confidential, illicit, and proprietary materials are not good choices for datasets for this assignment. You are welcome to upload datasets of your own as well, and link to them using a third party repository such as github, bitbucket, pastebin, etc. Please be aware of the Coursera terms of service with respect to intellectual property.
# 
# Also, you are welcome to preserve data in its original language, but for the purposes of grading you should provide english translations. You are welcome to provide multiple visuals in different languages if you would like!
# 
# As this assignment is for the whole course, you must incorporate principles discussed in the first week, such as having as high data-ink ratio (Tufte) and aligning with Cairo’s principles of truth, beauty, function, and insight.
# 
# Here are the assignment instructions:
# 
#  * State the region and the domain category that your data sets are about (e.g., **Colombes, Hauts-de-Seine, France** and **economic activity or measures**).
#  * You must state a question about the domain category and region that you identified as being interesting.
#  * You must provide at least two links to available datasets. These could be links to files such as CSV or Excel files, or links to websites which might have data in tabular form, such as Wikipedia pages.
#  * You must upload an image which addresses the research question you stated. In addition to addressing the question, this visual should follow Cairo's principles of truthfulness, functionality, beauty, and insightfulness.
#  * You must contribute a short (1-2 paragraph) written justification of how your visualization addresses your stated research question.
# 
# What do we mean by **economic activity or measures**?  For this category you might look at the inputs or outputs to the given economy, or major changes in the economy compared to other regions.
# 
# ## Tips
# * Wikipedia is an excellent source of data, and I strongly encourage you to explore it for new data sources.
# * Many governments run open data initiatives at the city, region, and country levels, and these are wonderful resources for localized data sources.
# * Several international agencies, such as the [United Nations](http://data.un.org/), the [World Bank](http://data.worldbank.org/), the [Global Open Data Index](http://index.okfn.org/place/) are other great places to look for data.
# * This assignment requires you to convert and clean datafiles. Check out the discussion forums for tips on how to do this from various sources, and share your successes with your fellow students!
# 
# ## Example
# Looking for an example? Here's what our course assistant put together for the **Ann Arbor, MI, USA** area using **sports and athletics** as the topic. [Example Solution File](./readonly/Assignment4_example.pdf)

# In[13]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().magic('matplotlib notebook')

seasons = ['20052006','20062007','20072008',
           '20082009','20092010','20102011','20112012','20122013',
           '20132014','20142015','20152016','20162017', '20172018' ]

categories = pd.Series(seasons).astype("category", categories=seasons, ordered=True)

# link = http://www.football-data.co.uk/englandm.php
epl = []

big6 = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 'Tottenham']
columnsToKeep = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
for s in categories:
    data = pd.read_csv('epl{}.csv'.format(s))
    data = data[columnsToKeep]
    data['Season'] = s
    data = data.replace('Man United', 'Manchester United')
    data = data.replace('Man City', 'Manchester City')
    # Keep all the matches that involves exactly one big6 member
    rowsToKeep = (data['HomeTeam'].isin(big6) & (~ data['AwayTeam'].isin(big6))) | (~ data['HomeTeam'].isin(big6) & (data['AwayTeam'].isin(big6)))
    data = data[rowsToKeep]
    data['FTHG'] = data['FTHG'].astype('int64')
    data['FTAG'] = data['FTAG'].astype('int64')
    #print(data[ ~((data['HomeTeam'].isin(big6)) & (data['AwayTeam'].isin(big6)))].head())
    #print(data[ (~(data['HomeTeam'].isin(big6)) & (~(data['AwayTeam'].isin(big6))))].head())
    epl = epl + [data]

eplDf = pd.concat(epl)
#eplDf = eplDf.set_index(['Season', 'HomeTeam', 'AwayTeam'])

def computeResult(row):
    if (row['HomeTeam'] in (big6)):
        if row['FTHG'] < row['FTAG']:
            row['Result'] = 'L'
        elif row['FTHG'] == row['FTAG']:
            row['Result'] = 'D'
        else :
            row['Result'] = 'W'
        row['Team'] = row['HomeTeam']
    else:
        if row['FTHG'] < row['FTAG']:
            row['Result'] = 'W'
        elif row['FTHG'] == row['FTAG']:
            row['Result'] = 'D'
        else :
            row['Result'] = 'L'
        row['Team'] = row['AwayTeam']
    
    return row
            

eplDf = eplDf.apply(computeResult, axis = 1)

nbMatchesPerSeason = len(eplDf[(eplDf['Team'] == 'Chelsea') & (eplDf['Season'] == '20082009')])

listDf = []

for team in big6:
    tmp = eplDf[(eplDf['Team'] == team) & (eplDf['Result'] == 'W')]
    series = (tmp.groupby(['Season', 'Result'])['Team'].count() / nbMatchesPerSeason)
    df = pd.DataFrame(series).reset_index().rename(columns={'Team': 'WinRatio'}).drop('Result', axis = 1)
    df['Team'] = team
    listDf = listDf + [df]

dfWithWinRatio = pd.concat(listDf)
fig, ax = plt.subplots(figsize=(9,6))

for label, df in dfWithWinRatio.groupby('Team'):
    df['WinRatio'].plot(x=categories, kind="line", ax=ax, label=label, ylim=(0,1))
plt.legend()

plt.axhline(y=0.5, color='r', linestyle='dashed', label='fgkhjkj')

plt.xticks(rotation=45)
ax.set_xticklabels(seasons, fontsize=5)
ax.set_xlabel('Seasons')
ax.set_ylabel('% of matches won per season')
fig.suptitle('Win ratio for Big 6 teams since 2005 against the rest of the league', fontsize=10)
#eplDf.pivot(index = 'Season', columns = 'HomeTeam', values = 'Result')
#print(eplDf.head(10))


# In[ ]:



