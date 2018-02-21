#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 19:00:58 2018

@author: IssaCamara
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


def count_all_items(list_of_items):
    dictionary = {}
    for item in list_of_items:
        if(dictionary.get(item) is None):
            dictionary[item] = 0
        else:
            dictionary[item] = dictionary.get(item) + 1
    
    return dictionary

train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)
#gender_submission = pd.read_csv('gender_submission.csv')

# Analyzing data by plotting some charts
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

dico = count_all_items(train["Pclass"])
plt.bar(list(dico.keys()), list(dico.values()))
plt.show()

table = pd.crosstab(train.Survived,train.Pclass)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)


#for col in train.columns:
#    if col not in ['Survived','PassengerId','Name','Age'] :
#        table = pd.crosstab(train['Survived'],train[col])
#        table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.Survived,train.Embarked)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    
# Cleaning data
train.drop(['Cabin'],1)
train.dropna(inplace=True)
train.isnull().sum()
test.drop(['Cabin'],1)
test.dropna(inplace=True)
test.isnull().sum()
#Creating dummy variables
dummy_var = [ 'Pclass', 'Sex']
for i in dummy_var:
    dummy = pd.get_dummies(train[i],prefix = i)
    train = train.join(dummy)
    test = test.join(dummy)
  
test = test.fillna(0)

data_vars=train.columns.values.tolist()
to_keep=[i for i in data_vars if i not in dummy_var]
train=train[to_keep]

data_vars=test.columns.values.tolist()
to_keep=[i for i in data_vars if i not in dummy_var]
test=test[to_keep]

predictors = ['Age', 'SibSp', 'Parch', 'Pclass_1', 'Pclass_2', 'Pclass_3',
       'Sex_female', 'Sex_male']

all_X = train[predictors]
all_y = train['Survived']
classifier = LogisticRegression(random_state=0)
classifier.fit(all_X, all_y)

y_pred = classifier.predict(all_X)
confusion_matrix = confusion_matrix(all_y, y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(all_X, all_y)))
print(classification_report(all_y, y_pred))


train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, 
                                                    test_size=0.20,random_state=0)

classifier.fit(train_X, train_y)
predictions = classifier.predict(test_X)
accuracy = accuracy_score(test_y, predictions)
print("The accuracy is ",accuracy)

scores = cross_val_score(classifier, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()

print("The scores are ",scores)
print("The accuracy is ",accuracy)
