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
from math import ceil
from sklearn import svm, tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier #For Classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler  

def count_all_items(list_of_items):
    dictionary = {}
    for item in list_of_items:
        if(dictionary.get(item) is None):
            dictionary[item] = 0
        else:
            dictionary[item] = dictionary.get(item) + 1
    
    return dictionary

def submit(filename, model):
    submission_df = {"PassengerId": test['PassengerId'],
                 "Survived": model.predict(test[predictors])}
    submission = pd.DataFrame(submission_df)
    submission.to_csv(filename,index=False)


train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)

full_data = [train, test]
#gender_submission = pd.read_csv('gender_submission.csv')

for dataset in full_data: 
    mean_age = np.mean(dataset.Age)
    dataset.Age = dataset.Age.fillna(mean_age)
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp']
    dataset['Fare'] = dataset['Fare'].fillna(np.mean(dataset.Fare))
    
    


#for dataset in full_data:
#    print(dataset[dataset['Pclass']==1]['Fare'].describe())
#    print(dataset[dataset['Pclass']==2]['Fare'].describe())
#    print(dataset[dataset['Pclass']==3]['Fare'].describe())
#    print('_'*20)

#    size = 10
#    age_min = np.min(dataset.Age)
#    age_max = np.max(dataset.Age)
#    nbCategories = ceil((age_max - age_min)/size)
#    for i in range(0, nbCategories):
#        print(i," = ",[age_min,age_min+size])
#        dataset.loc[(dataset['Age'] >= age_min) & (dataset['Age'] < age_min+size), 'CategoricalAge'] = i
#        age_min = age_min+size
#    dataset.loc[ dataset['Age'] <= 15, 'CategoricalAge'] = 0
#    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 40), 'CategoricalAge'] = 1
#    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'CategoricalAge'] = 2
#    dataset.loc[ dataset['Age'] > 60, 'CategoricalAge'] = 3 ;



# Analyzing data by plotting some charts
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

survived["Fare"].plot.hist(alpha=0.5,color='red',bins=50)
died["Fare"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()



dico = count_all_items(train["Pclass"])
plt.bar(list(dico.keys()), list(dico.values()))
plt.show()

''' -------------------------------------------------------------------- '''

table = pd.crosstab(train.CategoricalAge,train.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.Survived,train.CategoricalAge)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

''' -------------------------------------------------------------------- '''

table = pd.crosstab(train.Survived,train.Pclass)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.Pclass,train.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

''' -------------------------------------------------------------------- '''

table = pd.crosstab(train.Survived,train.Embarked)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.Embarked,train.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

''' -------------------------------------------------------------------- '''

table = pd.crosstab(train.Survived,train.Sex)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.Sex,train.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

''' -------------------------------------------------------------------- '''

table = pd.crosstab(train.Survived,train.FamilySize)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.FamilySize,train.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

''' -------------------------------------------------------------------- '''
table = pd.crosstab(train.Survived,train.CategoricalFare)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.CategoricalFare,train.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

''' -------------------------------------------------------------------- '''


#for col in train.columns:
#    if col not in ['Survived','PassengerId','Name','Age'] :
#        table = pd.crosstab(train['Survived'],train[col])
#        table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

 # Mapping Age
for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 15, 'CategoricalAge'] = 0
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 40), 'CategoricalAge'] = 1
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'CategoricalAge'] = 2
    dataset.loc[ dataset['Age'] > 60, 'CategoricalAge'] = 3 
    dataset.CategoricalAge = dataset.CategoricalAge.astype(int)

 # Mapping FamilySize
for dataset in full_data:
    dataset.loc[ dataset['FamilySize'] == 0, 'CategoricalFamilySize'] = 0
    dataset.loc[(dataset['FamilySize'] > 0) & (dataset['FamilySize'] <= 3), 'CategoricalFamilySize'] = 1
    dataset.loc[(dataset['FamilySize'] > 3), 'CategoricalFamilySize'] = 2
    dataset.CategoricalFamilySize = dataset.CategoricalFamilySize.astype(int)


 # Mapping Fare
for dataset in full_data:
    dataset.loc[ dataset['Fare'] < 50, 'CategoricalFare'] = 0
    dataset.loc[ dataset['Fare'] >= 50, 'CategoricalFare'] = 1
    dataset.CategoricalFare = dataset.CategoricalFare.astype(int)


# Cleaning data
train.drop(['Cabin'],1)
#train.dropna(inplace=True)
train.isnull().sum()
test.drop(['Cabin'],1)
test.isnull().sum()
#Creating dummy variables
dummy_var = ['CategoricalAge', 'Pclass', 'Sex', 'Embarked', 'CategoricalFare', 'CategoricalFamilySize']
predictors = []
for i in dummy_var:
    dummy = pd.get_dummies(train[i],prefix = i)
    train = train.join(dummy)
    test = test.join(dummy)
    predictors = predictors + list(dummy)
  

data_vars=train.columns.values.tolist()
to_keep=[i for i in data_vars if i not in dummy_var]
train=train[to_keep]

data_vars=test.columns.values.tolist()
to_keep=[i for i in data_vars if i not in dummy_var]
test=test[to_keep]

#predictors = predictors + (list(['SibSp', 'Parch']))

all_X = train[predictors]
all_y = train['Survived']

################################## Logistic Regression ##################################
logreg = LogisticRegression(random_state=0)
logreg.fit(all_X, all_y)

rfe = RFE(logreg, 3)
rfe = rfe.fit(all_X, all_y)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
#y_pred = logreg.predict(all_X)
#conf_matrix = confusion_matrix(all_y, y_pred)
#print(conf_matrix)
#print('Accuracy of logistic regression logreg on test set: {:.2f}'.format(logreg.score(all_X, all_y)))
#print(classification_report(all_y, y_pred))

train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, 
                                                    test_size=0.20,random_state=0)

#logreg.fit(train_X, train_y)
#predictions = logreg.predict(test_X)
#accuracy = accuracy_score(test_y, predictions)
#print("The accuracy is ",accuracy)

kfold = KFold(n_splits=5)
scores = cross_val_score(logreg, all_X, all_y, cv=kfold)
accuracy = scores.mean()

print("The accuracy for Logistic Regression is ",accuracy)

submit("submission_logreg.csv",logreg)


################################## KNN ##################################
#/!\ Warning /!\ It's important to normalize the predictors for KNN modeling #
scaler = StandardScaler()  
scaler.fit(all_X)

#train_X = scaler.transform(train_X)  
#test_X = scaler.transform(test_X)

accuracy = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(knn, scaler.transform(all_X), all_y, cv=kfold)
    accuracy.append(scores.mean())

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), accuracy, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Accuracy')  

K = np.array(np.where(accuracy == np.max(accuracy))) + 1 # because the index starts at 0 
K = K[0,0]

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(scaler.transform(all_X),all_y) 
 

#y_pred = knn.predict(test_X)  
#print(confusion_matrix(test_y, y_pred))  
#print(classification_report(test_y, y_pred))  
#
#scores = cross_val_score(knn, all_X, all_y, cv=kfold)
#accuracy = scores.mean()
#
#print("The accuracy for KNN is ",accuracy)


submission_df = {"PassengerId": test['PassengerId'],
                 "Survived": knn.predict(scaler.transform(test[predictors]))}
submission = pd.DataFrame(submission_df)
submission.to_csv("submission_knn.csv",index=False)





################################# Classification Tree #################################

dtc = tree.DecisionTreeClassifier(criterion='gini') # for classification
dtc.fit(all_X, all_y)
dtc.score(all_X, all_y)
#Predict Output
y_pred = dtc.predict(test_X)
print(confusion_matrix(test_y, y_pred))  
print(classification_report(test_y, y_pred))  

scores = cross_val_score(dtc, all_X, all_y, cv=kfold)
accuracy = scores.mean()

print("The accuracy is ",accuracy)

submit("submission_dtc.csv",dtc)

################################# Random Forest #################################

#Predict Output
#y_pred = rfc.predict(test_X)
#print(confusion_matrix(test_y, y_pred))  
#print(classification_report(test_y, y_pred))  
#
#scores = cross_val_score(rfc, all_X, all_y, cv=kfold)
#accuracy = scores.mean()
#
#print("The accuracy is ",accuracy)
val = [1, 5, 10, 30, 50, 100, 500]
accuracy = []
for i in val:  
    rfc = RandomForestClassifier(n_estimators=i)
    scores = cross_val_score(rfc, all_X, all_y, cv=kfold)
    accuracy.append(scores.mean())

plt.figure(figsize=(12, 6))  
plt.plot(val, accuracy, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Variation of accuracy by the number of trees')  
plt.xlabel('Number of trees')  
plt.ylabel('Mean Accuracy')  

n = np.array(np.where(accuracy == np.max(accuracy))) + 1 # because the index starts at 0 
n = n[0,0]
n


rfc = RandomForestClassifier(n_estimators=6) # after cross validation, 4 was the best 
# Train the model using the training sets and check score
rfc.fit(all_X, all_y)

submit("submission_rfc.csv",rfc)



######################################### GBM #########################################

accuracy = []
val = [1, 5, 10, 30, 50, 100]
for i in val:  
    gbm = GradientBoostingClassifier(n_estimators=i, learning_rate=0.3, max_depth=2)
    scores = cross_val_score(gbm, all_X, all_y, cv=kfold)
    accuracy.append(scores.mean())

plt.figure(figsize=(12, 6))  
plt.plot(val, accuracy, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Variation of accuracy by the number of trees')  
plt.xlabel('Number of trees')  
plt.ylabel('Mean Accuracy')  

n = np.array(np.where(accuracy == np.max(accuracy))) + 1 # because the index starts at 0 
n = n[0,0]
n

gbm = GradientBoostingClassifier(n_estimators=12, learning_rate=0.3, max_depth=2)
# Train the model using the training sets and check score
gbm.fit(all_X, all_y)
#Predict Output
y_pred = gbm.predict(test_X)
print(confusion_matrix(test_y, y_pred))  
print(classification_report(test_y, y_pred))  

scores = cross_val_score(gbm, all_X, all_y, cv=kfold)
#scores.sort()
accuracy = scores.mean()

print("The accuracy for GBM is ",accuracy)


submit("submission_gbm.csv",gbm)


######################################### SVM #########################################


val = [1, 5, 10, 100]
val = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 1]
accuracy = []
for i in val:  
    model = svm.SVC(kernel='rbf', C=5, gamma=i) 
    scores = cross_val_score(model, all_X, all_y, cv=kfold)
    accuracy.append(scores.mean())

plt.figure(figsize=(12, 6))  
plt.plot(val, accuracy, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Variation of accuracy in function of C or gamma')  
plt.xlabel('Number of trees')  
plt.ylabel('Mean of Accuracy')  

n = np.array(np.where(accuracy == np.max(accuracy))) + 1 # because the index starts at 0 
n = n[0,0]
n

model = svm.SVC(kernel='rbf', C=5, gamma=0.1) 
model.fit(all_X, all_y)




submit("submission_svm.csv",model)


#accuracy = []
#for i in range(1, 50):  
#    gbm = GradientBoostingClassifier(n_estimators=i, learning_rate=1.0, max_depth=1)
#    scores = cross_val_score(gbm, all_X, all_y, cv=kfold)
#    accuracy.append(scores.mean())
#
#plt.figure(figsize=(12, 6))  
#plt.plot(range(1, 50), accuracy, color='red', linestyle='dashed', marker='o',  
#         markerfacecolor='blue', markersize=10)
#plt.title('Variation of accuracy by the number of trees')  
#plt.xlabel('Number of trees')  
#plt.ylabel('Mean Accuracy')  
#
#n = np.array(np.where(accuracy == np.max(accuracy))) + 1 # because the index starts at 0 
#n = n[0,0]
#n

