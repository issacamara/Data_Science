#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 19:00:58 2018

@author: IssaCamara
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re, sys
import seaborn as sns
import statsmodels.api as sm
from math import ceil
from sklearn import svm, tree
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.preprocessing import StandardScaler  


from collections import Counter

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def get_first_letter(name):
    if (name == "nan") or (len(name) == 0):
        return ""
    return name[0]

pd.options.mode.chained_assignment = None  # default='warn'

train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)

full_data = [train, test]

for dataset in full_data: 
        
#    group = ["Pclass","Sex","Parch","SibSp"]
#    
#    mean_age = list(dataset.groupby(group).Age.mean())
    
    
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
    
    for i in index_NaN_age:
        age_med = dataset["Age"].median()
        age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & 
                          (dataset['Parch'] == dataset.iloc[i]["Parch"]) & 
                          (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
        if not np.isnan(age_pred) :
            dataset['Age'].iloc[i] = age_pred
        else :
            dataset['Age'].iloc[i] = age_med
                    
    
            
    #dataset["Age"] = dataset["Age"].fillna(mean_age)
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
    dataset['Fare'] = dataset['Fare'].fillna(np.mean(dataset.Fare))
    dataset['Cabin'] = dataset['Cabin'].astype(str)
    dataset['Cabin'] = dataset['Cabin'].apply(get_first_letter)
    dataset['Cabin'] = dataset['Cabin'].fillna('X')
    
    
    
 # Mapping Age
for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 15, 'CategoricalAge'] = 0
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 60), 'CategoricalAge'] = 1
    dataset.loc[ dataset['Age'] > 60, 'CategoricalAge'] = 2 
    dataset.CategoricalAge = dataset.CategoricalAge.astype(int)

 # Mapping Sex
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
        
 # Mapping FamilySize
for dataset in full_data:
    dataset.loc[ dataset['FamilySize'] == 1, 'CategoricalFamilySize'] = 0
    dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] <= 4), 
                'CategoricalFamilySize'] = 1
    dataset.loc[(dataset['FamilySize'] > 4), 'CategoricalFamilySize'] = 2
    dataset.CategoricalFamilySize = dataset.CategoricalFamilySize.astype(int)


 # Mapping Fare
for dataset in full_data:
    dataset.loc[ dataset['Fare'] < 15, 'CategoricalFare'] = 0
    dataset.loc[ (dataset['Fare'] >= 15) & (dataset['Fare'] < 40), 'CategoricalFare'] = 1
    dataset.loc[ (dataset['Fare'] >= 40) & (dataset['Fare'] < 80), 'CategoricalFare'] = 2
    dataset.loc[ dataset['Fare'] >= 80, 'CategoricalFare'] = 3
    dataset.CategoricalFare = dataset.CategoricalFare.astype(int)


for dataset in full_data:
    
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 
           'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 
           'C': 1, 'Q': 2} ).astype(int)


sns.heatmap(train[["Age","Sex","SibSp","Parch","Pclass",'FamilySize']].corr(),
            cmap="BrBG",annot=True)

sns.heatmap(test[["Age","Sex","SibSp","Parch","Pclass",'FamilySize']].corr(),
            cmap="BrBG",annot=True)


# Analyzing data by plotting some charts
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

f_min = 40
f_max = 80
survived[(survived["Fare"] > f_min) & (survived["Fare"] < f_max)]["Fare"].plot.hist(
        alpha=0.5, color='red',bins=50)
died[(died["Fare"] > f_min) & (died["Fare"] < f_max)]["Fare"].plot.hist(alpha=0.5,
     color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

a_min = 15
a_max = 80
survived[(survived["Age"] > a_min) & (survived["Age"] < a_max)]["Age"].plot.hist(
        alpha=0.5, color='red',bins=50)
died[(died["Age"] > a_min) & (died["Age"] < a_max)]["Age"].plot.hist(alpha=0.5,
     color='blue',bins=50)
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

table = pd.crosstab(train.Survived,train.Title)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

table = pd.crosstab(train.Title,train.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

''' -------------------------------------------------------------------- '''


#for col in train.columns:
#    if col not in ['Survived','PassengerId','Name','Age'] :
#        table = pd.crosstab(train['Survived'],train[col])
#        table.div(table.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True)



# Cleaning data
#train.drop(['Cabin'],1)
#train.dropna(inplace=True)
train.isnull().sum()
#test.drop(['Cabin'],1)
test.isnull().sum()
#Creating dummy variables
dummy_var = ['Title','CategoricalAge', 'Cabin', 'Pclass', 'Embarked', 
             'CategoricalFare', 'CategoricalFamilySize']
prefix = ['Title','Age', 'Cabin', 'Class', 'Embarked', 'Fare', 'FamilySize']
predictors = []
for i in dummy_var:
    dummy = pd.get_dummies(train[i],prefix = i)
    train = train.join(dummy)
    dummy = pd.get_dummies(test[i],prefix = i)
    test = test.join(dummy)
    predictors = predictors + list(dummy)
  

#data_vars=train.columns.values.tolist()
#to_keep=[i for i in data_vars if i not in dummy_var]
#train=train[to_keep]
#
#data_vars=test.columns.values.tolist()
#to_keep=[i for i in data_vars if i not in dummy_var]
#test=test[to_keep]

#predictors = predictors + (list(['SibSp', 'Parch']))

all_X = train[predictors]
all_y = train['Survived']

###########################################################################@
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),
                                      random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

kfold = KFold(n_splits=5)

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, all_X, y = all_y, 
                                      scoring = "accuracy", cv = kfold, n_jobs=1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
                       "Algorithm":["SVC","DecisionTree","AdaBoost","RandomForest",
                                    "ExtraTrees","GradientBoosting",
                                    "MultipleLayerPerceptron","KNeighboors",
                                    "LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",
                orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")



DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsadaDTC.fit(all_X,all_y)

ada_best = gsadaDTC.best_estimator_

gsadaDTC.best_score_

#########################################################################################
#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsExtC.fit(all_X,all_y)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_
#########################################################################################

# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 1, verbose = 1)

gsRFC.fit(all_X,all_y)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_

#########################################################################################

# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", 
                     n_jobs= 1, verbose = 1)

gsGBC.fit(all_X,all_y)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_
#########################################################################################

### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", 
                      n_jobs= 1, verbose = 1)

gsSVMC.fit(all_X,all_y)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_

#########################################################################################


nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", 
                         figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),
                     ("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=all_X.columns[indices][:40],
                        x = classifier.feature_importances_[indices][:40], 
                        orient='h', ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1

#########################################################################################

test_Survived_RFC = pd.Series(RFC_best.predict(test[predictors]), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test[predictors]), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test[predictors]), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test[predictors]), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test[predictors]), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g = sns.heatmap(ensemble_results.corr(),annot=True)


#########################################################################################
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=1)

votingC = votingC.fit(all_X, all_y)
#########################################################################################

test_Survived = pd.Series(votingC.predict(test[predictors]), name="Survived")


submit("ensemble_python_voting.csv",votingC)



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

print("The accuracy for Logistic Regression is", accuracy)

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
    #scores = cross_val_score(knn, scaler.transform(all_X), all_y, cv=kfold)
    scores = cross_val_score(knn, all_X, all_y, cv=kfold)
    accuracy.append(scores.mean())

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), accuracy, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Accuracy Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Accuracy')  

K = np.array(np.where(accuracy == np.max(accuracy))) + 1 # because the index starts at 0 
K = K[0,0]

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(all_X,all_y) 
#knn.fit(scaler.transform(all_X),all_y) 
 
submit("submission_knn.csv",knn)


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


rfc = RandomForestClassifier(n_estimators=50) # after cross validation, 4 was the best 
# Train the model using the training sets and check score
rfc.fit(all_X, all_y)

submit("submission_rfc.csv",rfc)



######################################### GBM #########################################

val = [1, 5, 10, 30, 50, 100]
val = [0.01, 0.1, 0.5, 1.0, 1.5]
val = [1, 5, 10, 30, 50, 100]
accuracy = []
for i in val:  
    gbm = GradientBoostingClassifier(n_estimators=i, learning_rate=0.1, max_depth=50)
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

gbm = GradientBoostingClassifier(n_estimators=5, learning_rate=0.3, max_depth=2)
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


val = [1, 5, 10, 50, 100]
val = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 1]
accuracy = []
for i in val:  
    model = svm.SVC(kernel='rbf', C=1, gamma=i) 
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

model = svm.SVC(kernel='rbf', C=1, gamma=0.1) 
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

