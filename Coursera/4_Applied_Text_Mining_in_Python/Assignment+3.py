
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:

def answer_one():
    return spam_data['target'].mean() * 100


# In[4]:

answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[4]:

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():

    # Fit the CountVectorizer to the training data
    vect = CountVectorizer().fit(X_train)
    vocabulary = vect.get_feature_names()
    return max(vocabulary, key=len)


# In[6]:

answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[5]:

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    return roc_auc_score(y_test, predictions)


# In[8]:

answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[6]:

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    
    feature_names = np.array(vect.get_feature_names())
    tfidf_values = X_train_vectorized.max(0).toarray()[0]
    
    smallest_features = feature_names[sorted_tfidf_index[:20]]
    smallest_tfids = tfidf_values[sorted_tfidf_index[:20]]
    smallest_series = pd.Series(index=smallest_features, data=smallest_tfids)
    smallest_series = smallest_series.iloc[np.lexsort([smallest_series.index, smallest_series.values])]
    largest_features = feature_names[sorted_tfidf_index[:-21:-1]]
    largest_tfids = tfidf_values[sorted_tfidf_index[:-21:-1]]
    largest_series = pd.Series(index=largest_features, data=largest_tfids)
    largest_series = largest_series.iloc[np.lexsort([largest_series.index, largest_series.values])]
    
    return (smallest_series, largest_series)


# In[10]:

answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:

def answer_five():
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))

    return roc_auc_score(y_test, predictions)


# In[12]:

answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[8]:

def answer_six():
    df = spam_data.groupby('target').aggregate(lambda x: np.mean(x.str.len()))
    return (df.loc[0,'text'], df.loc[1,'text'])


# In[48]:

answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[9]:

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[10]:

from sklearn.svm import SVC

def answer_seven():
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    length = X_train.str.len()
    X_train_vectorized = add_feature(vect.transform(X_train), length)
    
    model = SVC(C=10000)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(add_feature(vect.transform(X_test), X_test.str.len()))

    return roc_auc_score(y_test, predictions)
    


# In[26]:

answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[11]:

def answer_eight():
    df = spam_data.copy()
    def count_digit(row):
        row['# digit'] = len([c for c in row['text'] if c.isdigit()])
        return row
    df = df.apply(count_digit, axis=1)
    df = (df.groupby('target')['# digit'].aggregate(np.mean))
    return (df.iloc[0], df.iloc[1])
    


# In[58]:

answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[36]:

from sklearn.linear_model import LogisticRegression

def answer_nine():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression


    nbDigit_train = X_train.map(lambda x: len([c for c in x if c.isdigit()]))
    nbDigit_test = X_test.map(lambda x: len([c for c in x if c.isdigit()]))

    
    model = LogisticRegression(C=100)

    vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)

    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    X_train_vectorized = add_feature(X_train_vectorized, nbDigit_train)
    
    X_test_vectorized = vect.transform(X_test)
    X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())
    X_test_vectorized = add_feature(X_test_vectorized, nbDigit_test)

    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)

    
    return roc_auc_score(y_test, predictions)


# In[37]:

answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[34]:

def answer_ten():
    
    df = spam_data.copy()
    df['# Non words'] = df['text'].str.findall(r'\W').map(lambda x: len(x))
    df = (df.groupby('target').aggregate(np.mean))
    return (df.loc[0,'# Non words'], df.loc[1,'# Non words'])


# In[35]:

answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[88]:

def answer_eleven():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    

    nbDigit_train = X_train.map(lambda x: len([c for c in x if c.isdigit()]))
    nbDigit_test = X_test.map(lambda x: len([c for c in x if c.isdigit()]))

    nbNonWords_train = X_train.str.findall(r'\W').map(lambda x: len(x))
    nbNonWords_test = X_test.str.findall(r'\W').map(lambda x: len(x))
    
    model = LogisticRegression(C=100)

    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)

    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    X_train_vectorized = add_feature(X_train_vectorized, nbDigit_train)
    X_train_vectorized = add_feature(X_train_vectorized, nbNonWords_train)
    
    X_test_vectorized = vect.transform(X_test)
    X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())
    X_test_vectorized = add_feature(X_test_vectorized, nbDigit_test)
    X_test_vectorized = add_feature(X_test_vectorized, nbNonWords_test)

    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)

    coef = np.sort(model.coef_.tolist()[0])
    
    auc = roc_auc_score(y_test, predictions)
    return (auc, coef[:10], coef[-10:])


# In[89]:

answer_eleven()


# In[ ]:



