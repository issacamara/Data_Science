
# coding: utf-8

# # Module 2 (Python 3)

# ## Basic NLP Tasks with NLTK

# In[2]:

import nltk
from nltk.book import *


# ### Counting vocabulary of words

# In[3]:

text7


# In[4]:

sent7


# In[5]:

len(sent7)


# In[6]:

len(text7)


# In[7]:

len(set(text7))


# In[8]:

list(set(text7))[:10]


# ### Frequency of words

# In[9]:

dist = FreqDist(text7)
len(dist)


# In[10]:

vocab1 = dist.keys()
#vocab1[:10] 
# In Python 3 dict.keys() returns an iterable view instead of a list
list(vocab1)[:10]


# In[11]:

dist['four']


# In[12]:

freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
freqwords


# ### Normalization and stemming

# In[13]:

input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')
words1


# In[14]:

porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]


# ### Lemmatization

# In[15]:

udhr = nltk.corpus.udhr.words('English-Latin1')
udhr[:20]


# In[16]:

[porter.stem(t) for t in udhr[:20]] # Still Lemmatization


# In[17]:

WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in udhr[:20]]


# ### Tokenization

# In[31]:

text11 = "Children shouldn't drink a sugary drink before bed."
text11.split(' ')


# In[19]:

nltk.word_tokenize(text11)


# In[20]:

text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12)
len(sentences)


# In[21]:

sentences


# ## Advanced NLP Tasks with NLTK

# ### POS tagging

# In[22]:

nltk.help.upenn_tagset('MD')


# In[23]:

text13 = nltk.word_tokenize(text11)
nltk.pos_tag(text13)


# In[24]:

text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
nltk.pos_tag(text14)


# In[25]:

# Parsing sentence structure
text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

parser = nltk.ChartParser(grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


# In[26]:

text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.data.load('mygrammar.cfg')
grammar1


# In[27]:

parser = nltk.ChartParser(grammar1)
trees = parser.parse_all(text16)
for tree in trees:
    print(tree)


# In[28]:

from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)


# ### POS tagging and parsing ambiguity

# In[29]:

text18 = nltk.word_tokenize("The old man the boat")
nltk.pos_tag(text18)


# In[30]:

text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
nltk.pos_tag(text19)

