
# coding: utf-8

# # Translation Memory Retrieval using Weighted N-Grams

# In[81]:


import nltk
import math
from collections import Counter
import string
import numpy as np


# In[82]:


nltk.download('punkt')


# In[83]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# In[84]:


# input_line = input()

sentence = "I request you to remove the drive safely"
# sentence = "There are a few controversies surrounding the the company may keep changing its business strategy topic how many songs did Rafi sing during his lifetime"

M_ngrams = get_M_ngrams(sentence)


# ## Weighted N-Gram Precision

# ### Get sentences and IDF values

# In[85]:


idf_values = {}
with open("tm_data/tm_src_2000.txt") as source_file:
    sentences = source_file.read().splitlines()


tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
all_tokens_set = set([item for sublist in tokenized_sentences for item in sublist])
for tkn in all_tokens_set:
    contains_token = map(lambda doc: tkn in doc, tokenized_sentences)
    idf_values[tkn] = 1 + math.log(len(tokenized_sentences)/(sum(contains_token)))

print(idf_values)


# ### Getting the M_ngrams and C_ngrams

# In[86]:


def get_M_ngrams(sentence):
    ngrams_list_sent = []
    M_ngrams = []
    counter_ngrams = []
    
    ngrams = list(nltk.ngrams(sentence.split(), 4))
    ngrams_list_sent.append(list(ngrams))
    M_ngrams = [y for x in ngrams_list_sent for y in x]
    
    for ngrams in M_ngrams:
        counter_ngrams.append(Counter(ngrams))
        
    return M_ngrams


# In[87]:


def get_C_ngrams(candidate_sentence):
    ngrams_list_sent = []
    C_ngrams = []
    counter_ngrams = []
    
    ngrams = list(nltk.ngrams(candidate_sentence.split(), 4))
    ngrams_list_sent.append(list(ngrams))
    C_ngrams = [y for x in ngrams_list_sent for y in x]
    ngrams_sents = []
    ngrams_list_sent = []
    
    for ngrams in C_ngrams:
        counter_ngrams.append(Counter(ngrams))
    
    return C_ngrams


# ### To compute numerator and denominator

# In[88]:


def ngrams_intersection(candidate_sentence):
    C_ngrams = get_C_ngrams(candidate_sentence)
    
    M_set = set(M_ngrams)
    C_set = set(C_ngrams)
    
    return list(M_set & C_set)


# In[89]:


def compute_w_sum(ngrams_list):
    w = 0
    
    for ngram in ngrams_list:
        for token in ngram:
            if token in idf_values:
                w+= idf_values[token] 
    return w


# ### Final score for each sentence wrt to input sentence

# In[90]:


def compute_wpn(candidate_sentence):
    C_ngrams = get_C_ngrams(candidate_sentence)
    intersection_ngrams = ngrams_intersection(candidate_sentence)
    Z = 0.75
    
    w_M_ngrams = compute_w_sum(M_ngrams)
    w_C_ngrams = compute_w_sum(C_ngrams)
    w_intersection_ngrams = compute_w_sum(intersection_ngrams)
    
    
    wpn = w_intersection_ngrams / ((Z*w_M_ngrams) + ((1-Z)*w_C_ngrams))
    
    return wpn


# In[93]:


max_wpn = 0
wnp_all = []
N = 5

for sentence in sentences:
    wpn = compute_wpn(sentence)
    wnp_all.append(wpn)
    if wpn > max_wpn:
        max_wpn = wpn
        best_sentence = sentence
            
        
wnp_all = np.array(wnp_all)
sorted_indices = np.argsort(wnp_all) 
least_N_indices = sorted_indices[-N:] 

print()
for i in least_N_indices:
    print([i], sentences[i], wnp_all[i])


# ### Retrieval of Target from TM

# In[92]:


tgt_tm_array = []

with open('../project/tm_data/tm_tgt.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
  
    for i in least_N_indices:
        print([i], tgt_tm_array[i])

