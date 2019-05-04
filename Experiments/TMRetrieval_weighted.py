
# coding: utf-8

# # Translation Memory Retrieval using Weighted N-Grams

# In[116]:


import nltk
import math
from collections import Counter
import string
import numpy as np
import json
import ast


# In[117]:


nltk.download('punkt')


# In[118]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# In[119]:


input_line = input()

sentence = input_line
# sentence = "There were many controversies about the songs he performed during his lifetime ."


# ## Weighted N-Gram Precision

# ### Get sentences and IDF values

# In[120]:


with open("../tm_data/tm_src_2000.txt") as source_file:
    sentences = source_file.read().splitlines()


with open('../idf_values.json') as json_file:
    idf_values_str = json.load(json_file)

idf_values = ast.literal_eval(idf_values_str)


# ### Getting the M_ngrams and C_ngrams

# In[121]:


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


# In[122]:


M_ngrams = get_M_ngrams(sentence)


# In[123]:


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

# In[124]:


def ngrams_intersection(candidate_sentence):
    C_ngrams = get_C_ngrams(candidate_sentence)
    
    M_set = set(M_ngrams)
    C_set = set(C_ngrams)
    
    return list(M_set & C_set)


# In[125]:


def compute_w_sum(ngrams_list):
    w = 0
    
    for ngram in ngrams_list:
        for token in ngram:
            if token in idf_values:
                w+= idf_values[token] 
    return w


# ### Final score for each sentence wrt to input sentence

# In[126]:


def compute_wpn(candidate_sentence):
    C_ngrams = get_C_ngrams(candidate_sentence)
    intersection_ngrams = ngrams_intersection(candidate_sentence)
    Z = 0.75
    
    w_M_ngrams = compute_w_sum(M_ngrams)
    w_C_ngrams = compute_w_sum(C_ngrams)
    w_intersection_ngrams = compute_w_sum(intersection_ngrams)
    
    
    wpn = w_intersection_ngrams / ((Z*w_M_ngrams) + ((1-Z)*w_C_ngrams))
    
    return wpn


# In[127]:


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

# In[128]:


tgt_tm_array = []

with open('../../project/tm_data/tm_tgt.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
  
    for i in least_N_indices:
        print([i], tgt_tm_array[i])

