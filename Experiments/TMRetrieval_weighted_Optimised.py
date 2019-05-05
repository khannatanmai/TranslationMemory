
# coding: utf-8

# # Translation Memory Retrieval using Weighted N-Grams

# In[394]:


import nltk
import math
from collections import Counter
import string
import numpy as np
import json
import ast


# In[395]:


nltk.download('punkt')


# In[396]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# ### Getting the M_ngrams and C_ngrams

# In[397]:


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


# In[398]:


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


# In[399]:


input_line = input()

#convert input to lowercase
input_line = input_line.lower()

#tokenise
input_tokens = word_tokenize(input_line)

content_words = [word for word in input_tokens if word not in stop_words] #Removing Stopwords
print(content_words)

# new_M_sentence = ' '.join(content_words)
M_ngrams = get_M_ngrams(input_line)

# sentence = "There were many controversies about the songs he performed during his lifetime ."


# ## Weighted N-Gram Precision

# ### Load TM

# In[400]:


src_tm_words = [] #Content Words in Source TM

with open('../tm_data/tm_src_2000_pp.txt') as src_tm:
    line = src_tm.readline()
    
    while line:
        line = line.rstrip() #Removing Trailing Whitespace
        
        words = line.split('\t')
        src_tm_words.append(words)
        
        line = src_tm.readline()


# ### Get sentences and IDF values

# In[401]:


with open("../tm_data/tm_src_10000_lower.txt") as source_file:
    sentences = source_file.read().splitlines()


# In[402]:


with open('../tm_data/idf_values_2000.json') as json_file:
    idf_values_str = json.load(json_file)

idf_values = ast.literal_eval(idf_values_str)
print(idf_values)


# ### To compute numerator and denominator

# In[403]:


def ngrams_intersection(candidate_sentence):
    C_ngrams = get_C_ngrams(candidate_sentence)
    
    M_set = set(M_ngrams)
    C_set = set(C_ngrams)
    
    return list(M_set & C_set)


# In[404]:


def compute_w_sum(ngrams_list):
    w = 0
    
    
    for ngram in ngrams_list:
        for token in ngram:
            if token in idf_values:
                w += idf_values[token]
#                 print(w)
    
    return w


# ### Final score for each sentence wrt to input sentence

# In[405]:


def compute_wpn(candidate_sentence):
    C_ngrams = get_C_ngrams(candidate_sentence)
    intersection_ngrams = ngrams_intersection(candidate_sentence)
        
    Z = 0.75
    
    w_M_ngrams = compute_w_sum(M_ngrams)
    w_C_ngrams = compute_w_sum(C_ngrams)
    w_intersection_ngrams = compute_w_sum(intersection_ngrams)

    
    
    wpn = w_intersection_ngrams / ((Z*w_M_ngrams) + ((1-Z)*w_C_ngrams))
    
    return wpn


# ## Optimisation

# In[406]:


N = 5 #Top N matches returned

wpn_all = []
indices_all = []

j = 0
count = 0

for i, candidate in enumerate(src_tm_words):
    
    #Check if Content Words present in Candidate
    for word in content_words:
        if(word in candidate):
            count += 1
            
            wpn = compute_wpn(sentences[i])
            
            wpn_all.append(wpn)
            indices_all.append(j)
            
            break
    
    j += 1
    
print('Running WNGP on ' + str(count) + ' Candidates out of a possible ' + str(j) + '!\n')


#Get top N results
wpn_all = np.array(wpn_all)

sorted_indices = np.argsort(wpn_all) #Sorts in ascending order and returns the indices of indices_all array
least_N_indices = sorted_indices[-N:] 

for i in least_N_indices:
    print([indices_all[i]], src_tm_words[indices_all[i]], wpn_all[i])


# ### Retrieval of Target from TM

# In[407]:


tgt_tm_array = []

with open('../tm_data/tm_tgt_2000.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
        
for i in least_N_indices:
    print([indices_all[i]], tgt_tm_array[indices_all[i]])

