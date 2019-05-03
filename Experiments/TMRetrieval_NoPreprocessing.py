#!/usr/bin/env python
# coding: utf-8

# # Translation Memory Retrieval

# In[1]:


import sys
import nltk
import numpy as np
import time


# In[2]:


nltk.download('punkt')


# In[3]:


from nltk.tokenize import word_tokenize


# In[4]:


input_line = input()

start = time.time()

#tokenise
candidate_tokens = word_tokenize(input_line)

#print(candidate_tokens)


# ## Edit Distance

# ### Tokenize Source TM

# In[6]:


src_tm_tokenized = [] #Tokenized TM SRC

with open('../../tm_data/tm_src.txt') as src_tm:
    line = src_tm.readline()
    
    while line:
        tokens = word_tokenize(line)
        src_tm_tokenized.append(tokens)
        line = src_tm.readline()


# ## Execute Edit Distance

# In[7]:


edit_distance_all = []
N = 5 #Top N matches returned

for element in src_tm_tokenized:
    ed = nltk.edit_distance(candidate_tokens, element)
    edit_distance_all.append(ed)
    
#Get top N results
edit_distance_all = np.array(edit_distance_all)

sorted_indices = np.argsort(edit_distance_all) #Sorts in ascending order and returns the indices of elements in original array
least_N_indices = sorted_indices[:N] #We want least edit distance

#for i in least_N_indices:
    #print(src_tm_tokenized[i], edit_distance_all[i])


# ## Retrieval of Target from TM

# In[8]:


tgt_tm_array = []

with open('../../tm_data/tm_tgt.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
        
#for i in least_N_indices:
    #print(tgt_tm_array[i])

end = time.time()

print("Time Taken:", file=sys.stderr)
print(end - start, file=sys.stderr)




