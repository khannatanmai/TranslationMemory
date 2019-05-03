#!/usr/bin/env python
# coding: utf-8

# # Translation Memory Retrieval

# Note: Preprocessing is a separate module and must be done before using this!

# In[1]:


import sys
import nltk
import numpy as np
import time


# In[2]:


nltk.download('punkt')


# In[3]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# In[4]:


input_line = input()

start = time.time()

#convert input to lowercase
input_line = input_line.lower()

#tokenise
input_tokens = word_tokenize(input_line)

content_words = [word for word in input_tokens if word not in stop_words] #Removing Stopwords

#print(content_words)


# ## Edit Distance

# ### Load TM

# In[5]:


src_tm_words = [] #Content Words in Source TM

with open('../../tm_data/tm_src_10000_pp.txt') as src_tm:
    line = src_tm.readline()
    
    while line:
        line = line.rstrip() #Removing Trailing Whitespace
        
        words = line.split('\t')
        src_tm_words.append(words)
        
        line = src_tm.readline()


# ## Execute Edit Distance

# In[6]:


edit_distance_all = []
N = 5 #Top N matches returned

for element in src_tm_words:
    ed = nltk.edit_distance(content_words, element) #Calculate Edit Distance with content words
    edit_distance_all.append(ed)
    
#Get top N results
edit_distance_all = np.array(edit_distance_all)

sorted_indices = np.argsort(edit_distance_all) #Sorts in ascending order and returns the indices of elements in original array
least_N_indices = sorted_indices[:N] #We want least edit distance

#for i in least_N_indices:
#    print(src_tm_words[i], edit_distance_all[i])


# ## Retrieval of Target from TM

# In[7]:


tgt_tm_array = []

with open('../../tm_data/tm_tgt_10000.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
        
#for i in least_N_indices:
#    print(tgt_tm_array[i])


end = time.time()

print("Time Taken:", file=sys.stderr)
print(end - start, file=sys.stderr)




