#!/usr/bin/env python
# coding: utf-8

# # Translation Memory Retrieval

# Note: Preprocessing is a separate module and must be done before using this!

# In[7]:


import sys
import nltk
import numpy as np
import json
import ast


# In[8]:


nltk.download('punkt')


# In[9]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# In[10]:


input_line = input()

#convert input to lowercase
input_line = input_line.lower()

#tokenise
input_tokens = word_tokenize(input_line)

content_words = [word for word in input_tokens if word not in stop_words] #Removing Stopwords

print(content_words)


# ## Edit Distance

# ### Load TM
# 
# Now we are dealing with the whole file which has ~800000 sentences in the TM. 
# 
# Approach:
# We take each sentence in the TM and check if any of the content words are present in it. If they are, we then calculate edit-distance and store it. This way we save time as we don't have to calculate edit distance for each sentence in the TM.
# 
# Once we have a list of edit distances, we take the N lowest, i.e. N best matches and print from the Target TM.

# In[11]:


src_tm_words = [] #Content Words in Source TM

with open('../tm_data/tm_src_pp.txt') as src_tm:
    line = src_tm.readline()
    
    while line:
        line = line.rstrip() #Removing Trailing Whitespace
        
        words = line.split('\t')
        src_tm_words.append(words)
        
        line = src_tm.readline()


# ## Load Index
# 
# The Index has been created by a separate code to make look-up of content words faster.

# In[12]:


with open('../tm_data/indexed_values_full.json') as json_file:
    indexed_values_str = json.load(json_file)

indexed_dict = ast.literal_eval(indexed_values_str)


# ## Execute Edit Distance
# 
# Instead of performing a Naive Search, we look up through the index to find out which candidate sentences in the TM contain the content words in the input sentences. We then run edit-distance on only these sentences and return the top N results.

# In[38]:


N = 5 #Top N matches returned

picked_candidates_indices = []

edit_distance_all = []

i = 0
count = 0

for word in content_words: 
    try:
        picked_candidates_indices += indexed_dict[word] #Adding indices to picked indices
    except: #If word not in indexed_dict
        pass
    
picked_candidates_indices = list(set(picked_candidates_indices)) #Removing Duplicates due to overlap

#for x in picked_candidates_indices:
#    print(src_tm_words[x-1])

for index in picked_candidates_indices:
    #since TM is 1-indexed and an array is 0-indexed we subtract 1 when accessing src_tm_words
    ed = nltk.edit_distance(content_words, src_tm_words[index-1]) #Calculate Edit Distance only if content words exist
    edit_distance_all.append(ed)
    
print('Running Edit Distance on ' + str(len(picked_candidates_indices)) + ' Candidates out of a possible ' + str(len(src_tm_words)) + '!\n')
    
#Get top N results
edit_distance_all = np.array(edit_distance_all)

sorted_indices = np.argsort(edit_distance_all) #Sorts in ascending order and returns the indices of indices_all array
least_N_indices = sorted_indices[:N] #We want least edit distance

#print(sorted_indices[0:10])
#print(least_N_indices)

for i in least_N_indices:
    print(picked_candidates_indices[i], src_tm_words[picked_candidates_indices[i]-1], edit_distance_all[i])


# ## Retrieval of Target from TM

# In[37]:


tgt_tm_array = []

with open('../tm_data/tm_tgt.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
        
for i in least_N_indices:
    print(picked_candidates_indices[i], tgt_tm_array[picked_candidates_indices[i]-1])


# In[ ]:




