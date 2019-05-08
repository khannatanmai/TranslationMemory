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
# 
# Now we are dealing with the whole file which has ~800000 sentences in the TM. 
# 
# Approach:
# We take each sentence in the TM and check if any of the content words are present in it. 
# 
# Then we rank these candidate sentences based on how many content words they have in common with the input sentence.
# 
# We take the top 500 candidate sentences, calculate edit-distance and store it. This way we save time as we don't have to calculate edit distance for each sentence in the TM.
# 
# Once we have a list of edit distances, we take the N lowest, i.e. N best matches and print from the Target TM.

# In[7]:


src_tm_words = [] #Content Words in Source TM

with open('../../tm_data/tm_src_pp.txt') as src_tm:
    line = src_tm.readline()
    
    while line:
        line = line.rstrip() #Removing Trailing Whitespace
        
        words = line.split('\t')
        src_tm_words.append(words)
        
        line = src_tm.readline()

# ## Ranking

# In[26]:


match_counts = []
match_indices = []

i = 1
for candidate in src_tm_words:
    
    count_for_candidate = 0
    flag = 0 
    
    for word in content_words:
        if(word in candidate):
            count_for_candidate += 1 #adding count of input content words in candidate
            flag = 1 #to avoid adding same index multiple times
    
    if (flag == 1): #only add for candidates which have at least one input content word
        match_indices.append(i)
        match_counts.append(count_for_candidate)
    
    i += 1
    
#print(len(match_counts))

#Sorting match_indices based on match_counts
sorted_indices = [x for _,x in sorted(zip(match_counts,match_indices))]

#Now we want the top 500 ranks (or if less than 500 matches, then all of them)
final_indices = []

if(len(sorted_indices) <= 500):
    final_indices = sorted_indices
else:
    final_indices = sorted_indices[-500:] #Only put top 500 ranks in final indices
    
print(final_indices[-10:]) #Show top 10 ranks
print(str(len(match_counts)) + ' got reduced to ' + str(len(final_indices)))


# ## Execute Edit Distance

# In[32]:


N = 5 #Top N matches returned

edit_distance_all = []
indices_all = []

for index_tm in final_indices:
    ed = nltk.edit_distance(content_words, src_tm_words[index_tm - 1]) #Calculate Edit Distance (src_tm_words is 0 indexed)
            
    edit_distance_all.append(ed)
    indices_all.append(index_tm)
    
print('Running Edit Distance on ' + str(len(final_indices)) + ' Ranked Candidates out of a possible ' + str(len(src_tm_words)) + '!\n')
    
#Get top N results
edit_distance_all = np.array(edit_distance_all)

sorted_indices = np.argsort(edit_distance_all) #Sorts in ascending order and returns the indices of indices_all array
least_N_indices = sorted_indices[:N] #We want least edit distance

#print(least_N_indices)

for i in least_N_indices:
    print(indices_all[i], src_tm_words[indices_all[i]-1], edit_distance_all[i])


# ## Retrieval of Target from TM

# In[34]:


tgt_tm_array = []

with open('../tm_data/tm_tgt.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
        
for i in least_N_indices:
    print(indices_all[i], tgt_tm_array[indices_all[i]-1])

