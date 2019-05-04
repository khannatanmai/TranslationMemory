#!/usr/bin/env python
# coding: utf-8

# # Translation Memory Retrieval

# Note: Preprocessing is a separate module and must be done before using this!

# In[6]:


import sys
import nltk
import numpy as np
import time


# In[7]:


nltk.download('punkt')


# In[8]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# In[9]:


input_line = input()

#start = time.time()

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
# We take each sentence in the TM and check if any of the content words are present in it. If they are, we then calculate edit-distance and store it. This way we save time as we don't have to calculate edit distance for each sentence in the TM.
# 
# Once we have a list of edit distances, we take the N lowest, i.e. N best matches and print from the Target TM.

# In[10]:


src_tm_words = [] #Content Words in Source TM

with open('../../tm_data/tm_src_pp.txt') as src_tm:
    line = src_tm.readline()
    
    while line:
        line = line.rstrip() #Removing Trailing Whitespace
        
        words = line.split('\t')
        src_tm_words.append(words)
        
        line = src_tm.readline()


# ## Execute Edit Distance

# In[25]:


N = 5 #Top N matches returned

edit_distance_all = []
indices_all = []

i = 0
count = 0

start = time.time()
for candidate in src_tm_words:
    
    #Check if Content Words present in Candidate
    for word in content_words:
        if(word in candidate):
            count += 1
            #print(candidate)
            
            ed = nltk.edit_distance(content_words, candidate) #Calculate Edit Distance only if content words exist
            
            edit_distance_all.append(ed)
            indices_all.append(i)
            
            break
    
    i += 1

end = time.time()
    
#print('Running Edit Distance on ' + str(count) + ' Candidates out of a possible ' + str(i) + '!\n')
    
#Get top N results
edit_distance_all = np.array(edit_distance_all)

sorted_indices = np.argsort(edit_distance_all) #Sorts in ascending order and returns the indices of indices_all array
least_N_indices = sorted_indices[:N] #We want least edit distance

#print(least_N_indices)

for i in least_N_indices:
    print(indices_all[i], src_tm_words[indices_all[i]], edit_distance_all[i])


# ## Retrieval of Target from TM

# In[27]:


tgt_tm_array = []

with open('../../tm_data/tm_tgt.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
        
#for i in least_N_indices:
#    print(indices_all[i], tgt_tm_array[indices_all[i]])


#end = time.time()

print("Time Taken:", file=sys.stderr)
print(end - start, file=sys.stderr)





