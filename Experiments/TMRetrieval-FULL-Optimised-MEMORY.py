#!/usr/bin/env python
# coding: utf-8

# # Translation Memory Retrieval

# Note: Preprocessing is a separate module and must be done before using this!

# In[4]:


import sys
import nltk
import numpy as np
import time


# In[5]:


nltk.download('punkt')


# In[6]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# In[7]:


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
# We take a chunk of size N bytes and in that, We take each sentence in the TM and check if any of the content words are present in it. If they are, we then calculate edit-distance and store it. This way we save time as we don't have to calculate edit distance for each sentence in the TM.
# 
# This helps with huge files and is a scalable option, unlike the last option which would fail above a certain memory.
# 
# Once we have a list of edit distances, we take the N lowest, i.e. N best matches and print from the Target TM.

# In[8]:


def read_in_chunks(file_object, chunk_size):
    """Lazy function (generator) to read a file piece by piece."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


# In[18]:


N = 5 #Top N matches returned
csize = 3000000 #Chunk Size (In Bytes) ~10000 sentences in a chunk(assuming 30 bytes per sentence)

edit_distance_all = []
indices_all = []

add_to_next_chunk = ''

with open('../../tm_data/tm_src_pp.txt') as f: #f = sys.stdin
    last_noun = ''
    
    count = 0
    j = 0
    
    for piece in read_in_chunks(f, csize):

        piece = add_to_next_chunk + piece #Adding cut info from last chunk

        i = len(piece) - 1

        while(i >= 0): #looking for the last endline before the chunk ends
            if(piece[i] == '\n'):
                break

            i -= 1

        add_to_next_chunk = piece[i+1:] #Info for next chunk
        
        piece = piece[:i] #Remove part that has gone to next chunk
        piece = piece.rstrip()
        
        #Now, analysing this piece to get content words in Source TM
        
        src_tm = piece.split('\n')
        
        src_tm_words = [] #Content Words in Source TM
        
        for candidate in src_tm:
            words = candidate.split('\t') #Tab Separated TM
            
            src_tm_words.append(words)
            
        #Finding out which of these have the content words and calculating Edit Distance
        
        for candidate in src_tm_words:
            
            #Check if Content Words present in Candidate
            for word in content_words:
                if(word in candidate):
                    count += 1
                    #print(candidate)

                    ed = nltk.edit_distance(content_words, candidate) #Calculate Edit Distance only if content words exist

                    edit_distance_all.append(ed)
                    indices_all.append(j)

                    break

            j += 1
    
#    print('Running Edit Distance on ' + str(count) + ' Candidates out of a possible ' + str(j) + '!\n')


# ## Get Matches

# In[24]:


#Get top N results
edit_distance_all = np.array(edit_distance_all)

sorted_indices = np.argsort(edit_distance_all) #Sorts in ascending order and returns the indices of indices_all array
least_N_indices = sorted_indices[:N] #We want least edit distance

#print(least_N_indices)

final_indices = []

for i in least_N_indices:
    final_indices.append(indices_all[i])
    
#print(final_indices)


# ## Retrieval of Target from TM

# In[26]:


tgt_tm_array = []

with open('../../tm_data/tm_tgt.txt') as tgt_tm:
    line = tgt_tm.readline()
    
    while line:
        tgt_tm_array.append(line)
        line = tgt_tm.readline()
        
#for i in final_indices:
#    print(i, tgt_tm_array[i])


# In[ ]:

end = time.time()

print("Time Taken:", file=sys.stderr)
print(end - start, file=sys.stderr)





