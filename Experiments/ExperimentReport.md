# Experimental Set-Up

Sentence to be tested: I request you to please remove the drive safely.

## Time Taken (in seconds)

### Method Used: EDIT DISTANCE

Naive TM [10000 Sentences]: 1.8652219772338867

Naive TM [~8 lakh Sentences]: 144.49711513519287

Preprocessed TM (Content Words) [10000 Sentences]: 0.34960317611694336

Preprocessed TM (Content Words) [~8 lakh Sentences]: 25.34655499458313

Optimised TM (Content Word Pruning) [~8 lakh Sentences]: 14.540621042251587

Optimised TM using Chunk Reading [~8 lakh Sentences]: 15.623741865158081

Indexing Optimisation [~8 lakh Sentences]: 
Loading Index: 	17.758535861968994
Edit Distance: 	13.452616930007935
Remaining: 		0.7476191520690918

Total: 			31.95877194404602

### Method Used: WEIGHTED N_GRAMS

Naive TM (Without Storing IDF) [~8 lakh Sentences]: > 300.0

Naive TM [10000 Sentences]: 0.600862979888916

Naive TM [~8 lakh Sentences]: 39.4572548866272

Optimised TM (Without Storing IDF) [~8 lakh Sentences]: > 300.0

Optimised TM (Content Word Pruning) [~10000 Sentences]: 0.4254031181335449

Optimised TM (Content Word Pruning) [~8 lakh Sentences]: 18.43024206161499



## OBSERVATIONS:

### Edit Distance

- Preprocessing the TM and storing only content words as Tab Separated Values gives a significant speed-up (Almost 6x) as the retrieval becomes faster and Edit Distance becomes faster.

- Optimising by pruning using Content Words and only running Edit Distance on Candidate Sentences with Content Words also gives a significant speedup as the Edit Distance is running only on a small subset of the TM.

- Indexing was supposed to Optimise it even further, however, two factors prevented this speed-up:
	- We noticed that for the Optimised TM, the bulk of the time (~13 seconds out of ~14 seconds is used in calculating Edit Distance, which is a time that wasn't reduced)

	- The Searching using Index is supposed to make it faster, however, loading the json as an index takes a significant amount of time, which increases overall time. Without this, we see a slight speedup from the previous program. This also shows that even though the earlier search was Naive it didn't take a lot of time.

### Weighted N-Grams

 - In the numerator of the formula to compute WNGP, it computes the IDF scores of only the N-grams belonging to the intersection of M_N-grams and C_N-grams. Hence, even before pruning using Content words, a filter is used. 

 - As observed when Edit Distance is computed, using Content words as a filter sped up the process as the WNGP is only being computed on a smaller subset of the TM.

 - Storing the IDF values in a separate JSON file and retrieving the values only at the time of computing Wpn sped up the computation of WNGP by 7.5x for Naive TM and 16x for Optimised TM.

 - In the formula to calculate Wpn, an optimal value of Z = 0.75 was taken. A higher value of Z would mean getting longer translation matches, and a lower value of Z would mean getting shorter translation matches. But, we experimented with values Z = 0.3, 0.75, 0.9 and received the same results on the Non-Optimised Weighted file and the Optimised Weighted file. 

 - Initially we had decided to use 4-grams. But, bigger the N-gram, greater the discrimination/restriction (more information about the context of the specific instance). Hence, we decided to use Bigrams which gave better results. 