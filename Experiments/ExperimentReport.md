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

Naive TM [10000 Sentences]: 37.33906102180481

Naive TM [~8 lakh Sentences]: > 300 (Did Not Stop)

Optimised TM (Content Word Pruning) [~10000 Sentences]: 32.51241321231133

Optimised TM (Content Word Pruning) [~8 lakh Sentences]: > 300 (Did Not Stop)



## OBSERVATIONS:

### Edit Distance

- Preprocessing the TM and storing only content words as Tab Separated Values gives a significant speed-up (Almost 6x) as the retrieval becomes faster and Edit Distance becomes faster.

- Optimising by pruning using Content Words and only running Edit Distance on Candidate Sentences with Content Words also gives a significant speedup as the Edit Distance is running only on a small subset of the TM.

- Indexing was supposed to Optimise it even further, however, two factors prevented this speed-up:
	- We noticed that for the Optimised TM, the bulk of the time (~13 seconds out of ~14 seconds is used in calculating Edit Distance, which is a time that wasn't reduced)

	- The Searching using Index is supposed to make it faster, however, loading the json as an index takes a significant amount of time, which increases overall time. Without this, we see a slight speedup from the previous program. This also shows that even though the earlier search was Naive it didn't take a lot of time.

### Weighted N-Grams