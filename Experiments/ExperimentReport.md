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

### Method Used: WEIGHTED N_GRAMS

Naive TM [10000 Sentences]: 37.33906102180481

Naive TM [~8 lakh Sentences]: > 300 (Did Not Stop)

Optimised TM (Content Word Pruning) [~10000 Sentences]: 32.51241321231133

Optimised TM (Content Word Pruning) [~8 lakh Sentences]: > 300 (Did Not Stop)


