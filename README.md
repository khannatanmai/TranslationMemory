# NLP Applications Project

Tanmai Khanna         20161212
Vaishnavi Pamulapati  20161114

## Creating a Translation Memory and Retrieval

This is code to process a TM and retrieve phrases from it based on its Edit Distance from the Candidate Sentence.

## Objective

Create a Translation Memory from English to Hindi. When given an English sentence as input, return N best matches from the TM and their Hindi Translations.

### Matching

The Matching has been done by two methods:
- Edit Distance
- Weighted N-Grams

## Project Report

A report with Experimental Results and Observations can be found inside the `Experiments` folder: `ExperimentReport.md`

## Instruction to Run Project
- Install Dependecies `pip3 install -r requirements.txt`
- Download data from (link), create folder `Project` and store it inside in a folder called `tm_data`.
- Extract this folder in the folder `Project` such that project `Folder` consists of `tm_data` and this folder, i.e. `TranslationMemory`. Then go into the folder.

### Edit Distance
`TMRetrieval-FULL-Optimised.ipynb`
Run whole Notebook, Give Input Sentence when prompted.

### Weighted N-Grams
`TMRetrieval_weighted_Optimised.ipynb`
Run whole Notebook, Give Input Sentence when prompted.

## Project Details

### Data

Several versions of the data are available.
The original files ate `tm_src.txt` and `tm_tgt.txt` which are parallel phrase files (aligned) and can be accessed at (link) with several other versions of these files.

`tm_src_pp.txt` is the preprocessed Source TM which contains only content words for each sentence in the TM.

Several other files exist of multiple sizes. The sizes are mentioned in the file name. This size represents the number of sentences in them.

For eg., `tm_src_10000.txt` and `tm_tgt_10000.txt` both contain the first 10000 sentences of the original TM.

The smaller data files are available in this folder. The larger ones are available in the link provided above.

## Description of Files
Here is a Comprehensive Description of the files available in this Project.

### Experiments
- Contains the Experiment Report, which contains Experimental Results and Observations.

- Also contains Python files for all Python Notebooks

### Naive TM
`TMRetrieval_NoPreprocessing.ipynb` & `TMRetrieval_weighted.ipynb`
- Calculates Edit Distance / Weighted N-Grams on all sentences in TM
- Returns best N matches

### Optimised TM
`TMRetrieval-FULL-Optimised.ipynb` & `TMRetrieval_weighted_Optimised.ipynb`
- Uses Preprocessed Data
- Prunes Search Using Content Words
- Runs Edit Distance / Weighted N-grams on This Subset of TM
- Returns best N matches

### Optimised Memory TM
`TMRetrieval-FULL-Optimised-MEMORY.ipynb`
- Same as Optimised TM, but accesses TM from disk piece by piece.
- Scalable Solution

### Indexed Optimised TM
`TMRetrieval-FULL-Optimised-Indexed.ipynb`
- While pruning the search, instead of a Naive Search, an index is created from the preprocessed TM for faster search to prune Candidate Sentences and Edit Distance is run only on these Candidate Sentences.

### Preprocessing
`TMPreProcessing.ipynb`
Converts to lowercase and creates a new TM Source with only Content Words.

### Indexing
`IndexedSentences.ipynb`
Creates an Index from the TM.

### IDF Values
`IDF_values.ipynb`
Calculates IDF values of Document.

