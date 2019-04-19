import sys
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


with sys.stdin as f:
	input_line = f.read()

	#tokenise
	tokens = word_tokenize(input_line)

	print(tokens)