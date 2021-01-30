# CS 175 Story Generator
# Tiffany Kong 93065152
# Henry Fong 83642159
# Ashley Teves 10177429

import numpy as np 
import nltk

import torch
from torch import nn

"""
general flow:
* open & extract file contents #
* tokenize text #
* convert vocab to numeric format #
* build sequences #
* build model
* train model
* predict/generate text
"""


# file = 'alice.txt'
# raw = open(file, 'r', encoding='utf-8').read().lower()
def extract_text(filename):
	"""extract text from file and tokenize text"""
	#open and extract file contents
	with open(filename, "r", encoding = "utf-8") as myfile:
		raw_text = myfile.read()

	#tokenize raw text w/ nltk
	raw_tokens = nltk.word_tokenize(raw_text)

	#remove punctuation
	filtered1 = [t for t in raw_tokens if t.isalpha()]

	#make all text lowercase
	filtered2 = [r.lower() for r in filtered1]

	#remove stopwords (may ignore this step if necessary)
	stopwords = nltk.corpus.stopwords.words("english")
	filtered3 = [s for s in filtered2 if s not in stopwords]

	return filtered3


def text_to_idx_dict(tokens):
	"""assign word to index and vice versa"""
	idx_to_tok = {}
	tok_to_idx = {}

	for i, word in enumerate(tokens):
		idx_to_tok[i] = word
	# for j, word2 in enumerate(tokens):
		tok_to_idx[word] = i

	return idx_to_tok, tok_to_idx

def create_sequences(tokens, tok_to_idx, seq_len):
	"""create seq_len sized chunks of text"""
	x = []
	y = []

	for i in range(seq_len, len(tokens)):
		sequence = tokens[i - seq_len: i]
		sequence = [tok_to_idx[s] for s in sequence]

		target = tokens[i - seq_len]
		target = tok_to_idx[target]

		x.append(sequence)
		y.append(target)

	x = np.array(x)
	y = np.array(y)

	return x, y	


# #PROBABLY WILL NEED TO CHANGE THIS: JUST A BASE	
# class myRNN(nn.Module):
# 	"""sample RNN from assignment2.py"""
# 	def __init__(self, input_size, hidden_size, output_size):
# 		super(myRNN, self).__init__()

# 		# Put the declaration of the RNN network here
# 		self.hidden_size = hidden_size
# 		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
# 		self.i2o = nn.Linear(input_size + hidden_size, output_size)
# 		self.softmax = nn.LogSoftmax(dim = 1)

# 	def forward(self, input, hidden):
# 		# Put the computation for the forward pass here
# 		combined = torch.cat((input, hidden), 1) #
# 		hidden = self.i2h(combined) #

# 		output = self.i2o(combined)
# 		output = self.softmax(output)
# 		#hidden = 

# 		return output, hidden

# 	def initHidden(self):
# 		return (torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size))


# def training():
# 	pass


# def predict():
# 	pass


	
if __name__ == '__main__':
	tokens = extract_text("alice.txt")
	vocab_size = len(tokens)

	print(tokens[0:100])
	print(vocab_size)


	a, b = text_to_idx_dict(tokens)
	print(len(a))
	print(len(b))

	x, y = create_sequences(tokens, b, 5)
	print(x)
	print(y)