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
	"""
	create two dictionaries:
	(1) maps word to index
	(2) maps index to word
	"""
	idx_to_tok = {}
	tok_to_idx = {}

	for i, word in enumerate(tokens):
		idx_to_tok[i] = word
	# for j, word2 in enumerate(tokens):
		tok_to_idx[word] = i

	return idx_to_tok, tok_to_idx

def create_sequences(tokens, tok_to_idx, seq_len):
	"""create seq_len sized chunks of text (simulate sentence creating)"""
	seq = []
	targ = []

	for i in range(seq_len, len(tokens)):
		sequence = tokens[i - seq_len: i]
		sequence = [tok_to_idx[s] for s in sequence]

		target = tokens[i - seq_len]
		target = tok_to_idx[target]

		seq.append(sequence)
		targ.append(target)

	seq = np.array(seq)
	targ = np.array(targ)

	return seq, targ


class modelRNN(nn.Module):
	"""sample RNN from assignment2.py"""
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		super(modelRNN, self).__init__()

		self.hidden_size = hidden_size
		self.n_layers = n_layers

		#layer declaration
		self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first = True)
		self.lin = nn.Linear(hidden_size, output_size)

	def forward(self, input):
		batch_size = input.size(0)
		hidden = self.initHidden(batch_size)

		output, hidden = self.rnn(input, hidden)
		output = output.contiguous().view(-1, self.hidden_size) ##
		output = self.lin(out)

		return output, hidden

	def initHidden(self, batch_size):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size)


def training(model, n_epochs, learn_rate):
	#number of epochs (# times model go thru training data)
	#learning rate (rate at which model updates weights when back propogation done)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr= learn_rate)
	for epoch in range(n_epochs):
		pass


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