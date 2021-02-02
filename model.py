# CS 175 Story Generator
# Tiffany Kong 93065152
# Henry Fong 83642159
# Ashley Teves 10177429

import numpy as np 
import nltk

import torch
from torch import nn

from sklearn.preprocessing import OneHotEncoder
"""
general flow:
* open & extract file contents #
* tokenize text #
* convert vocab to numeric format #
* build sequences #
* encoding #
* build model #
* train model #
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
	"""create seq_len sized chunks of text & binary encode them"""
	sequences = []
	targets = []
	max_wordlen = len(max(tokens, key = len))

	for i in range(seq_len, len(tokens)):
		seq = tokens[i - seq_len: i]
		targ = tokens[i - seq_len]
		
		sequences.append(seq)
		targets.append(targ)

	#one-hot encoding (more like one hot mess)
	x = np.zeros((len(sequences), seq_len, len(tokens)), dtype = np.float32)
	y = np.zeros((len(sequences), len(tokens)), dtype = np.float32)

	for i, wordchunk in enumerate(sequences):
		for j, word in enumerate(wordchunk):
			x[i, j, tok_to_idx[word]] = 1
		y[i, tok_to_idx[targets[i]]] = 1

	return x, y



class modelRNN(nn.Module):
	"""simple RNN w/ a linear layer"""
	def __init__(self, input_size, hidden_size, output_size, n_layers):
		super(modelRNN, self).__init__()

		#param definition
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		#layer declaration
		self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first = True)
		self.lin = nn.Linear(hidden_size, output_size)

	def forward(self, x_input):
		batch_size = x_input.size(0)
		#batch_size = x_input.size(0)
		hidden = self.initHidden(batch_size)

		output, hidden = self.rnn(x_input, hidden)
		output = output.contiguous().view(-1, self.hidden_size) ##
		output = self.lin(output)

		return output, hidden

	def initHidden(self, batch_size):
		return torch.zeros(self.n_layers, batch_size, self.hidden_size)
		#return torch.zeros(self.n_layers, self.hidden_size)

def training(model, sequence, target, n_epochs, learn_rate):
	#number of epochs (# times model go thru training data)
	#learning rate (rate at which model updates weights when back propogation done)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr= learn_rate)

	for epoch in range(n_epochs):
		output, hidden = model.forward(sequence) #forward
		#loss = criterion(output, target.view())
		# loss.backward()
		# optimizer.step()

		# if epoch%10 == 0:
		# 	print("Epoch: {}/{}------Loss: {%.3f}".format(epoch, n_epochs, loss.item()))

# def predict(model, idx_to_text, n_words):
# 	pass


	
if __name__ == '__main__':
	tokens = extract_text("alice.txt")
	# tok_size = len(tokens)

	tokens = tokens[0:100]
	tok_size = len(tokens)

	print("Sample of First 100 Tokens:\n", tokens[0:100])
	print("Total Tokens: ", tok_size)


	a, b = text_to_idx_dict(tokens)
	# print(a)
	# print(b)
	print(len(a)) #idx_to_tok
	print(len(b)) #tok_to_idx

	vocab_size = len(b)
	print("Size of Vocab: ", vocab_size)
	print(b)

	x, y = create_sequences(tokens, b, 5)
	print(x[0:10]) #seq
	print(y[0:10]) #targ

	input_seq = torch.from_numpy(x)
	targ_seq = torch.Tensor(y)

	#input_size, hidden_size, output_size, n_layers
	n_hidden = 12
	n_layers = 2
	rnnModel = modelRNN(tok_size, n_hidden, tok_size, 1)

	print(rnnModel)

	n_epochs = 100
	learning_rate = 0.01
	training(rnnModel, input_seq, targ_seq, n_epochs, learning_rate)