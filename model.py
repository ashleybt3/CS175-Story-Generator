# CS 175 Story Generator
# Tiffany Kong 93065152
# Henry Fong 83642159
# Ashley Teves 10177429

import os
import sys
import time
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
import string
from collections import Counter


import torch
import torch.nn as nn
import torch.nn.functional as F


# source: https://trungtran.io/2019/02/08/text-generation-with-pytorch/
# -- FLOW -- 
# * creating vocab dictionaries
# * padding and splitting into input/labels
# * one-hot encoding
# * defining the model
# * training the model
# * evaluating the model
# -----------

# sequence size: length of the sequence that we're feeding into the model
seq_size = 32
# mini-batch gradient descent: [32, 64 128]
batch_size = 16
embedding_size = 64
lstm_size = 64
gradients_norm = 5
initial_words=['and', 'then']
predict_top_k = 5
# checkpoint_path='checkpoint'

def get_data(filename, batch_size, seq_size):
    # get text from data and convert them into ints
    with open(filename, 'r') as f:
        text = f.read()
    tokens = text.split()
    # tokenizer to include: alphabet, period
    # r'[\w.\']+'
    tokenizer = RegexpTokenizer(r'[\w\']+|\.|\?|\,')
    words = tokenizer.tokenize(text.lower())
    print(words[:100])

    # count how many words there are
    word_counts = Counter(words)
    # sort by highest word count
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    # key: int, value: word
    int_to_word = {k: w for k, w in enumerate(sorted_words)}
    # key: word, value: int
    word_to_int = {w: k for k, w in int_to_word.items()}
    n_vocab = len(int_to_word)

    
    int_text = [word_to_int[w] for w in words]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_txt = int_text[:num_batches * batch_size * seq_size]
    out_txt = np.zeros_like(in_txt)
    out_txt[:-1] = in_txt[1:]
    out_txt[-1] = in_txt[0]

    in_txt = np.reshape(in_txt, (batch_size, -1))
    out_txt = np.reshape(out_txt, (batch_size, -1))
    # print(in_txt[:100])
    # print(out_txt[:100])
    return int_to_word, word_to_int, n_vocab, in_txt, out_txt



    
def get_batches(in_txt, out_txt, batch_size, seq_size):
    # creates batches
    n_batches = np.prod(in_txt.shape) // (seq_size * batch_size)
    for i in range(0, n_batches * seq_size, seq_size):
        yield in_txt[:, i:i+seq_size], out_txt[:, i:i+seq_size]


class RnnModel(nn.Module):
    def __init__(self,n_vocab, seq_size, embedding_size, lstm_size):
        super(RnnModel, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        # convert words to ints
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first = True)
        # fully connected layer
        self.dense = nn.Linear(lstm_size, n_vocab)

    # feed forward
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        out, state = self.lstm(embed, prev_state)
        logits = self.dense(out)

        return logits, state
    
    # calls this at the start of every epoch to initialize the right shape of the state
    # generates the hidden states to zoer that will be used in the forward pass
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

    
def get_loss_and_train(net, lr = 0.001):
    # goal: tweak and change the weights of model to try and minimize the loss function
    # Log Loss (Cross Entropy Loss)
    # criterion: shows predictions from model, the lower the better
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer
    # optimizer: updates the model in response to the output of the loss function
    # and tells you if it's moving in the right or wrong direction
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    return criterion, optimizer


def predict(device, net, words, n_vocab, word_to_int, int_to_word, top_k=5):
    
    # turns off gradients
    net.eval()

    # hidden state and cell states are initialized back to zero
    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[word_to_int[w]]]).long().to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])
    
    words.append(int_to_word[choice])
    for _ in range(100):
        ix = torch.tensor([[choice]]).long().to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_word[choice])

    print(' '.join(words))



def train_main(filename):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # int_to_word: int to word dictionary
    # word_to_intn: word to int dictionary
    # in_txt: textual input data, converted to a set of numbers
    # - could be embeddings or one-hot encodings
    # out_txt: ...
    int_to_word, word_to_int, n_vocab, in_txt, out_txt = get_data(filename, batch_size, seq_size)

    # create RNN Model and write that model to device
    net = RnnModel(n_vocab, seq_size, embedding_size, lstm_size)
    net = net.to(device)

    
    # criterion: shows predictions from model, the lower the better
    # optimizer: updates the model in response to the output of the loss function and tells you if it's moving in the right or wrong direction
    # learning rate: multiply the gradients to scale them; stpes to converge to an optimum, the smaller the better
    # loss function: measures how wrong the predictions are
    criterion, optimizer = get_loss_and_train(net, 0.01)

    iteration = 0
    # epoch: defines the number times that the learning algorithm will work through the entire training dataset
    # - epoch with one batch is batch gradient descent learning algorithm
    # - sizes = [10, 100, 500, 1000, etc.]
    # - number of epochs is the number of complete passes through the training dataset
    for epoch in range(51):
        # batch size (group size): hyperparameter that defines the number of samples to work through before updating
        # -  1 ≤ size of batch ≤ # of samples in the training dataset


        batches = get_batches(in_txt, out_txt, batch_size, seq_size)
        # cell state: long term mem capability that stores and loads information of not necessarily immediately previous events
        # - present in LSTMs
        # hidden state: working mem capability that carries info from immediately previous events and overwrites at every step uncontrollably
        # - present at RNNs and LSTMs
        state_h, state_c = net.zero_state(batch_size)


        state_h = state_h.to(device)
        state_c = state_c.to(device)
        # iterates over batch of samples, where one batch has the specified batch size number of samples
        for x, y in batches:
            iteration += 1
            # turns on gradients
            net.train()
            optimizer.zero_grad() # clears existing gradients from previous epoch

            x = torch.tensor(x).long().to(device)
            y = torch.tensor(y).long().to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))

            # gets prediction and finds the loss, the lower the better
            loss = criterion(logits.transpose(1,2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()
            loss.backward() # does backpropagation and calculates gradients
            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), gradients_norm)

            optimizer.step() # updates weights accordingly

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(epoch, 50),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 1000 == 0:
                predict(device, net, initial_words, n_vocab,
                        word_to_int, int_to_word, top_k=5)
                # torch.save(net.state_dict(),
                #            'checkpoint_pt/model-{}.pth'.format(iteration))



if __name__ == '__main__':
    train_main('alice.txt')
