# CS 175 Story Generator
# Tiffany Kong 93065152
# Henry Fong 83642159
# Ashley Teves 10177429

import os
import sys
import time
import numpy as np
import nltk
import string
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# source: https://trungtran.io/2019/02/08/text-generation-with-pytorch/

seq_size = 32
batch_size = 16
embedding_size = 64
# embedding_size = 32
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
    # words = [word.lower() for word in tokens if word.isalpha() or word == "."]
    words = text.lower().split()
    # print(words[:10])

    word_counts = Counter(words)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_word = {k: w for k, w in enumerate(sorted_words)}
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
    return int_to_word, word_to_int, n_vocab, in_txt, out_txt



    
def get_batches(in_txt, out_txt, batch_size, seq_size):
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
        #self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first = True)
        
        # Bi-LSTM
        # foward and backward
        self.lstm_cell_forward = nn.LSTMCell(embedding_size, embedding_size)
        self.lstm_cell_backward = nn.LSTMCell(embedding_size, embedding_size)

        # concatenated layer of forward and backward
        self.lstm_cell = nn.LSTMCell(embedding_size*2, embedding_size*2)

        self.dropout = nn.Dropout(.5)
        
        # linear layer
        self.dense = nn.Linear(lstm_size*2, n_vocab)

    # feed forward
    # def forward(self, x, prev_state):
    def forward(self, x):
        # init hidden states, cell states
        # Bi-LSTM
        # hs = [batch_size x embedding_size]
        # cs = [batch_size x embedding_size]
        hs_forward = torch.zeros(x.size(0), embedding_size)
        cs_forward = torch.zeros(x.size(0), embedding_size)
        hs_backward = torch.zeros(x.size(0), embedding_size)
        cs_backward = torch.zeros(x.size(0), embedding_size)
        
        # LSTM
        # hs = [batch_size x (embedding_size * 2)]
        # cs = [batch_size x (embedding_size * 2)]
        hs_lstm = torch.zeros(x.size(0), embedding_size * 2)
        cs_lstm = torch.zeros(x.size(0), embedding_size * 2)
        
        # Weights initialization
        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_backward)
        torch.nn.init.kaiming_normal_(cs_backward)
        torch.nn.init.kaiming_normal_(hs_lstm)
        torch.nn.init.kaiming_normal_(cs_lstm)
        
        out = self.embedding(x)
        
        # Prepare the shape for LSTM Cells
        out = out.view(self.seq_size, x.size(0), -1)
        
        forward = []
        backward = []
        
        # Unfolding Bi-LSTM
        # Forward
        for i in range(self.seq_size):
            hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
            hs_forward = self.dropout(hs_forward)
            cs_forward = self.dropout(cs_forward)
            forward.append(hs_forward)
            
         # Backward
        for i in reversed(range(self.seq_size)):
            hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
            hs_backward = self.dropout(hs_backward)
            cs_backward = self.dropout(cs_backward)
            backward.append(hs_backward)
            
         # LSTM
        for fwd, bwd in zip(forward, backward):
            input_tensor = torch.cat((fwd, bwd), 1)
            hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))
        
         # Last hidden state is passed through a linear layer
        out = self.dense(hs_lstm)

        return out
    
    #  calls this at the start of every epoch to initialize the right shape of the state
    ### not used for lstm version. States are initialized at the beginning of forward
    # def zero_state(self, batch_size):
    #     return (torch.zeros(1, batch_size, self.lstm_size),
    #             torch.zeros(1, batch_size, self.lstm_size))

    
def get_loss_and_train(net, lr = 0.001):
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)

    return criterion, optimizer


def predict(device, net, words, n_vocab, word_to_int, int_to_word, top_k=5):
    net.eval()

    # state_h, state_c = net.zero_state(1)
    # state_h = state_h.to(device)
    # state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[word_to_int[w]]]).to(device)
        # output, (state_h, state_c) = net(ix, (state_h, state_c))
        output= net(ix)
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_word[choice])
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_word[choice])

    print(' '.join(words))


def train_main(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_word, word_to_int, n_vocab, in_txt, out_txt = get_data(filename, batch_size, seq_size)

    net = RnnModel(n_vocab, seq_size, embedding_size, lstm_size)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train(net, 0.01)

    iteration = 0

    # originally 50
    for i in range(50):
        batches = get_batches(in_txt, out_txt, batch_size, seq_size)
        # state_h, state_c = net.zero_state(batch_size)

        # state_h = state_h.to(device)
        # state_c = state_c.to(device)
        for x, y in batches:

            iteration += 1
            net.train()
            optimizer.zero_grad()

            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            # logits, (state_h, state_c) = net(x, (state_h, state_c))
            logits= net(x)
            
            loss = criterion(logits.transpose(1,2), y)

            # state_h = state_h.detach()
            # state_c = state_c.detach()

            loss_value = loss.item()
            loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(net.parameters(), gradients_norm)

            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(i, 10),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 1000 == 0:
                predict(device, net, initial_words, n_vocab,
                        word_to_int, int_to_word, top_k=5)
                # torch.save(net.state_dict(),
                #            'checkpoint_pt/model-{}.pth'.format(iteration))



if __name__ == '__main__':
    train_main('alice.txt')
