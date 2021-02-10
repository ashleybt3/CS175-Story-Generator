# CS 175 Story Generator
# Tiffany Kong 93065152
# Henry Fong 83642159
# Ashley Teves 10177429


import numpy as np
import nltk
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


seq_size=32
batch_size=16
embedding_size=64
lstm_size=64
gradients_norm=5,
initial_words=['there', 'was']
predict_top_k=5
checkpoint_path='checkpoint'

def get_data(filename, batch_size, seq_size):
    # get text from data and convert them into ints
    with open(filename, 'r', encoding = "utf-8") as f:
        text = f.read()
    #text = text.split()
    text = nltk.word_tokenize(text) #
    text = [t for t in text if t.isalpha() or t == "."]
    text = [t.lower() for t in text]

    word_counts = Counter(text)
    #print(word_counts)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_word = {k: w for k, w in enumerate(sorted_words)}
    word_to_int = {w: k for k, w in int_to_word.items()}

    n_vocab = len(int_to_word)

    
    int_text = [word_to_int[w] for w in text]
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
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first = True)
        #self.rnn = nn.RNN(n_vocab, lstm_size)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        out, state = self.lstm(embed, prev_state)
        logits = self.dense(out)

        return logits, state
    
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

    
def get_loss_and_train(net, lr = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)

    return criterion, optimizer


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).long().to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    print([int_to_vocab[c] for c in choices[0]])
    choice = np.random.choice(choices[0])
    print("the choice: ", choice)
    words.append(int_to_vocab[choice])
    for _ in range(100):
        ix = torch.tensor([[choice]]).long().to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    print("prediction: ")
    print(' '.join(words))


def train_main(filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    int_to_word, word_to_int, n_vocab, in_txt, out_txt = get_data(filename, batch_size, seq_size)

    net = RnnModel(n_vocab, seq_size, embedding_size, lstm_size)
    net = net.to(device)

    criterion, optimizer = get_loss_and_train(net, 0.01)

    iteration = 0

    for i in range(51):
        batches = get_batches(in_txt, out_txt, batch_size, seq_size)
        state_h, state_c = net.zero_state(batch_size)

        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:

            iteration += 1
            net.train()
            optimizer.zero_grad()

            x = torch.tensor(x).long().to(device)
            y = torch.tensor(y).long().to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))

            loss = criterion(logits.transpose(1,2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()
            loss.backward()
            # _ = torch.nn.utils.clip_grad_norm_(net.parameters(), gradients_norm)

            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(i, 100),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

    predict(device, net, initial_words, n_vocab, word_to_int, int_to_word, top_k=5)



if __name__ == '__main__':
    train_main('alice.txt')

