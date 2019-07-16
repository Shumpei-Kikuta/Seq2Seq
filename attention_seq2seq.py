import matplotlib.pyplot as plt
import numpy as np
import random
 
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.utils import shuffle
from math import ceil
import sequence

from typing import List
from util import calc_accuracy


(x_train, y_train), (x_test, y_test) = sequence.load_data()
char_to_id, id_to_char = sequence.get_vocab()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_train = torch.tensor(x_train.copy()).to(device)
x_test = torch.tensor(x_test.copy()).to(device)
y_train = torch.tensor(y_train.copy()).to(device)
y_test = torch.tensor(y_test.copy()).to(device)

# hyper-parameter
MAX_INPUT = 7
MAX_OUTPUT = 5
HIDDEN_DIM = 32
EMB_DIM = 32
vocab_size = len(char_to_id)
LEARNING_RATE = 0.01
BATCH_SIZE = 100
EPOCH = 1000
batch_num = ceil(len(x_train) // BATCH_SIZE)

def train_loader(data: torch.tensor, target: torch.tensor, batch_size: int =100):
    input_batchs = []
    output_batchs = []
    
    for i in range(batch_num):
        if i == batch_num - 1:
            each_input_batchs =  data[i * batch_size:]
            each_output_batchs = target[i * batch_size:]
        else:
            each_input_batchs =  data[i * batch_size: (i + 1) * batch_size]
            each_output_batchs = target[i * batch_size:(i + 1) * batch_size]
        input_batchs.append(each_input_batchs)
        output_batchs.append(each_output_batchs)

    return input_batchs, output_batchs
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.emb_dim) # padding_idx=5 #num_embeddings: inputの系列の長さ
    # 単語の分散表現の初期化
#     self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight)) #今回はいらない    
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, indices, batch_size=100):
        # indices = tensor([batch_size, MAX_INPUT(7)])
        embedding = self.embedding(indices)
        if embedding.dim() == 2: #　バッチサイズが1の時3次元にしている
            embedding = torch.unsqueeze(embedding, 1)
        output, state = self.gru(embedding, torch.zeros(1, batch_size, self.hidden_dim).to(device)) #最初の入力は0ベクトル # state-> 最終層の状態(1, N, dim)、output->各層の出力(N, 7, dim)
        return output, state

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.emb_dim)
        self.gru = nn.GRU(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def forward(self, indices, init_hidden, encoder_output):
        """indices: [N, MAX_OUTPUT - 1], init_hidden: [1, N, hidden_dim], encoder_output: [N, MAX_INPUT, hidden_dim]"""
        embedding = self.embedding(indices)
        if embedding.dim() == 2: #バッチサイズが1の時3次元に変換
            embedding = torch.unsqueeze(embedding, 1)
        output, state = self.gru(embedding, init_hidden) #最初の入力は0ベクトル
        
        ##attention
#         print(output.size())
#         print(encoder_output.size())
        t1 = output * encoder_output
#         print(t1.size())
        s = torch.sum(t1, dim=2)
#         print(s.size())
        a = F.softmax(s, dim=1).unsqueeze(2)
#         print(a.size())
#         print(encoder_output.size())
#         print(torch.sum(a[0, :])) ## 1
        t2 = encoder_output * a
#         print(t2.size())
        context = torch.sum(t2, dim=1)
#         print(context.size())
#         print(output.size())
        context = context.unsqueeze(dim=1)
#         print(context.size())
        output = torch.cat((output, context), dim=2)
#         print(output.size())
        
        output = self.linear(output)
        return output, state
        
class Seq2seq:
    def __init__(self):
        self.encoder = Encoder(vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, emb_dim=EMB_DIM).to(device)
        self.decoder = Decoder(vocab_size=vocab_size, hidden_dim=HIDDEN_DIM, emb_dim=EMB_DIM).to(device)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, encoder_input, decoder_input):
        batch_size = encoder_input.size(0)
        encoder_output, encoder_hidden = self.encoder(encoder_input, batch_size) 
        
        source = decoder_input[:, :-1]
        target = decoder_input[:, 1:]
        
        # 1文字ずつ入力し、outputのcross entropyを計算する
        loss = 0
        target_length = target.size(1)
        source_length = source.size(1)
        decoder_output = np.zeros((batch_size, target_length))
        for i in range(source_length):
            decoder_result, _ = self.decoder(source[:, i], encoder_hidden, encoder_output)
            decoder_result = torch.squeeze(decoder_result)
            decoder_output[:, i] = np.argmax(decoder_result.cpu().detach().numpy(), axis=1)
            loss += self.criterion(decoder_result, target[:, i])
        return loss, decoder_output
        
class Trainer:
    def __init__(self, model):
        self.model = model
        self.encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=LEARNING_RATE)
        self.decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=LEARNING_RATE)
        
    def fit(self, X, Y):
        # 重みの初期化
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        self.train_loss, self.train_output =  self.model.forward(X, Y)

        # backward
        self.train_loss.backward()
        self.train_loss = self.train_loss.item()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return self.train_output, self.train_loss

        
model = Seq2seq()
trainer = Trainer(model)

print("Training..")

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

train_answer = y_train[:, 1:]
test_answer = y_test[:, 1:]

for e in range(EPOCH):
    train_loss = 0
    input_batchs, output_batchs = train_loader(x_train, y_train)

    for i in range(len(input_batchs)): # mini-batchごとに最適化
        input_batch = input_batchs[i].to(device)
        output_batch = output_batchs[i].to(device)
        train_output, each_train_loss = trainer.fit(input_batch, output_batch)
        train_loss += each_train_loss
        train_batch_outputs = np.concatenate([train_batch_outputs, train_output], axis=0) if i != 0 else train_output

    train_loss /= batch_num

    train_accuracy = calc_accuracy(train_batch_outputs, train_answer.cpu().numpy())

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_outputs = model.forward(x_test, y_test)
        
    test_loss = test_loss.item()
    test_losses.append(test_loss)
    
    test_accuracy = calc_accuracy(test_outputs, test_answer.cpu().numpy())
    test_accuracies.append(test_accuracy)
    
    print(train_accuracy, test_accuracy)
    print(train_loss, test_loss)