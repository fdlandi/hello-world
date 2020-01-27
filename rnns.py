# -*- coding: utf-8 -*-
"""RNNs.ipynb
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision

import numpy as np

class RNNCell(nn.Module):
  def __init__(self, input_size, output_size, hidden_size):
    super(RNNCell, self).__init__()
    self.h2h = nn.Linear(hidden_size, hidden_size)
    self.x2h = nn.Linear(input_size, hidden_size)
    self.h2y = nn.Linear(hidden_size, output_size)
    self.tanh = nn.Tanh()
    self._init_weights()

  def _init_weights(self):
    nn.init.eye_(self.h2h.weight)
    nn.init.constant_(self.h2h.bias, 0.)

  def forward(self, x, state):
    new_state = self.tanh(self.h2h(state) + self.x2h(x))
    y = self.h2y(new_state)
    return y, new_state

class RNN(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, device):
    super(RNN, self).__init__()
    self.cell = RNNCell(input_size, output_size, hidden_size)
    self.linear = nn.Linear(hidden_size, output_size)
    self.hidden_size = hidden_size
    self.device = device

  def forward(self, x):
    h_t = torch.zeros(x.shape[0], self.hidden_size).to(self.device)

    for idx in range(x.shape[1]):
      pix = x[:, idx, :]
      _, h_t = self.cell(pix, h_t)

    out = self.linear(h_t)
    return out

class LSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(LSTMCell, self).__init__()
    self.x2i = nn.Linear(input_size, hidden_size)
    self.x2o = nn.Linear(input_size, hidden_size)
    self.x2f = nn.Linear(input_size, hidden_size)
    self.x2g = nn.Linear(input_size, hidden_size)

    self.h2i = nn.Linear(hidden_size, hidden_size)
    self.h2o = nn.Linear(hidden_size, hidden_size)
    self.h2f = nn.Linear(hidden_size, hidden_size)
    self.h2g = nn.Linear(hidden_size, hidden_size)

    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self._init_weights()

  def _init_weights(self):
    nn.init.constant_(self.x2f.bias, 1.)
    nn.init.constant_(self.h2f.bias, 1.)

  def forward(self, x, h, c):
    input_gate = self.sigmoid(self.x2i(x) + self.h2i(h))
    g_gate = self.tanh(self.x2g(x) + self.h2g(h))
    forget_gate = self.sigmoid(self.x2f(x) + self.h2f(h))
    output_gate = self.sigmoid(self.x2o(x) + self.h2o(h))

    c_new = forget_gate * c + input_gate * g_gate
    h_new = output_gate * self.tanh(c_new)
    
    return h_new, c_new

class LSTM(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, device):
    super(LSTM, self).__init__()
    self.cell = LSTMCell(input_size, hidden_size)
    self.linear = nn.Linear(hidden_size, output_size)
    self.hidden_size = hidden_size
    self.device = device

  def forward(self, x):
    h_t = torch.zeros(x.shape[0], self.hidden_size).to(self.device)
    c_t = torch.zeros(x.shape[0], self.hidden_size).to(self.device)

    for idx in range(x.shape[1]):
      pix = x[:, idx, :]
      h_t, c_t = self.cell(pix, h_t, c_t)

    out = self.linear(h_t)
    return out

trainset = torchvision.datasets.MNIST('./data', train=True, download=True,
                                     transform=torchvision.transforms.ToTensor())
testset = torchvision.datasets.MNIST('./data', train=False, download=True,
                                     transform=torchvision.transforms.ToTensor())

batch_size = 100
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=4)
testloader = DataLoader(testset, shuffle=True, batch_size=batch_size, num_workers=4)

def Image2Seq(img):
  b = img.shape[0]
  return img.reshape(b, 28*28, 1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = RNN(1, 10, 128, device).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,
#                             momentum=0.9, nesterov=True)
# loss_fn = nn.CrossEntropyLoss()

model = LSTM(1, 10, 128, device).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,
                            momentum=0.9, nesterov=True)
loss_fn = nn.CrossEntropyLoss()

num_epoch = 10
for epoch in range(num_epoch):
  running_loss = 0.
  for i, data in enumerate(trainloader):
    img, labels = data
    labels = labels.cuda()
    seq = Image2Seq(img).cuda()

    optimizer.zero_grad()
    pred = model(seq)
    loss = loss_fn(pred, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print('Loss epoch {}: {:.4f}'.format(epoch, running_loss / len(trainloader)))
