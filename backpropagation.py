# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import randn

n, d_in, d_out = 64, 100, 10

x = randn(n, d_in)
y = randn(n, d_out)

w = randn(d_in, d_out)

lr = 1e-4  # uguale a scrivere 0.0001 (1 * 10^-4)
num_iter = 2000

# update di gradient descent

for t in range(num_iter):
  # forward - trovare le predizioni
  y_pred = x.dot(w)
  loss = np.square(y_pred - y).sum()
  if t % 100 == 0:
    print('{} - loss: {:.4f}'.format(t, loss))

  # backward - backpropagation
  grad_loss = 1.0
  grad_y_pred = grad_loss * 2.0 * (y_pred - y)
  grad_w = x.T.dot(grad_y_pred)

  # update dei parametri
  w -= lr*grad_w
