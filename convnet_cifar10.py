# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as T

transform = T.Compose([
                       T.ToTensor(),
                       T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                      train=True, 
                                      download=True,
                                      transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data',
                                     train=False, 
                                     download=True,
                                     transform=transform)

batch_size = 100

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         num_workers=4,
                                         shuffle=True)

def imshow(img):
  np_img = img.numpy()
  plt.imshow(np.transpose(np_img, (1, 2, 0)))
  plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images[:10] / 2 + 0.5))
print([trainset.classes[i] for i in labels[:10]])

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

print('Using device: {}'.format(device))

in_features, output_size = 32*32, 10

model = nn.Sequential(
    # primo blocco
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    # secondo blocco
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),
    # fc
    nn.Flatten(),
    nn.Linear(8*8*64, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1024, output_size)
)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

def predict(model, data, loss_fn):
  images, labels = data

  images = images.to(device)
  labels = labels.to(device)
  
  # forward pass: model makes prediction
  out = model(images)

  # computes loss and accuracy
  loss = loss_fn(out, labels)
  _, pred = torch.max(out, 1)
  total = labels.shape[0]
  correct = (pred == labels).sum().item()

  # returns statistics
  return loss, total, correct

def train_epoch(model, optimizer, trainloader):
  # reset epoch statistics
  running_loss = 0.0
  total, correct = 0.0, 0.0

  for i, data in enumerate(trainloader, 1):
    # training step
    optimizer.zero_grad()
    loss, total_step, correct_step = predict(model, data, loss_fn)
    loss.backward()
    optimizer.step()

    # update statistics
    total += total_step
    correct += correct_step
    running_loss += loss.item()

    # print info
    if i % log_every == 0:
      print('Iter {} - Loss: {:.4f}'.format(i, running_loss/log_every))
      running_loss = 0.0

  return 100*correct/total

def evaluate(model, testloader):
  running_loss = 0.0
  total, correct = 0.0, 0.0

  for i, data in enumerate(testloader, 1):
    # forward function
    loss, total_step, correct_step = predict(model, data, loss_fn)

    # update statistics
    total += total_step
    correct += correct_step
    running_loss += loss.item()

  # print info
  print('Test loss: {:.4f}'.format(running_loss/i))
  running_loss = 0.0

  return 100*correct/total

learning_rate = 1e-3
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
#                            momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epoch = 10
log_every = 100

model.train()
for epoch in range(num_epoch):
  print('Starting Epoch {}/{}...'.format(epoch+1, num_epoch))
  accuracy = train_epoch(model, optimizer, trainloader)
  print('Epoch {} - Accuracy: {:.2f}%'.format(epoch+1, accuracy))
  
print('Finished training')

model.eval()
with torch.no_grad():
  accuracy = evaluate(model, testloader)
  print('Test accuracy: {:.2f}%'.format(accuracy))

dataiter = iter(testloader)
images, labels = dataiter.next()

images = images.to(device)
labels = labels.to(device)

model.eval()
out = model(images)
_, pred = torch.max(out, 1)

imshow(torchvision.utils.make_grid(images.cpu()[:10] / 2 + 0.5))
print('predictions:\t {}'.format([testset.classes[p] for p in pred[:10]]))
print('ground truth:\t {}'.format([testset.classes[l] for l in labels[:10]]))
