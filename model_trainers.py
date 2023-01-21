import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from constants import *

import torchvision

def train_model(network, dataset, test_dataset, loss_fn, optimizer, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, scheduler = None, returnAcc = False):
  # TODO: Implement Accuracy calculation for both validation and training. Also implement early stopping.
  # returned accuracy should be a list where accuracy = [training_accuracy, validation_accuracy]
  train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
  # By default, everything is loaded to cpu
  net = network.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda

  cudnn.benchmark # Calling this optimizes runtime

  current_step = 0
  # Start iterating over the epochs
  # training, validation history
  if returnAcc:
    accuracy = []
  for epoch in range(num_epochs):
    running_corrects = 0
    if scheduler is not None:
      print('Starting epoch {}/{} LR={}'.format(epoch+1, num_epochs, scheduler.get_lr()))
    else:
      print('Starting epoch {}/{}'.format(epoch+1, num_epochs))

    i = 0
    # Iterate over the dataset
    for images, labels in train_dataloader:
      # Bring data over the device of choice
      images = images.to(DEVICE)
      labels = labels.to(DEVICE)

      net.train() # Sets module in training mode

      # PyTorch, by default, accumulates gradients after each backward pass
      # We need to manually set the gradients to zero before starting a new iteration
      optimizer.zero_grad() # Zero-ing the gradients

      # Forward pass to the network
      outputs = net(images)

      # Compute loss based on output and ground truth
      loss = loss_fn(outputs, labels)

      # Log loss
      if current_step % LOG_FREQUENCY == 0:
        print('Step {}, Loss {}'.format(current_step, loss.item()))
      _, preds = torch.max(outputs.data, 1)
      # Update Corrects
      running_corrects += torch.sum(preds == labels.data).data.item()

      # Compute gradients for each layer and update weights
      loss.backward()  # backward pass: computes gradients
      optimizer.step() # update weights based on accumulated gradients
      if scheduler is not None:
        scheduler.step(epoch + i / len(train_dataloader))
      i += 1

      current_step += 1
    # end for dataset iteration

    # Step the scheduler
    # scheduler.step()
    if returnAcc:
      train_accuracy = running_corrects / len(dataset)
      test_accuracy = test_network(net, test_dataset, batch_size=BATCH_SIZE)
      accuracy.append({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})

  #end of epoch
  if returnAcc:
    return accuracy
# end train_model

def test_network(network, test_dataset, batch_size=BATCH_SIZE):
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
  net = network.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
  net.train(False) # Set Network to evaluation mode

  running_corrects = 0
  for images, labels in test_dataloader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Forward Pass
    outputs = net(images)

    # Get predictions
    _, preds = torch.max(outputs.data, 1)

    # Update Corrects
    running_corrects += torch.sum(preds == labels.data).data.item()

  # Calculate Accuracy
  accuracy = running_corrects / float(len(test_dataset))
  return accuracy