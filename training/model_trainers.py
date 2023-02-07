import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from training.trainer_utils import EarlyStop
from torch.backends import cudnn
from constants import *

def train_model(network, dataset, loss_fn, optimizer, test_dataset=None, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, scheduler = None, log_frequency=LOG_FREQUENCY, returnAcc = False, early_stop=None):
    # TODO: Implement Accuracy calculation for both validation and training. Also implement early stopping.
    # returned accuracy should be a list where accuracy = [training_accuracy, validation_accuracy]
    drop_last = len(dataset) // batch_size > 1
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
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

        total_loss = 0
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
            total_loss += loss

            # Log loss
            if current_step % log_frequency == 0:
                print('Step {}, Loss {}'.format(current_step, loss.item()))
            _, preds = torch.max(outputs.data, 1)
            # Update Corrects
            if returnAcc:
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
            
            # Early stopping
            if early_stop is not None and early_stop(total_loss):             
                if returnAcc:
                    return accuracy
                else:
                    return
        # end of epoch (all batches analyzed)
        if returnAcc:
            train_accuracy = running_corrects / len(dataset)
            test_accuracy = test_network(net, test_dataset, batch_size=batch_size)
            accuracy.append({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})
    #end of epochs
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
        # Sample 1 -> [2.3, 4.1, 4.3, ..., ]
        # Sample .. ->[3.1, 1.3, 2.4, ..., ]

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(len(test_dataset))
    return accuracy


def run_dataset(network, dataset, batch_size=BATCH_SIZE):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    net = network.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    net.train(False) # Set Network to evaluation mode

    total_values = torch.tensor([])
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = net(images)
        # Sample 1 -> [2.3, 4.1, 4.3, ..., ]
        # Sample .. ->[3.1, 1.3, 2.4, ..., ]

        # Get predictions
        total_values = torch.cat((total_values.to(DEVICE), outputs.data))

    # Calculate Accuracy
    return total_values
