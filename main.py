import argparse
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision.models import alexnet

from constants import * 
from client import cnn_2layers, cnn_3layers
from ResNet20 import resnet20
import CIFAR
import model_trainers
from FedMD import FedMD

from PIL import Image
from tqdm import tqdm

CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layers, 
                    "3_layer_CNN": cnn_3layers,
                    "ResNet20": resnet20} 


if __name__ == '__main__':
    model_config = CONF_MODELS["models"]
    pre_train_params = CONF_MODELS["pre_train_params"]
    model_saved_dir = CONF_MODELS["model_saved_dir"]
    model_saved_names = CONF_MODELS["model_saved_names"]
    is_early_stopping = CONF_MODELS["early_stopping"]
    public_classes = CONF_MODELS["public_classes"]
    private_classes = CONF_MODELS["private_classes"]
    n_classes = len(public_classes) + len(private_classes)

    emnist_data_dir = CONF_MODELS["EMNIST_dir"]    
    N_agents = CONF_MODELS["N_agents"]
    N_samples_per_class = CONF_MODELS["N_samples_per_class"]

    N_rounds = CONF_MODELS["N_rounds"]
    N_subset = CONF_MODELS["N_subset"]
    N_private_training_round = CONF_MODELS["N_private_training_round"]
    private_training_batchsize = CONF_MODELS["private_training_batchsize"]
    N_logits_matching_round = CONF_MODELS["N_logits_matching_round"]
    logits_matching_batchsize = CONF_MODELS["logits_matching_batchsize"]


    result_save_dir = CONF_MODELS["result_save_dir"]

    # This dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per 
    # class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a 
    # "coarse" label (the superclass to which it belongs).
    # Define transforms for training phase

    # random crop, random horizontal flip, per-pixel normalization 

    train_cifar10, test_cifar10   = CIFAR.load_CIFAR10()
    train_cifar100, test_cifar100 = CIFAR.load_CIFAR100()

    private_train_dataset = CIFAR.generate_class_subset(train_cifar100, private_classes)
    private_test_dataset  = CIFAR.generate_class_subset(test_cifar100,  private_classes)

    private_data, total_private_data = CIFAR.split_dataset(private_train_dataset, N_agents, N_samples_per_class, private_classes)

    agents = []
    for i, item in enumerate(model_config):
        model_name = item["model_type"]
        model_params = item["params"]
        tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes, 
                                            input_shape=(32,32,3),
                                            **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        agents.append(tmp)
        
        del model_name, model_params, tmp
    #END FOR LOOP
    
    for agent in agents:
        optimizer = optim.Adam(agent.parameters(), lr = 1e-3)
        loss = nn.CrossEntropyLoss()
        model_trainers.train_model(agent, train_cifar10, test_cifar10, loss_fn=loss, optimizer=optimizer, batch_size=128, num_epochs=20)
    
    fedmd = FedMD(agents, 
        public_dataset=train_cifar10, 
        private_data=private_data, 
        total_private_data=total_private_data, 
        private_test_data=private_test_dataset,
        N_rounds=N_rounds,
        N_subset=N_subset,
        N_logits_matching_round=N_logits_matching_round,
        logits_matching_batchsize=logits_matching_batchsize,
        N_private_training_round=N_private_training_round,
        private_training_batchsize=private_training_batchsize)
    
    collab = fedmd.collaborative_training()