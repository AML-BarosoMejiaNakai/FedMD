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
    N_parties = CONF_MODELS["N_parties"]
    N_samples_per_class = CONF_MODELS["N_samples_per_class"]

    N_rounds = CONF_MODELS["N_rounds"]
    N_alignment = CONF_MODELS["N_alignment"]
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