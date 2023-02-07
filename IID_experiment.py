"""
Implementation of FedMD based on the TensorFlow implementation of diogenes0319/FedMD_clean
Non-IID scenario where models need to correctly predict the super-class belonging to an image
by only being aware of one sub-class per super-class.
"""
import argparse
import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from constants import * 
from models.client import cnn_2layers, cnn_3layers
from models.ResNet20 import resnet20
import data.data_utils as data_utils
import data.CIFAR as CIFAR
import training.model_trainers as model_trainers
from training.FedMD import FedMD
from wandb_utils import *
import training.trainer_utils as trainer_utils

from PIL import Image
from tqdm import tqdm
import wandb

CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layers, 
                    "3_layer_CNN": cnn_3layers,
                    "ResNet20": resnet20} 

def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('-wandb', metavar='wandb', 
                        help='the wandb API key.'
                       )
    parser.add_argument('-run_id', metavar='run_id', 
                        help='the wandb run id to resume.'
                       )
    parser.add_argument('-restore_path', metavar='restore_path',
                        help='the wandb project path to restore files.'
                       )
    args = None
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
    return args

def main():
    args = parseArg()    
    
    wandb_api_key = args.wandb
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_MODE"] = "online" if args.wandb else "offline"
    ckpt_path = 'ckpt'
    paths = [ckpt_path, f"{ckpt_path}/ub"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

    run_id = args.run_id
    restore_path = args.restore_path

    model_config = CONF_MODELS_BALANCED["models"]
    pre_train_params = CONF_MODELS_BALANCED["pre_train_params"]
    model_saved_dir = CONF_MODELS_BALANCED["model_saved_dir"]
    model_saved_names = CONF_MODELS_BALANCED["model_saved_names"]
    is_early_stopping = CONF_MODELS_BALANCED["early_stopping"]
    public_classes = CONF_MODELS_BALANCED["public_classes"]
    private_classes = CONF_MODELS_BALANCED["private_classes"]
    n_classes = len(public_classes) + len(private_classes)

    N_agents = CONF_MODELS_BALANCED["N_agents"]
    N_samples_per_class = CONF_MODELS_BALANCED["N_samples_per_class"]

    N_rounds = CONF_MODELS_BALANCED["N_rounds"]
    N_subset = CONF_MODELS_BALANCED["N_subset"]
    N_private_training_round = CONF_MODELS_BALANCED["N_private_training_round"]
    private_training_batchsize = CONF_MODELS_BALANCED["private_training_batchsize"]
    N_logits_matching_round = CONF_MODELS_BALANCED["N_logits_matching_round"]
    logits_matching_batchsize = CONF_MODELS_BALANCED["logits_matching_batchsize"]


    result_save_dir = CONF_MODELS_BALANCED["result_save_dir"]

    # This dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per 
    # class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a 
    # "coarse" label (the superclass to which it belongs).

    print ("=== LOADING CIFAR 10 AND CIFAR 100 ===")

    train_cifar10, test_cifar10   = CIFAR.load_CIFAR10() # train_cifar10 = public_dataset
    train_cifar100, test_cifar100 = CIFAR.load_CIFAR100()

    print ("=== Generating class subsets ===")

    # Create a subset of dataset containing only the private_classes to use
    private_train_dataset = data_utils.generate_class_subset(train_cifar100, private_classes)
    private_test_dataset  = data_utils.generate_class_subset(test_cifar100,  private_classes)

    # Map the private classes to the range [10,16) 
    for index, cls_ in enumerate(private_classes):        
        private_train_dataset.targets[private_train_dataset.targets == cls_] = index + len(public_classes)
        private_test_dataset.targets[private_test_dataset.targets == cls_] = index + len(public_classes)
    del index, cls_
    mod_private_classes = torch.arange(len(private_classes)) + len(public_classes)

    # Split the train dataset among the agent, with N_samples_per_class images per class per agent
    print (f"=== Splitting private dataset for the {N_agents} agents ===")
    private_data, total_private_data = data_utils.split_dataset(private_train_dataset, N_agents, N_samples_per_class, classes_in_use=mod_private_classes, seed=SEED)

    # Create subset of the test dataset based on remapped private classes
    private_test_dataset = data_utils.generate_class_subset(private_test_dataset, mod_private_classes)

    run, job_id, resumed = init_wandb(run_id=run_id, config=CONF_MODELS_BALANCED)

    # Creation of agent models
    agents = []
    for i, item in enumerate(model_config):
        model_name = item["model_type"]
        model_params = item["params"]
        train_params = item["train_params"]
        tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes, 
                                            input_shape=(3,32,32),
                                            **model_params)
        print("model {0} : {1}".format(i, model_saved_names[i]))
        agents.append({"model": tmp, "train_params": train_params})
        
        del model_name, model_params, tmp
    # end for
    
    # Initial transfer learning training on the public dataset
    # For each model a checkpoint will be uploaded if it exists, otherwise it will perform training  
    for i, agent in enumerate(agents):
        loaded = load_checkpoint(f"{ckpt_path}/{model_saved_names[i]}_initial_pub.pt", agent["model"], restore_path)
        if not loaded:
            optimizer = trainer_utils.load_optimizer(agent["model"], agent["train_params"])#optim.Adam(agent.parameters(), lr = LR)
            loss = nn.CrossEntropyLoss()
            print(f"===== TRAINING {model_saved_names[i]} =====")
            accuracies = model_trainers.train_model(network=agent["model"], 
                dataset=train_cifar10, 
                test_dataset=test_cifar10, 
                loss_fn=loss, 
                optimizer=optimizer, 
                batch_size=128, 
                num_epochs=20, 
                returnAcc=True,
                early_stop=trainer_utils.EarlyStop(pre_train_params["patience"], pre_train_params["min_delta"])
            )
            wandb.run.summary[f"{model_saved_names[i]}_initial_pub_test_acc"] = accuracies[-1]["test_accuracy"]
            
            torch.save(agent["model"].state_dict(), f'{ckpt_path}/{model_saved_names[i]}_initial_pub.pt')
            wandb.save(f'{ckpt_path}/{model_saved_names[i]}_initial_pub.pt')
        else:
            test_acc = model_trainers.test_network(network=agent["model"], test_dataset=test_cifar10, batch_size=128)
            wandb.run.summary[f"{model_saved_names[i]}_initial_pub_test_acc"] = test_acc
        # end if load ckpt
    # end for

    # Creation of the FedMD algorithm
    # Performs initial private training + upper bound calculation
    fedmd = FedMD(agents, model_saved_names,
        public_dataset=train_cifar10, 
        private_data=private_data, 
        total_private_data=total_private_data, 
        private_test_data=private_test_dataset,
        N_rounds=N_rounds,
        N_subset=N_subset,
        N_logits_matching_round=N_logits_matching_round,
        logits_matching_batchsize=logits_matching_batchsize,
        N_private_training_round=N_private_training_round,
        private_training_batchsize=private_training_batchsize,
        restore_path=restore_path)
    
    collab = fedmd.collaborative_training()

    wandb.finish()
# end main

if __name__ == '__main__':
    main()
