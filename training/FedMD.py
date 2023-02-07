import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
from training.model_trainers import *
from training.trainer_utils import *
from data.data_utils import stratified_sampling
import wandb
import wandb_utils
from constants import *

class FedMD:
    """
    Class that implements the collaborative training of FedMD.

    Parameters:
        agents: Array of dict {model: torch.Model, train_params: dict of train parameters}
        model_saved_names: Names of the models
        public_dataset: Dataset to use to perform knowledge distillation
        private_data: Array of dataset Subsets used for each agent
        total_private_data: dataset Subset containing all of the private_data data
        N_subset: Sample size of the public_dataset for the aligment data on each round
        N_round: Number of rounds to run the collaborative training
        N_logits_matching_round: Number of epochs for knowledge distillation training
        logits_matching_batchsize: Batch size for knowledge distillation training
        N_private_training_round: Number of epochs for private Revisit phase
        private_training_batchsize: Batch size for private Revisit phase
        restore_path: wandb run path from where to restore checkpoint models
    """
    def __init__(
        self,
        agents,
        model_saved_names,
        public_dataset,
        private_data,
        total_private_data,
        private_test_data,
        N_subset,
        N_rounds,
        N_logits_matching_round,
        logits_matching_batchsize,
        N_private_training_round,
        private_training_batchsize,
        restore_path=None
    ):

        self.N_agents = len(agents)
        self.model_saved_names = model_saved_names
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_subset = N_subset

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize

        self.collaborative_agents = []
        self.init_result = {}

        # Initial training round on private round
        # For each model a checkpoint will be uploaded if it exists, otherwise it will perform training  
        print("Start model initialization: ")
        for i in range(self.N_agents):
            print("Model ", self.model_saved_names[i])
            model_A = copy.deepcopy(agents[i]["model"])

            if not wandb_utils.load_checkpoint(f"ckpt/{self.model_saved_names[i]}_initial_pri.pt", model_A, restore_path):
                model_A.load_state_dict(agents[i]["model"].state_dict())
                optimizer = load_optimizer(model_A, agents[i]["train_params"])
                loss = nn.CrossEntropyLoss()
                early_stopping = EarlyStop(patience=10, min_delta=0.01)

                print("start full stack training ... ")

                accuracy = train_model(
                    network=model_A,
                    dataset=private_data[i],
                    test_dataset=private_test_data,
                    loss_fn=loss,
                    optimizer=optimizer,
                    early_stop=early_stopping,
                    batch_size=32,
                    num_epochs=25,
                    log_frequency=10,
                    returnAcc=True,
                )

                torch.save(model_A.state_dict(), f'ckpt/{model_saved_names[i]}_initial_pri.pt')
                wandb.save(f'ckpt/{model_saved_names[i]}_initial_pri.pt')
                last_test_acc = accuracy[-1]
                wandb.run.summary[f"{model_saved_names[i]}_initial_test_acc"] = last_test_acc["test_accuracy"]
                self.init_result[f"{model_saved_names[i]}_initial_test_acc"] = last_test_acc["test_accuracy"]
                print(f"Full stack training done. Accuracy: {last_test_acc['test_accuracy']}")
            else:
                test_acc = test_network(model_A, private_test_data, 32)
                wandb.run.summary[f"{model_saved_names[i]}_initial_test_acc"] = test_acc
                self.init_result[f"{model_saved_names[i]}_initial_test_acc"] = test_acc
            # end if load_checkpoint

            self.collaborative_agents.append({
                "model_logits": model_A,
                "model_classifier": model_A,
                "model_weights": model_A.state_dict(),
                "train_params": agents[i]["train_params"]
            })

            del model_A
        # end for

        # Perform calculation of the upper bound accuracy, training the initial 
        #   public trained models on the whole of the data
        # For each model a checkpoint will be uploaded if it exists, otherwise it will perform training  
        print("Calculate the theoretical upper bounds for participants: ")
        self.upper_bounds = []
        self.pooled_train_result = {}
        for i, agent in enumerate(agents):
            model = agent["model"]
            print(f"UB - Model {self.model_saved_names[i]}")
            model_ub = copy.deepcopy(model)
            if not wandb_utils.load_checkpoint(f"ckpt/ub/{self.model_saved_names[i]}_ub.pt", model_ub, restore_path):
                model_ub.load_state_dict(model.state_dict())
                optimizer = load_optimizer(model_ub, agent["train_params"])
                loss = nn.CrossEntropyLoss()
                early_stopping = EarlyStop(patience=10, min_delta=0.01)

                accuracy = train_model(
                    network=model_ub,
                    dataset=total_private_data,
                    test_dataset=private_test_data,
                    loss_fn=loss,
                    optimizer=optimizer,
                    early_stop=early_stopping,
                    batch_size=BATCH_SIZE,
                    num_epochs=50,
                    log_frequency=100,
                    returnAcc=True,
                )

                torch.save(model_ub.state_dict(), f'ckpt/ub/{model_saved_names[i]}_ub.pt')
                wandb.save(f'ckpt/ub/{model_saved_names[i]}_ub.pt')
                last_acc = accuracy[-1]["test_accuracy"]
            else:
                last_acc = test_network(model_ub, private_test_data, 32)
            # end if load ckpt
            wandb.run.summary[f"{model_saved_names[i]}_ub_test_acc"] = last_acc

            self.upper_bounds.append(last_acc)
            self.pooled_train_result[f"{model_saved_names[i]}_ub_test_acc"] = last_acc

            del model_ub
        # end for
        print("The upper bounds are:", self.upper_bounds)
    # end __init__

    def collaborative_training(self):
        """
        Collaborative training following the FedMD algorithm loop
        For each round:
            Communicate: Each model computes the class scores on the alignment data 
            Aggregate: The class scores are averaged
            Distribute: Each agent downloads the averaged consensus
            Digest: Each agent trains its model for a few epochs to approach the score consensus
            Revisit: Each agent trains on its private dataset for a few epochs
        """
        # Start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_agents)}
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = stratified_sampling(self.public_dataset, self.N_subset)

            print(f"Round {r}/{self.N_rounds}")

            # Communicate phase
            print("update logits ... ")
            logits = 0
            for agent in self.collaborative_agents:
                agent["model_logits"].load_state_dict(agent["model_weights"])
                model_logits = run_dataset(agent["model_logits"], alignment_data)
                logits += model_logits.to('cpu')
            # Aggregate phase
            logits /= self.N_agents
            
            print("test performance ... ")
            performances = {}
            for index, agent in enumerate(self.collaborative_agents):
                accuracy = test_network(network=agent["model_classifier"], test_dataset=self.private_test_data)

                print(f"Model {self.model_saved_names[index]} got accuracy of {accuracy}")
                performances[f"{self.model_saved_names[index]}_test_acc"] = accuracy
                collaboration_performance[index].append(accuracy)
            
            # wandb logging
            if r < self.N_rounds // 3:
                wandb.log(self.init_result, commit=False)
            elif r >= 2*self.N_rounds // 3:
                wandb.log(self.pooled_train_result, commit=False)
            wandb.log(performances)
            
            r += 1
            if r > self.N_rounds:
                break

            print("Update models ...")
            for index, agent in enumerate(self.collaborative_agents):
                print(f"Model {self.model_saved_names[index]} starting alignment with public logits... ")

                # Distribute + Digest phase
                weights_to_use = None
                weights_to_use = agent["model_weights"]

                agent["model_logits"].load_state_dict(weights_to_use)
                optimizer = load_optimizer(agent["model_logits"], agent["train_params"])
                # To align logits scores, the logits are used as "class labels"
                # so the Mean Absolute error needs to be used
                logits_loss = nn.L1Loss()
                alignment_data.targets = logits
                train_model(
                    agent["model_logits"],
                    alignment_data,
                    loss_fn=logits_loss,
                    optimizer=optimizer,
                    batch_size=self.logits_matching_batchsize,
                    num_epochs=self.N_logits_matching_round,
                )

                agent["model_weights"] = agent["model_logits"].state_dict()

                print(f"Model {self.model_saved_names[index]} done alignment")
                print(f"Model {self.model_saved_names[index]} starting training with private data... ")
                weights_to_use = None
                weights_to_use = agent["model_weights"]

                # Revisit phase
                agent["model_classifier"].load_state_dict(weights_to_use)

                optimizer = load_optimizer(agent["model_classifier"], agent["train_params"])
                loss = nn.CrossEntropyLoss()
                train_model(
                    agent["model_classifier"],
                    self.private_data[index],
                    loss_fn=loss,
                    optimizer=optimizer,
                    batch_size=self.private_training_batchsize,
                    num_epochs=self.N_private_training_round,
                )

                agent["model_weights"] = agent["model_classifier"].state_dict()

                print(f"Model {self.model_saved_names[index]} done private training. \n")
            # end for
        # end while
        return collaboration_performance
