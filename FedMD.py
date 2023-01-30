import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
from model_trainers import *
from CIFAR import stratified_sampling
import wandb
import wandb_utils
from constants import *


class FedMD:
    # parties changed to agents
    # N_alignment changed to N_subset

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

        print("start model initialization: ")
        for i in range(self.N_agents):
            print("Model ", self.model_saved_names[i])
            model_A = copy.deepcopy(agents[i])  # Was clone_model
            # model_A.set_weights(agents[i].get_weights())
            if not wandb_utils.load_checkpoint(f"ckpt/{self.model_saved_names[i]}_initial_pri.pt", model_A, restore_path):
                model_A.load_state_dict(agents[i].state_dict())
                # model_A.compile(optimizer=tf.keras.optimizers.Adam(lr = LR),
                #                      loss = "sparse_categorical_crossentropy",
                #                      metrics = ["accuracy"])
                optimizer = optim.Adam(
                    model_A.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
                )
                loss = nn.CrossEntropyLoss()
                early_stopping = EarlyStop(patience=10, min_delta=0.01)

                print("start full stack training ... ")

                # model_A.fit(private_data[i]["X"], private_data[i]["y"],
                #                  batch_size = 32, epochs = 25, shuffle=True, verbose = 0,
                #                  validation_data = [private_test_data["X"], private_test_data["y"]],
                #                  callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=10)]
                #                 )
                # OBS: Also passes the validation data and uses EarlyStopping
                # TODO: Early stopping on train_model

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

                # model_A = remove_last_layer(model_A, loss="mean_absolute_error")
                # model_A = nn.Sequential(*(list(model_A.children())[:-1])) # Removing last layer of the model_A
            else:
                test_acc = test_network(model_A, private_test_data, 32)
                wandb.run.summary[f"{model_saved_names[i]}_initial_test_acc"] = test_acc
                self.init_result[f"{model_saved_names[i]}_initial_test_acc"] = last_test_acc["test_accuracy"]
            # end if load_checkpoint

            self.collaborative_agents.append({
                "model_logits": model_A,
                "model_classifier": model_A,
                "model_weights": model_A.state_dict(),
            })  # Was get_weights()

            # TODO: Need to include also the validation dataset on model_train and save these statistics
            # self.init_result.append({"val_acc": model_A.history.history['val_accuracy'],
            #                          "train_acc": model_A.history.history['accuracy'],
            #                          "val_loss": model_A.history.history['val_loss'],
            #                          "train_loss": model_A.history.history['loss'],
            #                         })

            print()
            del model_A
        # END FOR LOOP

        print("Calculate the theoretical upper bounds for participants: ")

        self.upper_bounds = []
        self.pooled_train_result = {}
        for i, model in enumerate(agents):
            print(f"UB - Model {self.model_saved_names[i]}")
            model_ub = copy.deepcopy(model)
            if not wandb_utils.load_checkpoint(f"ckpt/ub/{self.model_saved_names[i]}_ub.pt", model_ub, restore_path):
                # model_ub.set_weights(model.get_weights())
                model_ub.load_state_dict(model.state_dict())
                # model_ub.compile(optimizer=tf.keras.optimizers.Adam(lr = LR),
                #                  loss = "sparse_categorical_crossentropy",
                #                  metrics = ["accuracy"])
                optimizer = optim.Adam(
                    model_ub.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
                )
                loss = nn.CrossEntropyLoss()
                early_stopping = EarlyStop(patience=10, min_delta=0.01)


                # model_ub.fit(total_private_data["X"], total_private_data["y"],
                #              batch_size = 32, epochs = 50, shuffle=True, verbose = 0,
                #              validation_data = [private_test_data["X"], private_test_data["y"]],
                #              callbacks=[EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=10)])
                # TODO: EarlyStopping!
                # OBS: Validation accuracy == our test accuracy since it is the value at the end of each epoch

                accuracy = train_model(
                    network=model_ub,
                    dataset=total_private_data,
                    test_dataset=private_test_data,
                    loss_fn=loss,
                    optimizer=optimizer,
                    early_stop=early_stopping,
                    batch_size=BATCH_SIZE,
                    num_epochs=50,
                    returnAcc=True,
                )

                torch.save(model_ub.state_dict(), f'ckpt/ub/{model_saved_names[i]}_ub.pt')
                wandb.save(f'ckpt/ub/{model_saved_names[i]}_ub.pt')
                last_acc = accuracy[-1]["test_accuracy"]
            else:
                last_acc = test_network(model_ub, private_test_data, 32)

            wandb.run.summary[f"{model_saved_names[i]}_ub_test_acc"] = last_acc

            # self.upper_bounds.append(model_ub.history.history["val_accuracy"][-1])
            self.upper_bounds.append(last_acc)
            # self.pooled_train_result.append({"val_acc": model_ub.history.history["val_accuracy"],
            #                                  "acc": model_ub.history.history["accuracy"]}) # "accuracy" == train accuracy
            self.pooled_train_result[f"{model_saved_names[i]}_ub_test_acc"] = last_acc

            del model_ub
        print("The upper bounds are:", self.upper_bounds)

    # end init

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_agents)}
        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            # alignment_data = generate_alignment_data(self.public_dataset["X"],
            #                                          self.public_dataset["y"],
            #                                          self.N_subset)
            alignment_data = stratified_sampling(self.public_dataset, self.N_subset)

            print(f"Round {r}/{self.N_rounds}")

            print("update logits ... ")
            # update logits
            logits = 0
            for agent in self.collaborative_agents:
                # agent["model_logits"].set_weights(agent["model_weights"])
                agent["model_logits"].load_state_dict(agent["model_weights"])
                # logits += agent["model_logits"].predict(alignment_data["X"], verbose = 0)
                model_logits = run_dataset(agent["model_logits"], alignment_data)
                logits += model_logits.to('cpu')

            logits /= self.N_agents
            
            # test performance
            print("test performance ... ")

            performances = {}

            for index, agent in enumerate(self.collaborative_agents):
                # y_pred = agent["model_classifier"].predict(self.private_test_data["X"], verbose = 0).argmax(axis = 1)
                accuracy = test_network(network=agent["model_classifier"], test_dataset=self.private_test_data)

                print(f"Model {self.model_saved_names[index]} got accuracy of {accuracy}")
                performances[f"{self.model_saved_names[index]}_test_acc"] = accuracy
                collaboration_performance[index].append(accuracy)
            
            if r < self.N_rounds // 3:
                wandb.log(self.init_result, commit=False)
            elif r >= 2*self.N_rounds // 3:
                wandb.log(self.pooled_train_result, commit=False)
            wandb.log(performances)
            
            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, agent in enumerate(self.collaborative_agents):
                print(f"Model {self.model_saved_names[index]} starting alignment with public logits... ")

                weights_to_use = None
                weights_to_use = agent["model_weights"]

                # agent["model_logits"].set_weights(weights_to_use)
                agent["model_logits"].load_state_dict(weights_to_use)
                # agent["model_logits"].fit(alignment_data["X"], logits,
                #                       batch_size = self.logits_matching_batchsize,
                #                       epochs = self.N_logits_matching_round,
                #                       shuffle=True, verbose = 0)
                optimizer = optim.Adam(
                    agent["model_logits"].parameters(), lr=LR, weight_decay=WEIGHT_DECAY
                )
                logits_loss = nn.L1Loss()
                alignment_data.targets = logits
                train_model(
                    agent["model_logits"],
                    alignment_data,
                    loss_fn=logits_loss,
                    batch_size=self.logits_matching_batchsize,
                    num_epochs=self.N_logits_matching_round,
                    optimizer=optimizer,
                )

                # agent["model_weights"] = agent["model_logits"].get_weights()
                agent["model_weights"] = agent["model_logits"].state_dict()

                print(f"Model {self.model_saved_names[index]} done alignment")

                print(f"Model {self.model_saved_names[index]} starting training with private data... ")
                weights_to_use = None
                weights_to_use = agent["model_weights"]

                # agent["model_classifier"].set_weights(weights_to_use)
                agent["model_classifier"].load_state_dict(weights_to_use)

                # agent["model_classifier"].fit(self.private_data[index]["X"],
                #                           self.private_data[index]["y"],
                #                           batch_size = self.private_training_batchsize,
                #                           epochs = self.N_private_training_round,
                #                           shuffle=True, verbose = 0)

                optimizer = optim.Adam(
                    agent["model_classifier"].parameters(),
                    lr=LR,
                    weight_decay=WEIGHT_DECAY,
                )
                loss = nn.CrossEntropyLoss()
                train_model(
                    agent["model_classifier"],
                    self.private_data[index],
                    loss_fn=loss,
                    batch_size=self.private_training_batchsize,
                    num_epochs=self.N_private_training_round,
                    optimizer=optimizer,
                )

                # agent["model_weights"] = agent["model_classifier"].get_weights()
                agent["model_weights"] = agent["model_classifier"].state_dict()

                print(f"Model {self.model_saved_names[index]} done private training. \n")
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance
