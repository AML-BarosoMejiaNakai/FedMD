import wandb
import torch
import os
from constants import *
## -- WANDB --

def init_wandb(run_id=None):
    group_name = "fedmd"

    configuration = CONF_MODELS_BALANCED
    agents = ""
    for agent in configuration["models"]:
        agents += agent["model_type"][0]
    job_name = f"M{configuration['N_agents']}_N{configuration['N_rounds']}_S{CONF_MODELS_BALANCED['N_subset']}_lr{LR}_A{agents}"

    run = wandb.init(
                id = run_id,
                # Set entity to specify your username or team name
                entity="aml-30lsiuuu",
                # Set the project where this run will be logged
                project='fl_md',
                group=group_name,
                # Track hyperparameters and run metadata
                config=configuration,
                resume="allow")

    if os.environ["WANDB_MODE"] != "offline" and not wandb.run.resumed:
        random_number = wandb.run.name.split('-')[-1]
        wandb.run.name = job_name + '-' + random_number
        wandb.run.save()
        resumed = False
    if wandb.run.resumed:
        resumed = True

    return run, job_name, resumed


def load_checkpoint(path, model, restore_path = None):
    loaded = False
    if wandb.run.resumed or restore_path is not None:
        try:
            weights = wandb.restore(path, run_path=restore_path)
            model.load_state_dict(torch.load(weights.name))
            print(f"===== SUCCESFULLY LOADED {path} FROM CHECKPOINT =====")
            loaded = True
        except ValueError:
            print(f"===== CHECKPOINT FOR {path} DOES NOT EXIST =====")            
        except RuntimeError:
            print(f"===== CHECKPOINT FOR {path} IS CORRUPTED =====")
            print("Deleting...")
            files = wandb.run.files()
            for file in files:
                if file.name == path:
                    file.delete()
            print("Deleted. Sorry for the inconveniences")
    return loaded
