import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
from omegaconf import DictConfig, OmegaConf, open_dict
from time import time
from pytorch_lightning.loggers import WandbLogger, CSVLogger

def set_seed(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_intervened_concepts_predictions(predictions, labels, probability, return_index=False, all_entries=False, repeat=None):
    
    hard_predictions = torch.where(predictions > 0.5, 1, 0)

    if repeat!=None:
        hard_predictions = hard_predictions.unsqueeze(-1).expand(-1, -1, repeat)  
        labels = labels.unsqueeze(-1).expand(-1, -1, repeat)   
        predictions = predictions.unsqueeze(-1).expand(-1, -1, repeat)   
        
    # Find mismatched indices if all_entries is False, select all otherwise
    if all_entries:
        mismatched_mask = (torch.ones_like(hard_predictions))
    else:
        mismatched_mask = (hard_predictions != labels)

    # Generate a probability mask of the same shape
    random_mask = torch.rand_like(predictions, dtype=torch.float)

    # Apply probability threshold only on mismatched elements
    mask = (random_mask < probability) & mismatched_mask
    mask = mask.int()

    # Apply intervention
    intervened = labels * mask + predictions * (1-mask)

    if return_index:
        return mask, intervened
    else:
        return intervened

def set_loggers(cfg):
    name = f"seed{cfg.seed}.{int(time())}"
    group_format = (
        "{dataset}_"
        "{model}"
    )
    group = group_format.format(**parse_hyperparams(cfg))
    if cfg.wandb.project is None or cfg.wandb.entity is None:
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(project=cfg.wandb.project, 
                               entity=cfg.wandb.entity, 
                               name=name,
                               group=group)
    csv_logger = CSVLogger("logs/", 
                           name="experiment_metrics")
    return wandb_logger, csv_logger

def parse_hyperparams(cfg: DictConfig):
    hyperparams = {
        "dataset": cfg.dataset.metadata.name,
        "model": cfg.model.metadata.name,
        "seed": cfg.seed,
        "hydra_cfg": OmegaConf.to_container(cfg),
    }
    return hyperparams


def update_config_from_data(cfg: DictConfig, train_loader, c_names, y_names) -> DictConfig:
    """ can be used to update the config based on the data, e.g., set input and output size """
    x, c, y = next(iter(train_loader))
    input_size = x.shape[1]
    concept_size = c.shape[1]
    n_labels = len(y_names) if len(y_names) > 1 else 2
    
    with open_dict(cfg):
        cfg.engine.update(
            c_names = c_names,
            y_name = y_names,
        )
        
        cfg.model.params.update(
            input_size = input_size,
            output_size = n_labels,
            c_names = c_names,
            y_names = y_names,
            task = cfg.dataset.metadata.task,
            dataset = cfg.dataset.metadata.name
        )
    return cfg