import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import random
from omegaconf import DictConfig, OmegaConf, open_dict
from time import time
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

# I used scienceplots for the style of the plots,
# but you can use any other style you want.
warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

def set_seed(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_loggers(cfg):
    """ Set the loggers for the experiment """
    # Update the note in the config: if it is None, set it to an empty string
    with open_dict(cfg):
        cfg.update(
            note = "_" if cfg.note is None else "_"+str(cfg.note)
        )
    name = f"seed{cfg.seed}.{int(time())}"
    group_format = (
        "{dataset}_"
        "{model}"
        "{note}"
    )
    # Define the tags for wandb
    tags = [cfg.dataset.metadata.name, cfg.model.metadata.name, cfg.note]
    # Define the group for wandb
    group = group_format.format(**parse_hyperparams(cfg))
    if cfg.wandb.project is None or cfg.wandb.entity is None:
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(project=cfg.wandb.project, 
                               entity=cfg.wandb.entity, 
                               name=name,
                               group=group,
                               tags=tags)
        wandb_logger.log_hyperparams(parse_hyperparams(cfg))
    csv_logger = CSVLogger("logs/",
                           name="experiment_metrics")
    return wandb_logger, csv_logger

def parse_hyperparams(cfg: DictConfig):
    hyperparams = {
        "dataset": cfg.dataset.metadata.name,
        "model": cfg.model.metadata.name,
        "seed": cfg.seed,
        "hydra_cfg": OmegaConf.to_container(cfg),
        "note": cfg.note,
    }
    return hyperparams

def update_config_from_data(cfg: DictConfig, train_loader, c_names, y_names, c_groups, csv_log_dir) -> DictConfig:
    """
    Update the config with the input size, output size, and concept names.
    """
    x, c, y = next(iter(train_loader))
    input_size = x.shape[1]
    concept_size = c.shape[1]
    n_labels = len(y_names)
    if c_groups is None or not isinstance(c_groups, dict):
        c_groups = c_groups
    else:
        c_groups = dict(c_groups)
    
    with open_dict(cfg):
        cfg.engine.update(
            c_names = c_names,
            y_name = y_names,
            csv_log_dir = csv_log_dir
        )
        
        cfg.model.params.update(
            input_size = input_size,
            output_size = n_labels,
            c_names = c_names,
            y_names = y_names,
            task = cfg.dataset.metadata.task,
            c_groups = c_groups
        )
    return cfg

def plot_explanations(lmr_paths):

    for exp in lmr_paths:
        # If it does not exist, create the figs directory
        figs_path = f'figs/PredCBM_explanations/{exp['dataset']}'
        if not os.path.exists(figs_path):
            os.makedirs(figs_path)
        exp_info_path = os.path.join(exp['path'], 'logs/experiment_metrics/version_0')
        # Load all the required files
        pred_CBMs = torch.load(os.path.join(exp_info_path, 'pred_CBMs.pt'))
        c_trues = pd.read_csv(os.path.join(exp_info_path, 'c_trues.csv'))
        c_preds = pd.read_csv(os.path.join(exp_info_path, 'c_preds.csv'))
        y_trues = pd.read_csv(os.path.join(exp_info_path, 'y_trues.csv'))
        y_preds = pd.read_csv(os.path.join(exp_info_path, 'y_preds.csv'))

        # Sample N random samples from the dataset
        n_samples = 20
        random_indices = np.random.choice(len(y_trues), n_samples, replace=False)

        # Plot the explanations
        for i in range(n_samples):
            idx = random_indices[i]
            # Get the true and predicted concepts
            c_true = c_trues.iloc[idx].values
            c_pred = (c_preds.iloc[idx].values > 0.5).astype(int)

            # get the column name of the concepts corresponding to the highest value
            c_name_pred = [x.replace('_',' ') for idx, x in enumerate(c_preds.columns) if idx in np.argwhere(c_pred>0.5)]
            c_name_true = [x.replace('_',' ') for idx, x in enumerate(c_trues.columns) if idx in np.argwhere(c_true>0.5)]
            # Get the true and predicted labels
            y_true = y_trues.iloc[idx].values.argmax(-1)
            y_pred = y_preds.iloc[idx].values.argmax(-1)
            # get the column name of the concepts corresponding to the highest value
            y_name_pred = [x.replace('_',' ') for x in y_preds.columns[y_pred]]
            y_name_true = [x.replace('_',' ') for x in y_trues.columns[y_true]]
            # Get the weights associated to the predicted class of the 
            # predicted CBM
            pred_CBM = pred_CBMs[idx,y_pred,:,:].squeeze().cpu().numpy()
            # Perform the element-wise multiplication
            logits = np.multiply(pred_CBM, c_pred)

            # Plot the explanations
            fig, ax = plt.subplots(figsize=(10, 5))
            bar_colors = ['tab:blue' if val >= 0 else 'tab:red' for val in logits]
            y_pos = np.arange(len(logits))
            ax.barh(
                y_pos, logits, color=bar_colors,
                edgecolor='black', linewidth=1.5, alpha=0.6
            )
            # Add black vertical line at 0
            ax.axvline(x=0, color='black', linewidth=0.9)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([x.replace('_',' ') for x in c_preds.columns], 
                                fontsize=14)
            ax.set_xlabel('Logit Value', fontsize=12)
            ax.set_title(f'Predicted class: {y_name_pred}, True class: {y_name_true}\nTrue concepts: {c_name_true}', 
                         fontsize=12)
            # Eliminate minor ticks
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_minor_locator(plt.NullLocator())
            # Save the figure
            plt.tight_layout()
            plt.savefig(f'{figs_path}/explanations_{i}.pdf')