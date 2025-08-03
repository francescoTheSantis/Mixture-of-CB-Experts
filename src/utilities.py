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
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import scienceplots

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

def is_valid_experiment(cfg: DictConfig):
    """ 
    Check if the experiment is valid based on the dataset and model combination.
    """
    if cfg.dataset.metadata.name == 'cebab' and cfg.model.metadata.name in ['cem, dcr, cmr']:
        raise ValueError(f"The experiment is not valid. Please check the configuration.\n\
                         The combination of {cfg.dataset.metadata.name}, {cfg.model.metadata.name} cannot be executed.")

def set_loggers(cfg):
    """ Set the loggers for the experiment """
    # Update the note in the config: if it is None, set it to an empty string
    with open_dict(cfg):
        cfg.update(
            note = "_" if cfg.note is None else "_"+str(cfg.note)
        )
    name = f"{cfg.dataset.metadata.name}.{cfg.model.metadata.name}.{cfg.seed}.{int(time())}"
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

def get_backbone_latent_size(backbone):
    if backbone == 'resnet18':
        model = resnet18(pretrained=True)
    elif backbone == 'resnet34':
        model = resnet34(pretrained=True)
    elif backbone == 'resnet50':
        model = resnet50(pretrained=True)
    elif backbone == 'resnet101':
        model = resnet101(pretrained=True)
    elif backbone == 'resnet152':
        model = resnet152(pretrained=True)
    elif backbone == 'bert-base-uncased':
        return 768  # BERT base model has 768 hidden size
    else:
        raise ValueError(f"Image backbone {backbone} not recognized.")
    
    if 'resnet' in backbone:
        model = nn.Sequential(*list(model.children())[:-1])
        test = model(torch.randn((1,3,224,224)))
        latent_dim = test.flatten(start_dim=1).shape[1]
    else:
        pass # This is a placeholder for other backbones if needed
    # delete the model to free memory
    del model
    del test
    torch.cuda.empty_cache()
    return latent_dim

def get_type_from_name(dataset_name):
    if dataset_name in ['mnist_addition', 'cub', 'cub_incomplete', 'awa2', 'awa2_incomplete', 'xor']:
        return 'image'
    else:
        return 'text'

def update_config_from_data(cfg: DictConfig, train_loader, c_names,
                            y_names, c_groups, csv_log_dir) -> DictConfig:
    """
    Update the config with the input size, output size, and concept names.
    """
    batch = next(iter(train_loader))

    if get_type_from_name(cfg.dataset.metadata.name) == 'image':
        x = batch['x']
        data_type = 'image'
    else:
        x = batch['x']['input_ids']
        data_type = 'text'

    if cfg.dataset.metadata.name != 'sst2':
        input_size = torch.prod(torch.tensor(x.shape[1:])).item()
    else:
        input_size = x.shape[-1]
        
    n_labels = len(y_names)

    if c_groups is None or not isinstance(c_groups, dict):
        c_groups = c_groups
    else:
        c_groups = dict(c_groups)

    with open_dict(cfg):
        cfg.engine.update(
            c_names = c_names,
            y_name = y_names,
            csv_log_dir = csv_log_dir,
            data_type = data_type,
        )
        
        # store in the config the size of the embeddings produced by the backbone
        # check if type is dataset.encoder.encoder
        if 'encoder' in cfg.dataset and 'type' in cfg.dataset.encoder.encoder:
            backbone_latent_size = get_backbone_latent_size(cfg.dataset.encoder.encoder.type)
        else:
            backbone_latent_size = cfg.dataset.latent_size

        cfg.model.params.update(
            output_size = n_labels,
            c_names = c_names,
            y_names = y_names,
            task = cfg.dataset.metadata.task,
            c_groups = c_groups,
            #concept_loss_form = cfg.dataset.get('concept_loss_form', {'_target_': 'torch.nn.BCELoss'}),
            backbone_latent_size = backbone_latent_size,
            concept_type = cfg.dataset.metadata.concept_type

        )

        # if we want to extract the embeddings it means that we are NOT 
        # fine-tuning a pre-trained backbone during training.
        # This means that we just need a linear encoder.
        if cfg.extract_embeddings:
            input_size = backbone_latent_size 
            cfg.model.params.encoder = {
                '_target_': 'src.models.encoders.linear.LinearEncoder',
                'output_size': backbone_latent_size, # we do not want the linear layer to reduce the size of the embeddings
                'activation': cfg.activation,
            }
        else:
            # If we are fine-tuning a pre-trained model,
            # we need to set the encoder to the one defined in the dataset config.
            cfg.model.params.encoder = cfg.dataset.encoder.encoder

        cfg.model.params.encoder.update(
            input_size = input_size,
        )

    return cfg

# def plot_data(data, predictions):
#     """
#     Return the figure of a plotting of data and predictions for logical problems with two variables
#     and one output class (e.g. OR, NOR, XOR, XNOR). The data should be a 2D tensor
#     with shape (n_samples, 2) and the predictions should be a 1D tensor with shape
#     (n_samples,). The function will create a scatter plot of the data points and color them
#     according to the predictions. The prediction take values between 0 and 1.
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from matplotlib.colors import Normalize
#     from matplotlib.cm import ScalarMappable
#
#     assert data.shape[1] == 2, "Data should have shape (n_samples, 2)"
#     predictions = predictions.squeeze().cpu().numpy()
#     assert len(predictions.shape) == 1, "Predictions should be a 1D tensor"
#
#     # Create a scatter plot of the data points
#     fig = plt.figure(figsize=(8, 6))
#     norm = Normalize(vmin=0, vmax=1)
#     cmap = sns.color_palette("coolwarm", as_cmap=True)
#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#
#     plt.scatter(data[:, 0], data[:, 1], c=predictions, cmap=cmap, norm=norm, s=100)
#     plt.colorbar(sm, label='Predictions')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.title('Data and Predictions')
#
#     return fig


def plot_explanations(lmr_paths):

    for exp in lmr_paths:
        try:
            # If it does not exist, create the figs directory
            figs_path = f'figs/{exp["model"]}_explanations/{exp["dataset"]}'
            if not os.path.exists(figs_path):
                os.makedirs(figs_path)
            exp_info_path = os.path.join(exp['path'], 'logs/experiment_metrics/version_0')
            # Load all the required files
            pred_CBMs = torch.load(os.path.join(exp_info_path, 'pred_CBMs.pt'))
            c_trues = pd.read_csv(os.path.join(exp_info_path, 'c_trues.csv'))
            c_preds = pd.read_csv(os.path.join(exp_info_path, 'c_preds.csv'))
            y_trues = pd.read_csv(os.path.join(exp_info_path, 'y_trues.csv'))
            y_preds = pd.read_csv(os.path.join(exp_info_path, 'y_preds.csv'))
            binary_classification = True if len(y_trues.columns) == 2 else False

            # Sample N random samples from the test-set
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
                if binary_classification:
                    y_true = y_trues.iloc[idx,1]
                    y_pred = y_preds.iloc[idx,1]
                else:
                    y_true = y_trues.iloc[idx].values.argmax(-1)
                    y_pred = y_preds.iloc[idx].values.argmax(-1)
                # get the column name of the concepts corresponding to the highest value
                y_name_pred = y_preds.columns[y_pred].replace('_',' ')
                y_name_true = y_trues.columns[y_true].replace('_',' ')
                # Get the weights associated to the predicted class of the 
                # predicted CBM
                if binary_classification:
                    y_pred = 0
                pred_CBM = pred_CBMs[idx,y_pred,:,:].squeeze().cpu().numpy()
                # Perform the element-wise multiplication
                logits = np.multiply(pred_CBM, c_pred)

                # Sort the logits by absolute value in ascending order and take the k highest values
                top_k = 10
                indices = np.argsort(np.abs(logits))[::-1][:top_k]
                logits = logits[indices]
                c_preds_names = [c_preds.columns[idx] for idx in indices]
                # c_trues_names = [c_trues.columns[idx] for idx in indices]

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
                ax.set_yticklabels([x.replace('_',' ') for x in c_preds_names], 
                                    fontsize=14)
                ax.set_xlabel('Logit Value', fontsize=12)
                ax.set_title(f'Predicted class: {y_name_pred}, True class: {y_name_true}', 
                            fontsize=12)
                # Eliminate minor ticks
                ax.xaxis.set_minor_locator(plt.NullLocator())
                ax.yaxis.set_minor_locator(plt.NullLocator())
                # Save the figure
                plt.tight_layout()
                plt.savefig(f'{figs_path}/explanations_{i}.pdf')
        except:
            print(f"Error while plotting explanations for {exp['model']} on {exp['dataset']}. Skipping...")
            continue
    print("Explanations plotted successfully.")