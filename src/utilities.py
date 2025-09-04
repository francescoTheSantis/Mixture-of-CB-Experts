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
from transformers import CLIPProcessor, CLIPModel
import transformers
import scienceplots
from env import CACHE

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
    if cfg.dataset.metadata.name == 'cebab' and cfg.model.metadata.name in ['cem', 'dcr', 'cmr']:
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
    elif 'vit' in backbone or 'dino' in backbone:
        model = transformers.ViTModel.from_pretrained(backbone)
    elif backbone == 'bert-base-uncased':
        model = transformers.AutoModel.from_pretrained(backbone)
        return 768  # BERT base model has 768 hidden size
    elif backbone == 'sentence-transformers/all-mpnet-base-v2':
        return 768  # all-mpnet-base-v2 model has 768 hidden size
    elif backbone == 'sentence-transformers/all-MiniLM-L6-v2':
        return 384  # all-MiniLM-L6-v2 model has 384 hidden size
    else:
        raise ValueError(f"Image backbone {backbone} not recognized.")
    
    if 'resnet' in backbone:
        model = nn.Sequential(*list(model.children())[:-1])
        test = model(torch.randn((1,3,224,224)))
        latent_dim = test.flatten(start_dim=1).shape[1]
    elif 'vit' in backbone or 'dino' in backbone:
        test = model(torch.randn((1,3,224,224)))
        latent_dim = test.last_hidden_state[:, 0, :].shape[1]
    else:
        pass 
    
    # delete the model to free memory
    del model
    del test
    torch.cuda.empty_cache()
    return latent_dim

def get_type_from_name(dataset_name):
    if dataset_name in ['mnist_addition', 'cub', 'cub_incomplete', \
                        'awa2', 'awa2_incomplete', 'xor', 'celeba', \
                        'cifar10', 'cifar100', 'mnist_arithmetic','pendulum']:
        return 'image'
    else:
        return 'text'

def get_batch_from_loader(train_loader, device):
    with torch.cuda.device(device if device != 'cpu' else 'cpu'):
        # Temporarily set default tensor type to CPU to avoid automatic GPU allocation
        original_default_tensor_type = torch.get_default_dtype()
        if device == 'cpu':
            torch.set_default_tensor_type('torch.FloatTensor')
        
        batch = next(iter(train_loader))
        
        # Restore original tensor type
        torch.set_default_dtype(original_default_tensor_type)
    return batch

def setup_encoder(cfg: DictConfig, input_size: int, backbone_latent_size: int) -> DictConfig:

    type = None

    # if we want to extract the embeddings it means that we are NOT 
    # fine-tuning a pre-trained backbone during training.
    # This means that we just need a linear encoder.
    if cfg.extract_embeddings:
        input_size = input_size if cfg.dataset.metadata.name != 'xor' else 2 
        cfg.model.params.encoder = {
            '_target_': 'src.models.encoders.mlp.MLPEncoder',
            'output_size': backbone_latent_size, 
            'activation': cfg.activation,
            'input_transform': {
                '_target_': 'src.models.encoders.transform.FlattenTransform',
            },
            'dropout': 0.1
        }

    else:
        backbone = cfg.img_backbone_name if get_type_from_name(cfg.dataset.metadata.name) == 'image' \
                                            else cfg.text_backbone_name
        
        if cfg.dataset.metadata.data_type == 'toy':
            target = 'src.models.encoders.linear.LinearEncoder'
            transform = None
        else:
            if 'vit' in backbone:
                target = 'src.models.encoders.vit.VitEncoder'
                transform = {
                    '_target_': "src.models.encoders.transform.VitTransform",
                    'flatten': False
                }
            elif 'resnet' in backbone:
                target = 'src.models.encoders.resnet.ResNetEncoder'
                transform = {
                    '_target_': "src.models.encoders.transform.ImageTransform",
                    'flatten': False
                }
            elif ('sentence-transformers' in backbone) or ('bert' in backbone):
                target = 'src.models.encoders.transformer.TransformerEncoder'
                transform = None
                type = backbone

        # If the dataset is mnist_addition, then perform its own preprocessing
        if cfg.dataset.metadata.name == 'mnist_addition':
            transform = {
                '_target_': "src.models.encoders.transform.MNISTTransform",
                'flatten': False
            }

        # If we are fine-tuning a pre-trained model,
        # we need to set the encoder to the one defined in the dataset config.
        cfg.model.params.encoder = {
            '_target_': target,
            'output_size': backbone_latent_size, # we do not want the linear layer to reduce the size of the embeddings
            'input_transform' : transform,
        } 

    with open_dict(cfg):
        cfg.model.params.encoder.update(
            input_size = input_size,
        )

        if type != None:
            cfg.model.params.encoder.update(
                type = type
            )

    return cfg

def allow_hard_concepts(dataset_name):
    # Datasets that do not allow hard concepts so far: cebab, mnist_arithmetic.
    return dataset_name in ['xor', 'cub', 'mnist_addition', 'awa2', 'cub_incomplete', 'awa2_incomplete', 'cifar10', 'cifar100']


def update_config_from_data(cfg: DictConfig, train_loader, c_names,
                            y_names, c_groups, csv_log_dir) -> DictConfig:
    """
    Update the config with the input size, output size, and concept names.
    """
    # Create a temporary dataloader that loads data directly to the specified device
    # or force CPU loading to avoid automatic GPU allocation
    device = cfg.gpus[0]
    batch = get_batch_from_loader(train_loader, device)

    if get_type_from_name(cfg.dataset.metadata.name) == 'image':
        x = batch['x']
        data_type = 'image'
    else:
        x = batch['x'] if cfg.extract_embeddings else batch['x']['input_ids']
        data_type = 'text'

    if cfg.dataset.metadata.name != 'sst2':
        input_size = torch.prod(torch.tensor(x.shape[1:])).item()
    else:
        input_size = x.shape[-1]
        
    # Clean up GPU memory
    del batch
    del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_labels = len(y_names)

    if c_groups is None or not isinstance(c_groups, dict):
        c_groups = c_groups
    else:
        c_groups = dict(c_groups)

    backbone = cfg.img_backbone_name if get_type_from_name(cfg.dataset.metadata.name) == 'image' \
                                        else cfg.text_backbone_name
    backbone_latent_size = cfg.dataset.latent_size if cfg.extract_embeddings else get_backbone_latent_size(backbone)

    with open_dict(cfg):
        cfg.engine.update(
            c_names = c_names,
            y_name = y_names,
            csv_log_dir = csv_log_dir,
            data_type = data_type,
            dataset_name = cfg.dataset.metadata.name
        )

        hard_concepts = cfg.hard_concepts
        concept_type = prepare_concept_type(cfg.dataset.metadata.concept_type, cfg.engine.c_names)

        # If cfg.dataset.equations exists, then we want to use the known equations,
        # null otherwise
        known_equations = cfg.dataset.equations if 'equations' in cfg.dataset and cfg.dataset.equations is not None else None

        cfg.model.params.update(
            output_size = n_labels,
            c_names = c_names,
            y_names = y_names,
            task = cfg.dataset.metadata.task,
            c_groups = c_groups,
            backbone_latent_size = backbone_latent_size,
            concept_type = concept_type,
            hard_concepts = hard_concepts,
            known_equations = known_equations
        )

        cfg = setup_encoder(cfg, input_size, backbone_latent_size)

    return cfg

def prepare_concept_type(c_types, c_names):
    n_concepts = len(c_names)
    if isinstance(c_types, list):
        return c_types
    else:
        return [c_types] * n_concepts

def generate_data_path(cfg):
    data_path = os.path.join(str(CACHE), 
                             'stored_tensors', 
                             'embeddings' if cfg.extract_embeddings else 'raw', # whether it contains embeddings or not
                             cfg.dataset.metadata.name)

    # Add backbone name
    img_backbone_name = cfg.img_backbone_name.replace('/', '_')
    text_backbone_name = cfg.text_backbone_name.replace('/', '_')

    data_path += f"/{img_backbone_name}" if get_type_from_name(cfg.dataset.metadata.name) == 'image' \
                                                    else f"/{text_backbone_name}"

    if cfg.dataset.loader.concept_percentage != None:
        data_path += f"_{str(cfg.dataset.loader.concept_percentage).replace('.', '')}"


    train_path = f"{data_path}/train.pt"
    val_path = f"{data_path}/val.pt"
    test_path = f"{data_path}/test.pt"

    return data_path, train_path, val_path, test_path