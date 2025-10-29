"""
Dataset Factory Module

This module provides factory functions for creating dataset splits and retrieving
metadata for all supported datasets. This centralizes dataset-specific logic and
makes the dataloader more maintainable.
"""

import os
import random
import itertools
import torch
import pandas as pd
from torch.utils.data import random_split
from torchvision import transforms

from torch_concepts.data import ToyDataset
from torch_concepts.data.mnist import MNISTAddition
from torch_concepts.data.celeba import CelebADataset

from src.loaders.datasets.cub import (
    CUBDataset,
    SELECTED_CONCEPTS as cub_selected_concepts,
    CONCEPT_SEMANTICS as cub_concept_semantics,
    CLASS_NAMES as cub_class_names,
    CONCEPT_GROUP_MAP as cub_concept_groups,
)
from src.loaders.datasets.awa2 import (
    AwA2Dataset,
    CONCEPT_SEMANTICS as awa2_concept_semantics,
    CLASS_NAMES as awa2_class_names,
    CONCEPT_GROUPS as awa2_concept_groups,
)
from src.loaders.datasets.cebab import CEBaBDataset
from src.loaders.datasets.cifar.cifar10 import get_CIFAR10_CBM_dataloader
from src.loaders.datasets.cifar.cifar100 import get_CIFAR100_CBM_dataloader
from src.loaders.datasets.mnist_arithmetic import (
    ArithmeticMNISTDataset,
    CONCEPT_NAMES as mnist_arithmetic_concept_names,
)
from src.loaders.datasets.pendulum import (
    PendulumDataset,
    CONCEPT_NAMES as concept_names_pendulum,
    TASK_NAMES as task_names_pendulum,
)
from src.loaders.datasets.dsprites import DSprites
from src.loaders.datasets.mnist_exponential import MNISTExponential
from src.loaders.datasets.mawps import (
    MAWPSDataset,
    CONCEPT_NAMES as mawps_concept_names,
    TASK_NAMES as mawps_task_names,
)

from env import DATA_PATH


# Helper for CUB concept names
CUB_CONCEPT_NAMES = [
    x for i, x in enumerate(cub_concept_semantics) if i in cub_selected_concepts
]


class DatasetMetadata:
    """Container for dataset metadata"""
    def __init__(self, concept_names, task_names, concept_groups=None):
        self.concept_names = concept_names
        self.task_names = task_names
        self.concept_groups = concept_groups


class DatasetFactory:
    """Factory for creating dataset instances and retrieving metadata"""
    
    @staticmethod
    def create_toy_dataset(name, seed=42, **kwargs):
        """Create toy datasets (xor, trigonometry, dot, checkmark, or, nor, xnor)"""
        if name in ['or', 'nor', 'xnor']:
            # These are based on the XOR dataset with modified labels
            dataset = ToyDataset('xor', size=1000, random_state=seed)
            
            if name == 'xnor':
                dataset.target_labels = 1 - dataset.target_labels
                assert torch.isclose(
                    dataset.target_labels.mean(), torch.tensor(0.5), atol=0.1
                ), "XNOR dataset not generated correctly"
                dataset.name = 'xnor'
                dataset.task_attr_names = 'xnor'
            elif name == 'nor':
                dataset.target_labels = (
                    (dataset.data[:, 0] < 0.5).float() * 
                    (dataset.data[:, 1] < 0.5).float()
                )
                assert torch.isclose(
                    dataset.target_labels.mean(), torch.tensor(0.25), atol=0.1
                ), "NOR dataset not generated correctly"
                dataset.name = 'nor'
                dataset.task_attr_names = 'nor'
            else:  # or
                dataset.target_labels = torch.clip(
                    (dataset.data[:, 0] > 0.5).float() + 
                    (dataset.data[:, 1] > 0.5).float(), 0, 1
                )
                assert torch.isclose(
                    dataset.target_labels.mean(), torch.tensor(0.75), atol=0.1
                ), "OR dataset not generated correctly"
                dataset.name = 'or'
                dataset.task_attr_names = 'or'
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [0.7, 0.1, 0.2]
            )
        else:
            # Standard toy datasets
            dataset = ToyDataset(name, size=1000, random_state=seed)
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [0.7, 0.1, 0.2]
            )
        
        # Metadata
        concept_names = dataset.concept_attr_names
        task_names = [name]
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_mnist_addition(**kwargs):
        """Create MNIST Addition dataset"""
        train_dataset = MNISTAddition(root=DATA_PATH, train=True)
        test_dataset = MNISTAddition(root=DATA_PATH, train=False)
        
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Get metadata from a sample
        concept_names = MNISTAddition(root=DATA_PATH, train=True).concept_names
        task_names = MNISTAddition(root=DATA_PATH, train=True).task_names
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_mnist_arithmetic(n_samples, data_path, hard_mode=False, **kwargs):
        """Create MNIST Arithmetic dataset"""
        operators = ('x', '/') if hard_mode else ('+', '-', 'x', '/')
        
        train_dataset = ArithmeticMNISTDataset(
            mnist_root=DATA_PATH, train=True, 
            num_samples=int(n_samples * 0.7), img_size=224, operators=operators
        )
        val_dataset = ArithmeticMNISTDataset(
            mnist_root=DATA_PATH, train=True,
            num_samples=int(n_samples * 0.1), img_size=224, operators=operators
        )
        test_dataset = ArithmeticMNISTDataset(
            mnist_root=DATA_PATH, train=False,
            num_samples=int(n_samples * 0.2), img_size=224, operators=operators
        )
        
        # Save equations
        operators_list = test_dataset.operator_list
        equations = [f"c0 {op} c1" for op in operators_list]
        equations_df = pd.DataFrame(equations, columns=['equation'])
        suffix = '_hard' if hard_mode else ''
        equations_df.to_csv(
            f"{data_path}/mnist_arithmetic{suffix}_equations.csv", index=False
        )
        
        concept_names = mnist_arithmetic_concept_names
        task_names = ['Result']
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_mnist_exponential(**kwargs):
        """Create MNIST Exponential dataset"""
        train_set = MNISTExponential(train=True)
        test_dataset = MNISTExponential(train=False)
        
        total_size = len(train_set)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        train_dataset, val_dataset = random_split(
            train_set, [train_size, val_size]
        )
        
        concept_names = ['digit']
        task_names = ['exponential']
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_cub_dataset(
        selected_concept_idxes=None, 
        concept_percentage=None,
        incomplete_cub_groups=None,
        **kwargs
    ):
        """Create CUB dataset (full or incomplete)"""
        train_dataset = CUBDataset(
            root=DATA_PATH, split='train', 
            selected_concepts=selected_concept_idxes
        )
        val_dataset = CUBDataset(
            root=DATA_PATH, split='val',
            selected_concepts=selected_concept_idxes
        )
        test_dataset = CUBDataset(
            root=DATA_PATH, split='test',
            selected_concepts=selected_concept_idxes
        )
        
        # Determine concept names
        if selected_concept_idxes is not None:
            concept_names = [
                cub_concept_semantics[i] for i in selected_concept_idxes
            ]
        else:
            concept_names = CUB_CONCEPT_NAMES
        
        task_names = cub_class_names
        concept_groups = incomplete_cub_groups if incomplete_cub_groups else cub_concept_groups
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, concept_groups
        )
    
    @staticmethod
    def create_celeba_dataset(task_names, transform, **kwargs):
        """Create CelebA dataset"""
        train_dataset = CelebADataset(
            root=DATA_PATH, split='train',
            class_attributes=task_names,
            transform=transform,
            download=True
        )
        test_dataset = CelebADataset(
            root=DATA_PATH, split='test',
            class_attributes=task_names,
            transform=transform,
            download=True
        )
        
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Get concept names from a test instance
        test_instance = CelebADataset(
            root=DATA_PATH, split='test',
            class_attributes=task_names,
            transform=transform
        )
        concept_names = test_instance.concept_attr_names
        task_name_list = ["class_" + str(x) for x in range(2 ** len(task_names))]
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_name_list, None
        )
    
    @staticmethod
    def create_awa2_dataset(
        selected_concept_idxes=None,
        incomplete_awa2_groups=None,
        **kwargs
    ):
        """Create AwA2 dataset (full or incomplete)"""
        path = os.path.join(DATA_PATH, 'Animals_with_Attributes2')
        
        train_dataset = AwA2Dataset(
            root=path, split='train',
            selected_concepts=selected_concept_idxes
        )
        val_dataset = AwA2Dataset(
            root=path, split='val',
            selected_concepts=selected_concept_idxes
        )
        test_dataset = AwA2Dataset(
            root=path, split='test',
            selected_concepts=selected_concept_idxes
        )
        
        # Determine concept names
        if selected_concept_idxes is not None:
            concept_names = [
                awa2_concept_semantics[i] for i in selected_concept_idxes
            ]
        else:
            concept_names = awa2_concept_semantics
        
        task_names = awa2_class_names
        concept_groups = incomplete_awa2_groups if incomplete_awa2_groups else awa2_concept_groups
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, concept_groups
        )
    
    @staticmethod
    def create_sst2_dataset(**kwargs):
        """Create SST2 dataset"""
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        train_dataset = load_dataset('SetFit/sst2', split='train')
        val_dataset = load_dataset('SetFit/sst2', split='validation')
        test_dataset = load_dataset('SetFit/sst2', split='test')
        
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.pad_token = tokenizer.eos_token
        
        encoded_datasets = []
        for dataset in [train_dataset, val_dataset, test_dataset]:
            encoded_dataset = dataset.map(
                lambda e: tokenizer(
                    e['text'],
                    padding=True,
                    truncation=True,
                    max_length=350
                ),
                batched=True,
                batch_size=len(dataset)
            )
            encoded_dataset = encoded_dataset.remove_columns(['text'])
            encoded_dataset = encoded_dataset.remove_columns(['label_text'])
            encoded_dataset = encoded_dataset[:len(encoded_dataset)]
            encoded_datasets.append(encoded_dataset)
        
        from src.loaders.dataloader import TextDataset
        
        train_ds = TextDataset(encoded_datasets[0])
        val_ds = TextDataset(encoded_datasets[1])
        test_ds = TextDataset(encoded_datasets[2])
        
        concept_names = ['negative', 'positive']
        task_names = [f"w_{i}" for i in range(len(tokenizer.get_vocab().keys()))]
        
        return train_ds, val_ds, test_ds, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_cebab_dataset(cfg, batch_size, **kwargs):
        """Create CEBaB dataset - returns loaders directly"""
        loader = CEBaBDataset(cfg.text_backbone_name, batch_size)
        loaded_train, loaded_val, loaded_test = loader.collator()
        
        concept_names = [
            'food_negative', 'food_unknown', 'food_positive',
            'ambiance_negative', 'ambiance_unknown', 'ambiance_positive',
            'service_negative', 'service_unknown', 'service_positive',
            'noise_negative', 'noise_unknown', 'noise_positive'
        ]
        task_names = ['review']
        concept_groups = {
            'food': [0, 1, 2],
            'ambiance': [3, 4, 5],
            'service': [6, 7, 8],
            'noise': [9, 10, 11]
        }
        
        return loaded_train, loaded_val, loaded_test, DatasetMetadata(
            concept_names, task_names, concept_groups
        )
    
    @staticmethod
    def create_cifar_dataset(name, concept_idxs_cifar, **kwargs):
        """Create CIFAR10 or CIFAR100 dataset"""
        if name == 'cifar10':
            train_dataset, test_dataset = get_CIFAR10_CBM_dataloader(
                DATA_PATH, concept_idxs_cifar
            )
        else:  # cifar100
            train_dataset, test_dataset = get_CIFAR100_CBM_dataloader(
                DATA_PATH, concept_idxs_cifar
            )
        
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Read concept and task names from files
        with open(f"{DATA_PATH}{name}/{name}_filtered.txt", "r") as file:
            all_concept_names = [line.strip() for line in file]
        
        with open(f"{DATA_PATH}{name}/{name}_classes.txt", "r") as file:
            task_names = [line.strip() for line in file]
        
        # Filter concept names based on selected indices
        concept_names = [all_concept_names[i] for i in concept_idxs_cifar]
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_pendulum_dataset(
        dataset_already_created, batch_size, num_workers, **kwargs
    ):
        """Create Pendulum dataset - returns loaders directly"""
        loader = PendulumDataset(
            already_created=dataset_already_created,
            batch_size=batch_size
        )
        loaded_train, loaded_val, loaded_test = loader.collator(
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,
        )
        
        concept_names = concept_names_pendulum
        task_names = task_names_pendulum
        
        return loaded_train, loaded_val, loaded_test, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_dsprites_dataset(
        selected_concepts, formulas, n_samples, seed, **kwargs
    ):
        """Create DSprites dataset"""
        dsprites_dataset = DSprites(
            concepts=selected_concepts,
            formulas=formulas,
            split='train',
            num_samples=n_samples,
            random_seed=seed
        )
        
        total_size = len(dsprites_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dsprites_dataset, [train_size, val_size, test_size]
        )
        
        concept_names = selected_concepts
        task_names = ['custom_target']
        
        return train_dataset, val_dataset, test_dataset, DatasetMetadata(
            concept_names, task_names, None
        )
    
    @staticmethod
    def create_mawps_dataset(cfg, batch_size, seed, **kwargs):
        """Create MAWPS dataset - returns loaders directly"""
        loader = MAWPSDataset(
            cfg.dataset.loader.dataset_already_created,
            batch_size=batch_size,
            shuffle_seed=seed,
            pre_trained_transformer=cfg.dataset.metadata.pretrained_transformer
        )
        loaded_train, loaded_val, loaded_test = loader.collator()
        
        concept_names = mawps_concept_names
        task_names = mawps_task_names
        
        return loaded_train, loaded_val, loaded_test, DatasetMetadata(
            concept_names, task_names, None
        )


def get_dataset(name, **kwargs):
    """
    Main factory function to get dataset splits and metadata.
    
    Args:
        name: Name of the dataset
        **kwargs: Dataset-specific parameters
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, metadata)
    """
    factory = DatasetFactory()
    
    # Map dataset names to factory methods
    if name in ['xor', 'trigonometry', 'dot', 'checkmark', 'or', 'nor', 'xnor']:
        return factory.create_toy_dataset(name, **kwargs)
    elif name == 'mnist_addition':
        return factory.create_mnist_addition(**kwargs)
    elif name == 'mnist_arithmetic':
        return factory.create_mnist_arithmetic(**kwargs)
    elif name == 'mnist_arithmetic_hard':
        return factory.create_mnist_arithmetic(hard_mode=True, **kwargs)
    elif name == 'mnist_exponential':
        return factory.create_mnist_exponential(**kwargs)
    elif name in ['cub', 'cub_incomplete']:
        return factory.create_cub_dataset(**kwargs)
    elif name == 'celeba':
        return factory.create_celeba_dataset(**kwargs)
    elif name in ['awa2', 'awa2_incomplete']:
        return factory.create_awa2_dataset(**kwargs)
    elif name == 'sst2':
        return factory.create_sst2_dataset(**kwargs)
    elif name == 'cebab':
        return factory.create_cebab_dataset(**kwargs)
    elif name in ['cifar10', 'cifar100']:
        return factory.create_cifar_dataset(name, **kwargs)
    elif name == 'pendulum':
        return factory.create_pendulum_dataset(**kwargs)
    elif name in ['dsprites_simple', 'dsprites_complex']:
        return factory.create_dsprites_dataset(**kwargs)
    elif name == 'mawps':
        return factory.create_mawps_dataset(**kwargs)
    else:
        raise ValueError(f"Dataset {name} not recognized.")
