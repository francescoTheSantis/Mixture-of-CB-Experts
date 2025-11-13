"""
Data Loader Module

This module provides a clean, maintainable data loader that delegates dataset creation
to factory functions and focuses on post-processing and batching.
"""

import torch
from torch.utils.data import DataLoader
import omegaconf
from torchvision import transforms
import os
import itertools
import random

from env import DATA_PATH
from src.loaders.preprocessing import EmbeddingExtractor, TextEmbeddingExtractor
from src.loaders.dataset_factory import get_dataset
from src.loaders.datasets.cub import (
    SELECTED_CONCEPTS as cub_selected_concepts,
    CONCEPT_SEMANTICS as cub_concept_semantics,
    CONCEPT_GROUP_MAP as cub_concept_groups,
)
from src.loaders.datasets.awa2 import (
    CONCEPT_SEMANTICS as awa2_concept_semantics,
    CONCEPT_GROUPS as awa2_concept_groups,
)
from src.utilities import get_type_from_name, sanitize_concept_names


class TextDataset(torch.utils.data.Dataset):
    """Dataset wrapper for text data"""
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in
             self.encoded_text.items()}
        return t

    def __len__(self):
        return len(self.encoded_text['input_ids'])


CUB_CONCEPT_NAMES = [
    x for i, x in enumerate(cub_concept_semantics) if i in cub_selected_concepts
]


class loader(object):
    """
    Data loader class to manage loading, preprocessing, and batching of various datasets.
    
    This class delegates dataset creation to factory functions and focuses on
    post-processing (e.g., embedding extraction) and batching. Dataset-specific
    logic is handled by the dataset factory module.
    """
    
    def __init__(
        self,
        name,
        batch_size,
        num_workers,
        device,
        selected_concepts=None,
        selected_concept_groups=None,
        concept_percentage=None,
        class_attributes=None,
        extract_embeddings=True,
        data_path=None,
        dataset_already_created=False,
        formulas=None,
        seed=42,
        n_samples=None,
        latent_dim=None,
        noise_std=None,
        train_autoencoder=True,
        use_stored_dataset=False,
    ):
        """Initialize the loader with dataset parameters."""
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.selected_concepts = selected_concepts
        self.selected_concept_groups = selected_concept_groups
        self.concept_percentage = concept_percentage
        self.task_names = class_attributes
        self.extract_embeddings = extract_embeddings
        self.data_path = data_path
        self.dataset_already_created = dataset_already_created
        self.formulas = formulas
        self.seed = seed
        self.n_samples = n_samples
        
        # Symbolic regression parameters
        self.latent_dim = latent_dim
        self.noise_std = noise_std if noise_std is not None else 0.0
        self.train_autoencoder = train_autoencoder
        self.use_stored_dataset = use_stored_dataset
        self.device = device[0] if isinstance(device, omegaconf.listconfig.ListConfig) else device
        self.concept_groups = None

        # Standard transform for image datasets
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Initialize dataset-specific parameters
        self._prepare_dataset_parameters()
    
    def _prepare_dataset_parameters(self):
        """
        Prepare dataset-specific parameters that will be passed to the factory.
        This handles special cases like incomplete datasets and concept filtering.
        """
        # Initialize common attributes
        self.selected_concept_idxes = None
        self.incomplete_cub_groups = None
        self.incomplete_awa2_groups = None
        self.concept_idxs_cifar = None
        self.CUB_CONCEPT_NAMES = None
        
        # Handle CUB with concept percentage
        if self.name == 'cub' and self.concept_percentage is not None:
            selected_concepts = random.sample(
                CUB_CONCEPT_NAMES, 
                int(len(CUB_CONCEPT_NAMES) * self.concept_percentage)
            )
            self.selected_concept_idxes = [
                CUB_CONCEPT_NAMES.index(x) for x in selected_concepts
            ]
            
            # Filter groups according to selected concepts
            self.incomplete_cub_groups = {}
            for k, v in cub_concept_groups.items():
                idxs = [idx for idx in self.selected_concept_idxes if idx in v]
                if len(idxs) > 0:
                    self.incomplete_cub_groups[k] = idxs
            
            self.CUB_CONCEPT_NAMES = selected_concepts
        
        # Handle CUB incomplete with concept groups
        if self.selected_concept_groups is not None and self.name == 'cub_incomplete':
            # Filter groups by selected group names
            self.incomplete_cub_groups = {
                k: v for k, v in cub_concept_groups.items() 
                if k in self.selected_concept_groups
            }
            # Get concept indexes from selected groups
            self.selected_concept_idxes = list(itertools.chain.from_iterable(
                self.incomplete_cub_groups.values()
            ))
            self.CUB_CONCEPT_NAMES = [
                x for i, x in enumerate(cub_concept_semantics) 
                if i in self.selected_concept_idxes
            ]
            # Reset indexes in groups
            cnt = 0
            for k, v in self.incomplete_cub_groups.items():
                self.incomplete_cub_groups[k] = list(range(len(v)))
                cnt += len(v)
        
        # Handle AWA2 incomplete
        if self.selected_concepts is not None and self.name == 'awa2_incomplete':
            self.selected_concept_idxes = [
                awa2_concept_semantics.index(x) for x in self.selected_concepts
            ]
            
            # Filter groups according to selected concepts
            self.incomplete_awa2_groups = {}
            for k, v in awa2_concept_groups.items():
                idxs = [idx for idx in self.selected_concept_idxes if idx in v]
                if len(idxs) > 0:
                    self.incomplete_awa2_groups[k] = idxs
            
            # Reset indexes in groups
            cnt = 0
            for k, v in self.incomplete_awa2_groups.items():
                self.incomplete_awa2_groups[k] = list(range(len(v)))
                cnt += len(v)
        
        # Handle CIFAR10/100
        if self.name in ['cifar10', 'cifar100']:
            # Read concept and task names
            with open(f"{DATA_PATH}{self.name}/{self.name}_filtered.txt", "r") as file:
                concept_list = [line.strip() for line in file]
            self.concept_names_cifar = concept_list
            
            with open(f"{DATA_PATH}{self.name}/{self.name}_classes.txt", "r") as file:
                task_label_list = [line.strip() for line in file]
            self.task_names_cifar = task_label_list
            
            # Generate concept indexes
            concept_idxs = {
                self.concept_names_cifar.index(x): x 
                for x in self.concept_names_cifar
            }
            
            # Reduce concept size if percentage is set
            if self.concept_percentage is not None:
                filtered_idxs = random.sample(
                    range(0, len(self.concept_names_cifar)),
                    int(len(self.concept_names_cifar) * self.concept_percentage)
                )
                filtered_idxs.sort()
                self.concept_names_cifar = [concept_idxs[x] for x in filtered_idxs]
                concept_idxs = {
                    idx: name for idx, name in zip(filtered_idxs, self.concept_names_cifar)
                }
            
            self.concept_idxs_cifar = list(sorted(concept_idxs.keys()))

    def get_names(self, cfg=None):
        """
        Get concept names, task names, and concept groups for the dataset.
        
        This method delegates to the dataset factory to retrieve metadata,
        eliminating hard-coded logic.
        
        Args:
            cfg: Configuration object (required for some datasets)
        
        Returns:
            tuple: (concept_names, task_names, concept_groups)
        """
        # Build parameters for factory
        params = self._build_factory_params()
        
        # Add cfg-specific parameters
        if cfg is not None:
            if self.name == 'cebab':
                params['cfg'] = cfg
            if self.name == 'mawps':
                params['cfg'] = cfg
        
        # Get dataset and metadata from factory
        _, _, _, metadata = get_dataset(self.name, **params)
        
        # Ensure concept names are SymPy-safe before returning. We sanitize
        # names by replacing spaces and special characters with underscores,
        # collapsing repeated underscores, and prefixing an underscore if the
        # name starts with a digit. This preserves readable names (e.g.
        # "eye-color" -> "eye_color") rather than mapping to generic c0, c1.
        safe_c_names, _ = sanitize_concept_names(metadata.concept_names)
        return safe_c_names, metadata.task_names, metadata.concept_groups
    
    def _build_factory_params(self):
        """Build parameters dictionary for the dataset factory"""
        params = {
            'seed': self.seed,
            'data_path': self.data_path,
            'n_samples': self.n_samples,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'dataset_already_created': self.dataset_already_created,
        }
        
        # Add dataset-specific parameters
        if self.name in ['cub', 'cub_incomplete']:
            params['selected_concept_idxes'] = self.selected_concept_idxes
            params['concept_percentage'] = self.concept_percentage
            params['incomplete_cub_groups'] = self.incomplete_cub_groups
        
        if self.name in ['awa2', 'awa2_incomplete']:
            params['selected_concept_idxes'] = self.selected_concept_idxes
            params['incomplete_awa2_groups'] = self.incomplete_awa2_groups
        
        if self.name == 'celeba':
            params['task_names'] = self.task_names
            params['transform'] = self.transform
        
        if self.name in ['cifar10', 'cifar100']:
            params['concept_idxs_cifar'] = self.concept_idxs_cifar
        
        if self.name in ['dsprites_simple', 'dsprites_complex']:
            params['selected_concepts'] = self.selected_concepts
            params['formulas'] = self.formulas
        
        # Add symbolic regression parameters
        if self._is_symbolic_regression():
            params['latent_dim'] = getattr(self, 'latent_dim', 4)
            params['noise_std'] = getattr(self, 'noise_std', 0.0)
            params['train_autoencoder'] = getattr(self, 'train_autoencoder', True)
            params['use_stored_dataset'] = getattr(self, 'use_stored_dataset', False)
        
        return params
    
    def _is_symbolic_regression(self):
        """Check if the dataset is a symbolic regression dataset"""
        return self.name.startswith('feynman_')

    def load_data(self, cfg=None):
        """
        Load and process dataset splits.
        
        This method uses the dataset factory to create splits, then applies
        post-processing such as embedding extraction and batching.
        
        Args:
            cfg: Configuration object (required for some datasets)
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        # Build parameters for factory
        params = self._build_factory_params()
        
        # Add cfg-specific parameters
        if cfg is not None:
            if self.name == 'cebab':
                params['cfg'] = cfg
            if self.name == 'mawps':
                params['cfg'] = cfg
        
        # Get datasets from factory
        train_data, val_data, test_data, _ = get_dataset(self.name, **params)
        
        # Some datasets return loaders directly (text datasets, pendulum, etc.)
        returns_loaders = self.name in ['cebab', 'pendulum', 'mawps']
        
        if not returns_loaders:
            # Create DataLoaders for datasets that return Dataset objects
            loaded_train = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                pin_memory=True,
                collate_fn=self._custom_collate_fn
            )
            loaded_val = DataLoader(
                val_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                pin_memory=True,
                collate_fn=self._custom_collate_fn
            )
            loaded_test = DataLoader(
                test_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=False,
                collate_fn=self._custom_collate_fn
            )
        else:
            # These already return loaders
            loaded_train, loaded_val, loaded_test = train_data, val_data, test_data
        
        # Apply embedding extraction if needed
        # Note: cfg is required for embedding extraction
        if cfg is not None:
            if get_type_from_name(self.name) == 'image' and self.extract_embeddings:
                celeba_flag = True if self.name == 'celeba' else False
                E_extr = EmbeddingExtractor(
                    cfg,
                    loaded_train,
                    loaded_val,
                    loaded_test,
                    self.device,
                    celeba_flag,
                    self.task_names,
                    self.extract_embeddings
                )
                loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
            elif get_type_from_name(self.name) == 'text':
                E_extr = TextEmbeddingExtractor(
                    cfg,
                    loaded_train,
                    loaded_val,
                    loaded_test,
                    self.device,
                    self.extract_embeddings
                )
                loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
        
        return loaded_train, loaded_val, loaded_test

    def _custom_collate_fn(self, batch):
        """
        Custom collate function to handle different data types in the batch.
        """
        batch_dict = {}
        for idx, key in enumerate(['x', 'c', 'y']):
            values = [item[idx] for item in batch]
            if isinstance(values[0], torch.Tensor):
                batch_dict[key] = torch.stack(values)
            else:
                batch_dict[key] = torch.tensor(values)
        return batch_dict
