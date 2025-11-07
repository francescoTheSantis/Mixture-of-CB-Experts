"""
Symbolic Regression Datasets Module

This module provides datasets for symbolic regression tasks within the Concept-Based framework.
Each dataset follows the structure:
- x: latent representation (learned via autoencoder)
- c: high-level concepts (interpretable variables in the equation)
- y: target variable (output of the equation)

The autoencoder learns a compressed representation of the concept space, providing
a latent embedding that can be used by the encoder model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import os
import pickle
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from env import HOME, DATA_PATH


class SimpleAutoencoder(nn.Module):
    """
    Simple autoencoder for learning latent representations of concept variables.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [max(input_dim // 2, latent_dim * 2)]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        return self.encoder(x)


class SymbolicRegressionDataset(Dataset):
    """
    Base class for symbolic regression datasets in the Concept-Based framework.
    
    This class generates synthetic data for symbolic regression problems where:
    - Concepts (c) are the input variables to symbolic equations
    - Target (y) is the output of the equation
    - Latent (x) is learned via an autoencoder trained on concepts
    
    Args:
        num_samples: Number of samples to generate
        equation_func: Function that computes y from concepts
        concept_ranges: List of (min, max) tuples for each concept variable
        concept_names: Names of the concept variables
        latent_dim: Dimension of the latent representation
        noise_std: Standard deviation of Gaussian noise added to targets
        train_autoencoder: Whether to train the autoencoder
        autoencoder_epochs: Number of epochs for autoencoder training
        autoencoder_lr: Learning rate for autoencoder training
        cache_dir: Directory to cache trained autoencoders
        random_seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_samples: int,
        equation_func: Callable,
        concept_ranges: List[Tuple[float, float]],
        concept_names: List[str],
        latent_dim: int = 4,
        noise_std: float = 0.0,
        train_autoencoder: bool = True,
        autoencoder_epochs: int = 100,
        autoencoder_lr: float = 0.001,
        cache_dir: Optional[str] = None,
        random_seed: Optional[int] = None,
        dataset_name: str = "symbolic_regression",
        use_stored_dataset: bool = False,
        data_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.num_samples = num_samples
        self.equation_func = equation_func
        self.concept_ranges = concept_ranges
        self.concept_names = concept_names
        self.num_concepts = len(concept_names)
        self.latent_dim = latent_dim
        self.noise_std = noise_std
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.use_stored_dataset = use_stored_dataset
        self.data_path = data_path
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        
        # Generate concept values
        self.concepts = self._generate_concepts()
        
        # Compute targets
        self.targets = self._compute_targets()
        
        # Remove equation_func after targets are computed (no longer needed)
        del self.equation_func
        
        # Setup cache directory for autoencoder
        if cache_dir is None:
            cache_dir = os.path.join(DATA_PATH, 'symbolic_regression', 'autoencoders')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Train or load autoencoder (only if not using stored processed dataset)
        # The autoencoder is only needed to generate latents for the first time
        if not use_stored_dataset:
            self.autoencoder = self._setup_autoencoder(
                train_autoencoder, autoencoder_epochs, autoencoder_lr
            )
            
            # Generate latent representations
            self.latents = self._generate_latents()
        else:
            # When use_stored_dataset=True, latents will be loaded from the main.py cache
            # We don't need to train/load the autoencoder or generate latents here
            # Just create placeholder that won't be used
            self.autoencoder = None
            self.latents = None
    
    def _generate_concepts(self) -> torch.Tensor:
        """Generate random concept values within specified ranges."""
        concepts = torch.zeros(self.num_samples, self.num_concepts)
        for i, (min_val, max_val) in enumerate(self.concept_ranges):
            concepts[:, i] = torch.rand(self.num_samples) * (max_val - min_val) + min_val
        return concepts
    
    def _compute_targets(self) -> torch.Tensor:
        """Compute target values using the equation function."""
        targets = self.equation_func(self.concepts)
        
        # Ensure targets are 1D (batch,) not (batch, 1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        return targets
    
    def _get_autoencoder_cache_path(self) -> Path:
        """Get the cache path for the autoencoder."""
        cache_name = f"{self.dataset_name}_latent{self.latent_dim}_concepts{self.num_concepts}.pkl"
        return self.cache_dir / cache_name
    
    def _setup_autoencoder(
        self, 
        train: bool, 
        epochs: int, 
        lr: float
    ) -> SimpleAutoencoder:
        """Setup autoencoder by loading from cache or training new one."""
        cache_path = self._get_autoencoder_cache_path()
        
        # Try to load from cache
        if cache_path.exists() and not train:
            print(f"Loading autoencoder from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                autoencoder = pickle.load(f)
            return autoencoder
        
        # Train new autoencoder
        print(f"Training autoencoder for {self.dataset_name}...")
        autoencoder = SimpleAutoencoder(
            input_dim=self.num_concepts,
            latent_dim=self.latent_dim
        )
        
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        autoencoder.train()
        batch_size = min(256, self.num_samples)
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, self.num_samples, batch_size):
                batch = self.concepts[i:i+batch_size]
                
                optimizer.zero_grad()
                reconstructed, _ = autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / (self.num_samples / batch_size)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        autoencoder.eval()
        
        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(autoencoder, f)
        print(f"Autoencoder saved to cache: {cache_path}")
        
        return autoencoder
    
    def _generate_latents(self) -> torch.Tensor:
        """Generate latent representations using the autoencoder."""
        with torch.no_grad():
            latents = self.autoencoder.encode(self.concepts)
            # Add noise if specified
            if self.noise_std > 0:
                noise = torch.randn_like(latents) * self.noise_std
                latents = latents + noise
        return latents
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (latent, concepts, target)
        """
        if self.latents is not None:
            return self.latents[idx], self.concepts[idx], self.targets[idx]
        else:
            # When use_stored_dataset=True, return dummy latent
            # The actual latents will be loaded from cache in main.py
            return torch.zeros(self.latent_dim), self.concepts[idx], self.targets[idx]


# ============================================================================
# Specific Symbolic Regression Datasets (Feynman-inspired and common benchmarks)
# ============================================================================

class FeynmanDataset(SymbolicRegressionDataset):
    """
    Base class for Feynman symbolic regression datasets.
    """
    EQUATIONS = {}  # To be filled by subclasses
    
    @classmethod
    def get_equation_info(cls, equation_id: str) -> Dict:
        """Get equation information by ID."""
        return cls.EQUATIONS.get(equation_id)


class Feynman_I_6_2(FeynmanDataset):
    """Feynman I.6.2: exp(-(theta/sigma)**2/2) / (sqrt(2*pi) * sigma)"""
    
    def __init__(self, num_samples: int = 10000, latent_dim: int = 2, **kwargs):
        concept_names = ['theta', 'sigma']
        concept_ranges = [(-3.0, 3.0), (1.0, 3.0)]
        
        def equation(c):
            theta, sigma = c[:, 0], c[:, 1]
            return torch.exp(-(theta/sigma)**2/2) / (torch.sqrt(2*torch.tensor(np.pi)) * sigma)
        
        super().__init__(
            num_samples=num_samples,
            equation_func=equation,
            concept_ranges=concept_ranges,
            concept_names=concept_names,
            latent_dim=latent_dim,
            dataset_name="feynman_I_6_2",
            **kwargs
        )


class Feynman_I_9_18(FeynmanDataset):
    """Feynman I.9.18: G*m1*m2/((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)"""
    
    def __init__(self, num_samples: int = 10000, latent_dim: int = 4, **kwargs):
        concept_names = ['G', 'm1', 'm2', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2']
        concept_ranges = [
            (0.5, 1.5), (0.5, 2.0), (0.5, 2.0),  # G, m1, m2
            (-1.0, -0.5), (0.5, 1.0),  # x1, x2
            (-1.0, -0.5), (0.5, 1.0),  # y1, y2
            (-1.0, -0.5), (0.5, 1.0)   # z1, z2
        ]
        
        def equation(c):
            G = c[:, 0]
            m1, m2 = c[:, 1], c[:, 2]
            x1, x2 = c[:, 3], c[:, 4]
            y1, y2 = c[:, 5], c[:, 6]
            z1, z2 = c[:, 7], c[:, 8]
            return G * m1 * m2 / ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        
        super().__init__(
            num_samples=num_samples,
            equation_func=equation,
            concept_ranges=concept_ranges,
            concept_names=concept_names,
            latent_dim=latent_dim,
            dataset_name="feynman_I_9_18",
            **kwargs
        )


class Feynman_I_12_1(FeynmanDataset):
    """Feynman I.12.1: mu * Nn (simple multiplication)"""
    
    def __init__(self, num_samples: int = 10000, latent_dim: int = 2, **kwargs):
        concept_names = ['mu', 'Nn']
        concept_ranges = [(-2.0, 2.0), (-2.0, 2.0)]
        
        def equation(c):
            mu, Nn = c[:, 0], c[:, 1]
            return mu * Nn
        
        super().__init__(
            num_samples=num_samples,
            equation_func=equation,
            concept_ranges=concept_ranges,
            concept_names=concept_names,
            latent_dim=latent_dim,
            dataset_name="feynman_I_12_1",
            **kwargs
        )


class Feynman_I_13_4(FeynmanDataset):
    """Feynman I.13.4: (1/2) * m * (v**2 + u**2 + w**2) (kinetic energy)"""
    
    def __init__(self, num_samples: int = 10000, latent_dim: int = 3, **kwargs):
        concept_names = ['m', 'v', 'u', 'w']
        concept_ranges = [(0.5, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
        
        def equation(c):
            m = c[:, 0]
            v, u, w = c[:, 1], c[:, 2], c[:, 3]
            return 0.5 * m * (v**2 + u**2 + w**2)
        
        super().__init__(
            num_samples=num_samples,
            equation_func=equation,
            concept_ranges=concept_ranges,
            concept_names=concept_names,
            latent_dim=latent_dim,
            dataset_name="feynman_I_13_4",
            **kwargs
        )


class Feynman_I_14_3(FeynmanDataset):
    """Feynman I.14.3: m * g * z (potential energy)"""
    
    def __init__(self, num_samples: int = 10000, latent_dim: int = 2, **kwargs):
        concept_names = ['m', 'g', 'z']
        concept_ranges = [(0.1, 2.0), (9.0, 11.0), (-10.0, 10.0)]
        
        def equation(c):
            m, g, z = c[:, 0], c[:, 1], c[:, 2]
            return m * g * z
        
        super().__init__(
            num_samples=num_samples,
            equation_func=equation,
            concept_ranges=concept_ranges,
            concept_names=concept_names,
            latent_dim=latent_dim,
            dataset_name="feynman_I_14_3",
            **kwargs
        )


class Feynman_I_15_10(FeynmanDataset):
    """Feynman I.15.10: m0 * v / sqrt(1 - v**2/c**2) (relativistic momentum)"""
    
    def __init__(self, num_samples: int = 10000, latent_dim: int = 2, **kwargs):
        concept_names = ['m0', 'v', 'c']
        concept_ranges = [(0.5, 2.0), (-0.8, 0.8), (1.2, 2.0)]
        
        def equation(c):
            m0, v, c_val = c[:, 0], c[:, 1], c[:, 2]
            return m0 * v / torch.sqrt(1 - v**2/c_val**2)
        
        super().__init__(
            num_samples=num_samples,
            equation_func=equation,
            concept_ranges=concept_ranges,
            concept_names=concept_names,
            latent_dim=latent_dim,
            dataset_name="feynman_I_15_10",
            **kwargs
        )


# ============================================================================
# Dataset Registry
# ============================================================================

SYMBOLIC_REGRESSION_DATASETS = {
    # Feynman datasets
    'feynman_I_6_2': Feynman_I_6_2,
    'feynman_I_9_18': Feynman_I_9_18,
    'feynman_I_12_1': Feynman_I_12_1,
    'feynman_I_13_4': Feynman_I_13_4,
    'feynman_I_14_3': Feynman_I_14_3,
    'feynman_I_15_10': Feynman_I_15_10,
}


def get_symbolic_dataset(name: str, **kwargs):
    """
    Factory function to get a symbolic regression dataset by name.
    
    Args:
        name: Name of the dataset
        **kwargs: Additional arguments passed to the dataset constructor
    
    Returns:
        SymbolicRegressionDataset instance
    """
    if name not in SYMBOLIC_REGRESSION_DATASETS:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: "
            f"{list(SYMBOLIC_REGRESSION_DATASETS.keys())}"
        )
    
    dataset_class = SYMBOLIC_REGRESSION_DATASETS[name]
    return dataset_class(**kwargs)


# ============================================================================
# Concept names for datasets (for metadata)
# ============================================================================

SYMBOLIC_CONCEPT_NAMES = {
    'feynman_I_6_2': ['theta', 'sigma'],
    'feynman_I_9_18': ['G', 'm1', 'm2', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2'],
    'feynman_I_12_1': ['mu', 'Nn'],
    'feynman_I_13_4': ['m', 'v', 'u', 'w'],
    'feynman_I_14_3': ['m', 'g', 'z'],
    'feynman_I_15_10': ['m0', 'v', 'c'],
}
