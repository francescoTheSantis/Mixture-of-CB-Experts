import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from typing import List, Callable, Optional, Dict, Any
import sympy
import sympytorch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from env import HOME

def load_dsprites_dataset():
    """
    Download and load the dsprites dataset from Hugging Face.
    
    Returns:
        dataset: The loaded dsprites dataset
    """
    dataset = load_dataset("dpdl-benchmark/dsprites")
    return dataset

class DSprites(Dataset):
    """
    DSprites dataset class that allows setting a formula to combine concepts
    for computing a target variable.
    """
    
    def __init__(
        self,
        concepts: List[str],
        formulas: Callable[[Dict[str, str]], str],
        split: str = "train", # The huggingface repo contains only the train split
        num_samples: Optional[int] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the DSprites dataset.
        
        Args:
            concepts: List of concept names to extract (e.g., ['value_y_position', 'value_orientation'])
            formula: Function that takes a dict of concept values and returns the target value
            split: Dataset split to use ('train' by default)
            num_samples: Number of samples to randomly select from the dataset (optional)
            random_seed: Seed for random number generator (optional)
        """
        self.dataset = load_dataset("dpdl-benchmark/dsprites")[split]
        # Subsample indices if requested
        full_indices = np.arange(len(self.dataset))
        if num_samples is not None:
            rng = np.random.default_rng(random_seed)
            self._indices = rng.choice(full_indices, size=num_samples, replace=False)
        else:
            self._indices = full_indices
        self._num_samples = len(self._indices)
        self.formulas = formulas
        self.available_concepts = [col for col in self.dataset.column_names if col.startswith('value_')]

        if concepts is None:
            # Use all concepts
            self.concepts = self.available_concepts
        else:
            self.concepts = concepts
            # Validate that all concepts exist in the dataset
            for concept in concepts:
                if concept not in self.available_concepts:
                    raise ValueError(f"Concept '{concept}' not found. Available concepts: {self.available_concepts}")
            # Sort the concepts according to available concepts
            self.concepts = sorted(self.concepts, key=lambda x: self.available_concepts.index(x))

        # Shapes dictionary index: shape
        self.ids_to_shapes = {1: 'square', 2: 'circle', 3: 'heart'}

        # create sympy variables with the selected concepts
        self.sympy_vars = sympy.symbols([c for c in self.concepts])

        self.torch_formulas = {}
        # We are going to have a formula for each shape
        for shape, formula in self.formulas.items():
            torch_exp = sympytorch.SymPyModule(expressions=[sympy.sympify(formula)])
            self.torch_formulas[shape] = torch_exp

    def __len__(self) -> int:
        return self._num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
                num_samples: Optional[int] = None,
                random_seed: Optional[int] = None
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
            - 'image': The image as a tensor
            - 'concepts': Dictionary with concept values
            - 'target': Target value computed using the formula
                    num_samples: Number of samples to randomly select from the dataset (optional)
                    random_seed: Seed for random number generator (optional)
        """

        real_idx = int(self._indices[idx])
        sample = self.dataset[real_idx]
        
        # Extract image and convert to tensor
        image = torch.tensor(np.array(sample['image']), dtype=torch.float32)
        
        # Extract concept values
        concept_values = torch.tensor([sample[c] for c in self.available_concepts if c in self.concepts], dtype=torch.float32)

        # Get shapes
        shape = sample['value_shape']
        shape = self.ids_to_shapes[shape]

        # Compute target using the formula
        var_dict = dict(zip(self.concepts, [concept_values[i] for i in range(concept_values.shape[0])]))
        target = self.torch_formulas[shape](**var_dict)

        return (image.unsqueeze(0), concept_values, torch.tensor(target, dtype=torch.float32), shape)


def plot_samples(root, dataset, num_samples=10, figsize=(15, 3)):
    """
    Plot sample images from the DSprites dataset.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        image, concepts, target, shape = dataset[i]
        # Remove the channel dimension and convert to numpy for plotting
        img_array = image.squeeze(0).numpy()
        axes[i].imshow(img_array, cmap='gray')
        axes[i].set_title(f'{shape}\nTarget: {target.item():.3f}' + \
                          f'\nConcepts: {", ".join([f"{c}: {concepts[j].item():.2f}" for j, c in enumerate(dataset.concepts)])}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{root}/dsprites_samples.pdf")
    plt.show()


if __name__ == "__main__":
    
    # Create the directory continaing the dsprties figures
    root = f"{HOME}/figs/dsprites"
    os.makedirs(root, exist_ok=True)

    # Example 1: Using exponential formula with y_position
    dataset1 = DSprites(
        num_samples = 10000,
        random_seed = 1,
        concepts=['value_orientation', 'value_x_position', 'value_y_position'],
        formulas={
            'square': 'exp(-((value_orientation - 3.14)^2)/3)', 
            'circle': 'exp(-((value_orientation - 3.14)^2)/3)', 
            'heart': 'exp(-((value_orientation - 3.14)^2)/3)'},
    )

    # Plot samples to visualize the dataset
    plot_samples(root, dataset1, num_samples=10)

    # Print some sample information
    for i in range(100):
        image, concepts, target, shape = dataset1[i]
        print(f"Sample {i}: Concepts: {concepts}, Target: {target.item():.3f}, Shape: {shape}")
