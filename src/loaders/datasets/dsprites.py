import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from typing import List, Callable, Optional, Dict, Any
import sympy
import sympytorch

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
        split: str = "train"
    ):
        """
        Initialize the DSprites dataset.
        
        Args:
            concepts: List of concept names to extract (e.g., ['value_y_position', 'value_orientation'])
            formula: Function that takes a dict of concept values and returns the target value
            split: Dataset split to use ('train' by default)
        """
        self.dataset = load_dataset("dpdl-benchmark/dsprites")[split]
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
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Dictionary containing:
            - 'image': The image as a tensor
            - 'concepts': Dictionary with concept values
            - 'target': Target value computed using the formula
        """
        sample = self.dataset[idx]
        
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

        return (image, concept_values, torch.tensor(target, dtype=torch.float32), shape)

if __name__ == "__main__":
    # Example 1: Using exponential formula with y_position
    dataset1 = DSprites(
        concepts=['value_orientation', 'value_x_position', 'value_y_position'],
        formulas={
            'square': 'cos(value_orientation)', 
            'circle': 'exp(10*(value_x_position + value_y_position))', 
            'heart': 'sin(value_orientation * value_x_position) + cos(value_y_position^2) * exp(-0.5 * (value_x_position - value_y_position)^2)'},
    )

    for i in range(30):
        image, concepts, target, shape = dataset1[i]
        print(f"Sample {i}: Concepts: {concepts}, Target: {target.item()}, Shape: {shape}")
