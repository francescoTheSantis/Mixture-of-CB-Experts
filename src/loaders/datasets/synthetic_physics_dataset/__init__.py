"""
Synthetic Physics Dataset Package

This package contains tools for generating and loading synthetic physics video datasets
for concept-based reasoning experiments.
"""

from .synthetic_motion_dataset import (
    SyntheticMotionDataset,
    get_synthetic_motion_loaders,
    ensure_dataset_ready,
    CONCEPT_NAMES,
    TASK_NAMES
)

__all__ = [
    'SyntheticMotionDataset',
    'get_synthetic_motion_loaders',
    'ensure_dataset_ready',
    'CONCEPT_NAMES',
    'TASK_NAMES'
]
