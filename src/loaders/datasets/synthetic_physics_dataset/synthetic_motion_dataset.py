import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

try:
    from env import DATA_PATH
except:
    import sys
    from pathlib import Path
    # Add the project root to path (3 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from env import DATA_PATH


# Define concept and task names for the dataset
CONCEPT_NAMES = ['x_0', 'y_0', 'x_t', 'y_t', 't']
TASK_NAMES = ['v_t']


def check_dataset_exists(dataset_dir):
    """
    Check if dataset videos and embeddings exist.
    
    Returns:
        tuple: (videos_exist, embeddings_exist)
    """
    videos_dir = dataset_dir / "videos"
    splits_file = dataset_dir / "splits.json"
    embeddings_file = dataset_dir / "embeddings.npz"
    
    videos_exist = videos_dir.exists() and splits_file.exists()
    embeddings_exist = embeddings_file.exists()
    
    return videos_exist, embeddings_exist


def ensure_dataset_ready(n_samples=300, acceleration_values=None, dataset_dir=None, force_regenerate=False):
    """
    Ensure dataset is generated and embeddings are extracted.
    
    Args:
        n_samples: Number of samples to generate
        acceleration_values: List of acceleration values
        dataset_dir: Dataset directory (defaults to DATA_PATH/synthetic_motion)
        force_regenerate: If True, regenerate even if dataset exists
    
    Returns:
        dataset_dir: Path to the dataset directory
    """
    if dataset_dir is None:
        dataset_dir = Path(DATA_PATH) / "synthetic_motion"
    else:
        dataset_dir = Path(dataset_dir)
    
    if acceleration_values is None:
        acceleration_values = [0.5]
    
    videos_exist, embeddings_exist = check_dataset_exists(dataset_dir)
    
    # Generate videos if needed
    if not videos_exist or force_regenerate:
        print(f"Generating synthetic motion dataset with {n_samples} samples...")
        from . import synthetic_motion
        synthetic_motion.generate_dataset(
            n_samples=n_samples,
            output_dir=str(dataset_dir),
            acceleration_values=acceleration_values
        )
        videos_exist = True
        embeddings_exist = False  # Need to regenerate embeddings
    
    # Extract embeddings if needed
    if not embeddings_exist or force_regenerate:
        print("Extracting TimeSformer embeddings from videos...")
        from . import extract_embeddings
        # Temporarily set the dataset directory
        original_dataset_dir = extract_embeddings.DATASET_DIR
        extract_embeddings.DATASET_DIR = str(dataset_dir)
        try:
            extract_embeddings.extract_embeddings_for_dataset()
        finally:
            extract_embeddings.DATASET_DIR = original_dataset_dir
    
    print(f"Dataset ready at {dataset_dir}")
    return dataset_dir


class SyntheticMotionDataset(Dataset):
    """
    PyTorch Dataset for synthetic motion videos with TimeSformer embeddings.
    Returns data in the format expected by the framework:
    - x: video embedding
    - c: concepts [x_0, y_0, x_t, y_t, t]
    - y: target [v_t]
    """
    
    def __init__(self, split="train", embeddings_file="embeddings.npz", dataset_dir=None):
        """
        Args:
            split (str): One of 'train', 'val', or 'test'.
            embeddings_file (str): Name of the embeddings file.
            dataset_dir (str): Path to dataset directory. If None, uses DATA_PATH/synthetic_motion.
        """
        if dataset_dir is None:
            self.dataset_dir = Path(DATA_PATH) / "synthetic_motion"
        else:
            self.dataset_dir = Path(dataset_dir)
        
        self.split = split
        self.embeddings_file = embeddings_file
        
        # Store paths for lazy loading
        self.embeddings_path = self.dataset_dir / embeddings_file
        self.ann_path = self.dataset_dir / "embeddings_annotations.json"
        
        # Load splits
        splits_path = self.dataset_dir / "splits.json"
        with open(splits_path, "r") as f:
            splits = json.load(f)
        
        self.sample_indices = splits[split]
        
        # Initialize embeddings and annotations as None (will be loaded lazily)
        self._embeddings = None
        self._annotations = None
        
        # Get list of available samples by checking embeddings file
        data = np.load(self.embeddings_path)
        available_samples = set(data.files)
        data.close()
        
        # Filter to only include samples in this split that have embeddings
        self.sample_names = [
            f"sample_{idx}" for idx in self.sample_indices 
            if f"sample_{idx}" in available_samples
        ]
        
        print(f"Loaded {len(self.sample_names)} samples for {split} split")
    
    @property
    def embeddings(self):
        """Lazy load embeddings to handle multiprocessing properly"""
        if self._embeddings is None:
            data = np.load(self.embeddings_path)
            self._embeddings = {key: data[key] for key in data.files}
        return self._embeddings
    
    @property
    def annotations(self):
        """Lazy load annotations to handle multiprocessing properly"""
        if self._annotations is None:
            with open(self.ann_path, "r") as f:
                self._annotations = json.load(f)
        return self._annotations
    
    def __getstate__(self):
        """Custom pickle state to handle multiprocessing"""
        state = self.__dict__.copy()
        # Don't pickle the loaded data, only the paths
        state['_embeddings'] = None
        state['_annotations'] = None
        # Convert Path objects to strings for pickling
        state['embeddings_path'] = str(state['embeddings_path'])
        state['ann_path'] = str(state['ann_path'])
        state['dataset_dir'] = str(state['dataset_dir'])
        return state
    
    def __setstate__(self, state):
        """Custom unpickle to restore state in worker processes"""
        # Convert string paths back to Path objects (if they exist)
        if 'embeddings_path' in state:
            state['embeddings_path'] = Path(state['embeddings_path'])
        if 'ann_path' in state:
            state['ann_path'] = Path(state['ann_path'])
        if 'dataset_dir' in state:
            state['dataset_dir'] = Path(state['dataset_dir'])
        
        self.__dict__.update(state)
        
        # Reconstruct missing attributes from older pickled versions
        if not hasattr(self, 'embeddings_path'):
            self.embeddings_path = self.dataset_dir / self.embeddings_file
        if not hasattr(self, 'ann_path'):
            self.ann_path = self.dataset_dir / "embeddings_annotations.json"
        
        # Ensure _embeddings and _annotations are initialized
        if not hasattr(self, '_embeddings'):
            self._embeddings = None
        if not hasattr(self, '_annotations'):
            self._annotations = None
    
    def __len__(self):
        return len(self.sample_names)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (x, c, y, video_idx) where:
                - x: video embedding (tensor)
                - c: concepts [x_0, y_0, x_t, y_t, t] (tensor)
                - y: target [v_t] (tensor)
                - video_idx: index of the video (int)
        """
        sample_name = self.sample_names[idx]
        
        # Get embedding (x)
        x = torch.from_numpy(self.embeddings[sample_name]).float()
        
        # Get annotation
        ann = self.annotations[sample_name]
        
        # Build concepts: [x_0, y_0, x_t, y_t, t]
        c = torch.tensor([
            ann["initial_position"][0],  # x_0
            ann["initial_position"][1],  # y_0
            ann["final_position"][0],    # x_t
            ann["final_position"][1],    # y_t
            ann["time_final"]             # t
        ], dtype=torch.float32)
        
        # Build target: v_t (scalar, not list)
        y = torch.tensor(ann["velocity_final"], dtype=torch.float32)
        
        # Extract video index from sample_name (e.g., "sample_123" -> 123)
        video_idx = int(sample_name.split('_')[1])
        
        return x, c, y, video_idx


def collate_fn(batch):
    """
    Custom collate function to convert batch of tuples to dictionary format.
    Expected by the framework: {'x': embeddings, 'c': concepts, 'y': targets, 'video_idx': indices}
    """
    x_batch, c_batch, y_batch, idx_batch = zip(*batch)
    return {
        'x': torch.stack(x_batch),
        'c': torch.stack(c_batch),
        'y': torch.stack(y_batch),
        'video_idx': torch.tensor(idx_batch, dtype=torch.long)
    }


# Function to create loaders (used by the dataloader module)
def get_synthetic_motion_loaders(
    batch_size=32, 
    num_workers=4, 
    embeddings_file="embeddings.npz",
    n_samples=300,
    acceleration_values=None,
    dataset_already_created=False
):
    """
    Create dataloaders for the synthetic motion dataset.
    Automatically generates videos and extracts embeddings if they don't exist.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        embeddings_file: Name of the embeddings file
        n_samples: Number of samples to generate (if not already created)
        acceleration_values: List of acceleration values (if not already created)
        dataset_already_created: If True, skip generation check
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset_dir = Path(DATA_PATH) / "synthetic_motion"
    
    # Ensure dataset is ready (generate + extract embeddings if needed)
    if not dataset_already_created:
        ensure_dataset_ready(
            n_samples=n_samples,
            acceleration_values=acceleration_values,
            dataset_dir=dataset_dir,
            force_regenerate=False
        )
    
    train_dataset = SyntheticMotionDataset(split="train", embeddings_file=embeddings_file)
    val_dataset = SyntheticMotionDataset(split="val", embeddings_file=embeddings_file)
    test_dataset = SyntheticMotionDataset(split="test", embeddings_file=embeddings_file)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader


# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    # Create datasets
    train_dataset = SyntheticMotionDataset(split="train")
    val_dataset = SyntheticMotionDataset(split="val")
    test_dataset = SyntheticMotionDataset(split="test")
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # Get a sample
    x, c, y = train_dataset[0]
    print(f"\nSample format:")
    print(f"x (embedding) shape: {x.shape}")
    print(f"c (concepts) shape: {c.shape}")
    print(f"c (concepts) values: {c}")
    print(f"  [x_0, y_0, x_t, y_t, t] = {CONCEPT_NAMES}")
    print(f"y (target) shape: {y.shape}")
    print(f"y (target) value: {y.item():.3f}")
    
    # Test DataLoader
    train_loader, val_loader, test_loader = get_synthetic_motion_loaders(batch_size=4)
    
    for batch_x, batch_c, batch_y in train_loader:
        print(f"\nBatch shapes:")
        print(f"x (embeddings): {batch_x.shape}")
        print(f"c (concepts): {batch_c.shape}")
        print(f"y (targets): {batch_y.shape}")
        break

