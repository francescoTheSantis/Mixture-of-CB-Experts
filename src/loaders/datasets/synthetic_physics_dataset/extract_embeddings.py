import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

try:
    from env import DATA_PATH
except:
    import sys
    from pathlib import Path
    # Add the project root to path (3 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from env import DATA_PATH

# Import the VideoEmbeddingExtractor from preprocessing
try:
    from src.loaders.preprocessing import VideoEmbeddingExtractor
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.loaders.preprocessing import VideoEmbeddingExtractor

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = os.path.join(DATA_PATH, "synthetic_motion")

# -----------------------------
# MAIN EXTRACTION FUNCTION
# -----------------------------
def extract_embeddings_for_dataset(img_backbone_name="facebook/dinov2-base", output_file="embeddings.npz"):
    """
    Extract embeddings for all videos in the dataset and save them.
    
    Args:
        img_backbone_name (str): Name of the image backbone model to use.
        output_file (str): Name of the output file.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a minimal config object for VideoEmbeddingExtractor
    class Config:
        def __init__(self, backbone_name):
            self.img_backbone_name = backbone_name
    
    cfg = Config(img_backbone_name)
    
    # Initialize the VideoEmbeddingExtractor
    extractor = VideoEmbeddingExtractor(cfg, device=device)
    
    # Load splits
    splits_path = Path(DATASET_DIR) / "splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    # Get all sample indices
    all_indices = splits["train"] + splits["val"] + splits["test"]
    
    # Load annotations and prepare video paths
    print("Loading annotations and preparing video paths...")
    video_paths = {}
    annotations = {}
    
    for idx in tqdm(all_indices, desc="Loading annotations"):
        ann_path = Path(DATASET_DIR) / "annotations" / f"sample_{idx}.json"
        with open(ann_path, "r") as f:
            ann = json.load(f)
        
        video_path = Path(DATASET_DIR) / ann["video"]
        video_paths[f"sample_{idx}"] = video_path
        annotations[f"sample_{idx}"] = ann
    
    # Extract embeddings using the VideoEmbeddingExtractor
    metadata = extractor.extract_embeddings_for_dataset(
        video_paths=video_paths,
        output_file=output_file,
        dataset_dir=DATASET_DIR
    )
    
    # Save annotations separately
    ann_output_path = Path(DATASET_DIR) / "embeddings_annotations.json"
    with open(ann_output_path, "w") as f:
        json.dump(annotations, f, indent=4)
    
    # Save metadata
    metadata_path = Path(DATASET_DIR) / "embeddings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Annotations saved to {ann_output_path}")
    print(f"Metadata saved to {metadata_path}")


def load_embeddings(embeddings_file="embeddings.npz"):
    """
    Load embeddings and annotations from saved files.
    Returns:
        embeddings (dict): Dictionary mapping sample names to embeddings.
        annotations (dict): Dictionary mapping sample names to annotations.
    """
    embeddings_path = Path(DATASET_DIR) / embeddings_file
    ann_path = Path(DATASET_DIR) / "embeddings_annotations.json"
    
    # Load embeddings
    data = np.load(embeddings_path)
    embeddings = {key: data[key] for key in data.files}
    
    # Load annotations
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    
    return embeddings, annotations


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract video embeddings using image backbone')
    parser.add_argument('--load', action='store_true', help='Test loading embeddings')
    parser.add_argument('--img_backbone_name', type=str, default='facebook/dinov2-base',
                        help='Image backbone model name (e.g., resnet18, facebook/dinov2-base)')
    parser.add_argument('--output_file', type=str, default='embeddings.npz',
                        help='Output filename for embeddings')
    
    args = parser.parse_args()
    
    if args.load:
        # Test loading
        print("Loading embeddings...")
        embeddings, annotations = load_embeddings(args.output_file)
        print(f"Loaded {len(embeddings)} embeddings")
        print(f"Sample embedding shape: {list(embeddings.values())[0].shape}")
        
        # Load metadata if available
        metadata_path = Path(DATASET_DIR) / "embeddings_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"\nMetadata:")
            print(json.dumps(metadata, indent=2))
        
        print(f"\nSample annotation:")
        print(json.dumps(list(annotations.values())[0], indent=2))
    else:
        # Extract embeddings
        extract_embeddings_for_dataset(
            img_backbone_name=args.img_backbone_name,
            output_file=args.output_file
        )
