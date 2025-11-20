import os
import av
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor, TimesformerModel

try:
    from env import DATA_PATH
except:
    import sys
    from pathlib import Path
    # Add the project root to path (3 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from env import DATA_PATH

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = os.path.join(DATA_PATH, "synthetic_motion")
CLIP_LEN = 8  # Number of frames to sample
FRAME_SAMPLE_RATE = 4  # Sample every n-th frame
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
PROCESSOR_NAME = "MCG-NJU/videomae-base"

# -----------------------------
# VIDEO PROCESSING FUNCTIONS
# -----------------------------
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    
    # If video is shorter than required, sample uniformly across available frames
    if seg_len <= converted_len:
        indices = np.linspace(0, seg_len - 1, num=clip_len)
        indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    else:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    
    return indices


def extract_embedding_from_video(video_path, model, image_processor, device):
    '''
    Extract embedding from a video file using TimeSformer.
    Args:
        video_path (str): Path to the video file.
        model: TimeSformer model.
        image_processor: Image processor for the model.
        device: Device to run the model on.
    Returns:
        embedding (np.ndarray): Extracted embedding.
    '''
    container = av.open(video_path)
    
    # Get total number of frames by actually counting them
    frames_list = []
    for frame in container.decode(video=0):
        frames_list.append(frame.to_ndarray(format="rgb24"))
    container.close()
    
    num_frames = len(frames_list)
    
    # Sample exactly CLIP_LEN frames uniformly from the video
    if num_frames >= CLIP_LEN:
        # Sample uniformly
        indices = np.linspace(0, num_frames - 1, num=CLIP_LEN).astype(int)
        video_frames = [frames_list[i] for i in indices]
    else:
        # If video is shorter than CLIP_LEN, repeat last frame to reach CLIP_LEN
        video_frames = frames_list.copy()
        while len(video_frames) < CLIP_LEN:
            video_frames.append(frames_list[-1])
    
    # Ensure we have exactly CLIP_LEN frames
    assert len(video_frames) == CLIP_LEN, f"Expected {CLIP_LEN} frames but got {len(video_frames)}"
    
    # Prepare video for the model
    inputs = image_processor(video_frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
    
    # Use the CLS token embedding (first token)
    embedding = last_hidden_states[:, 0, :].cpu().numpy()
    
    return embedding.squeeze()


# -----------------------------
# MAIN EXTRACTION FUNCTION
# -----------------------------
def extract_embeddings_for_dataset(output_file="embeddings.npz"):
    """
    Extract embeddings for all videos in the dataset and save them.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print("Loading TimeSformer model...")
    image_processor = AutoImageProcessor.from_pretrained(PROCESSOR_NAME)
    model = TimesformerModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.to(device)
    model.eval()
    
    # Load splits
    splits_path = Path(DATASET_DIR) / "splits.json"
    with open(splits_path, "r") as f:
        splits = json.load(f)
    
    # Process all samples
    embeddings = {}
    annotations = {}
    
    # Get all sample indices
    all_indices = splits["train"] + splits["val"] + splits["test"]
    
    print(f"Extracting embeddings for {len(all_indices)} videos...")
    
    for idx in tqdm(all_indices):
        # Load annotation
        ann_path = Path(DATASET_DIR) / "annotations" / f"sample_{idx}.json"
        with open(ann_path, "r") as f:
            ann = json.load(f)
        
        # Get video path
        video_path = Path(DATASET_DIR) / ann["video"]
        
        # Extract embedding
        try:
            embedding = extract_embedding_from_video(
                str(video_path), model, image_processor, device
            )
            embeddings[f"sample_{idx}"] = embedding
            annotations[f"sample_{idx}"] = ann
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue
    
    # Save embeddings and annotations
    output_path = Path(DATASET_DIR) / output_file
    np.savez(
        output_path,
        **embeddings
    )
    
    # Save annotations separately
    ann_output_path = Path(DATASET_DIR) / "embeddings_annotations.json"
    with open(ann_output_path, "w") as f:
        json.dump(annotations, f, indent=4)
    
    print(f"\nEmbeddings saved to {output_path}")
    print(f"Annotations saved to {ann_output_path}")
    print(f"Total embeddings extracted: {len(embeddings)}")
    print(f"Embedding shape: {list(embeddings.values())[0].shape}")


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
    
    if "--load" in sys.argv:
        # Test loading
        print("Loading embeddings...")
        embeddings, annotations = load_embeddings()
        print(f"Loaded {len(embeddings)} embeddings")
        print(f"Sample embedding shape: {list(embeddings.values())[0].shape}")
        print(f"\nSample annotation:")
        print(json.dumps(list(annotations.values())[0], indent=2))
    else:
        # Extract embeddings
        extract_embeddings_for_dataset()
