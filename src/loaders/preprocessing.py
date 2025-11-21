import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoProcessor, AutoModel, AutoImageProcessor
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from tqdm.auto import tqdm
import torch.nn.functional as F
import pandas as pd
import os
import av
import numpy as np
from PIL import Image
from pathlib import Path

class EmbeddingExtractor:
    """
    Extracts image embeddings using a pre-trained backbone (Vision Transformer) and 
    produces DataLoaders containing the respective embeddings instead of the original images.
    """
    def __init__(self, 
                 cfg,
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 device='cuda', 
                 celeba=False, 
                 task_names=None,
                 extract_embeddings=True):

        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.celeba = celeba
        self.task_names = task_names
        self.img_backbone_name = cfg.img_backbone_name
        self.extract_embeddings = extract_embeddings

        if 'resnet' in self.img_backbone_name:
            if self.img_backbone_name == 'resnet18':
                self.model = resnet18(pretrained=True)
            elif self.img_backbone_name == 'resnet34':
                self.model = resnet34(pretrained=True)
            elif self.img_backbone_name == 'resnet50':
                self.model = resnet50(pretrained=True)
            elif self.img_backbone_name == 'resnet101':
                self.model = resnet101(pretrained=True)
            elif self.img_backbone_name == 'resnet152':
                self.model = resnet152(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            # save the latent dimension of the backbone's output
            self.latent_dim = self.model[-2][-1].bn2.num_features
        elif 'vit' in self.img_backbone_name or 'dino' in self.img_backbone_name:
            self.model = AutoModel.from_pretrained(self.img_backbone_name)
            self.latent_dim = self.model.config.hidden_size
        else:
            raise ValueError(f"Image backbone {self.img_backbone_name} not recognized.")

        self.model = self.model.to(self.device)
        self.model.eval()

    def _extract_embeddings(self, loader):
        """Helper function to extract embeddings for a given DataLoader."""
        embeddings = []
        concepts_list = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(loader):
                images = batch['x']
                concepts = batch['c']
                targets = batch['y']
                
                if self.extract_embeddings and not self.cfg.dataset.metadata.name=='xor':
                    images = images.to(self.device)
                    # If the tensor has not the correct shape 
                    if images.shape[-1] != 224:
                        images = F.interpolate(images, 
                                                size=(224, 224), 
                                                mode='bilinear', 
                                                align_corners=False)
                    if images.shape[1] == 1:
                        # Repeat the single channel 3 times to simulate RGB
                        images = images.repeat(1, 3, 1, 1)  # (N, 3, H, W)

                    # Extract embeddings
                    outputs = self.model(images)

                    if 'vit' in self.cfg.img_backbone_name or 'dino' in self.cfg.img_backbone_name:
                        outputs = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
                    else:
                        outputs = outputs.flatten(start_dim=1)
                    
                    # Move to CPU immediately to free GPU memory for next batch
                    embeddings.append(outputs.cpu())
                else:
                    # If embeddings are not extracted, just append the images (already on CPU)
                    embeddings.append(images)
                    
                if self.celeba:
                    targets = self._batch_binary_to_decimal_torch(
                        torch.stack([targets[:,i] for i in range(len(self.task_names))], dim=1)
                    )
                # Append CPU tensors
                labels.append(targets)
                concepts_list.append(concepts)
                
        # Concatenate all embeddings and labels (all already on CPU)
        embeddings = torch.cat(embeddings, dim=0)
        concepts = torch.cat(concepts_list, dim=0)
        labels = torch.cat(labels, dim=0)

        if len(labels.shape)>1:
            labels = labels.squeeze()

        return embeddings, concepts.float(), labels

    def _create_loader(self, embeddings, concepts, labels, batch_size):
        """Helper function to create a DataLoader from embeddings and labels."""
        dataset = [{'x': e, 'c': c, 'y': l} 
                   for e, c, l in zip(embeddings, concepts, labels)]
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _batch_binary_to_decimal_torch(self, binary_matrix):
        # Ensure binary_matrix is 2D (even if it has only one row)
        if binary_matrix.dim() == 1:
            binary_matrix = binary_matrix.unsqueeze(0)  # Add batch dimension

        # Compute powers of 2 dynamically based on input size
        powers_of_two = 2 ** torch.arange(binary_matrix.shape[1] - 1, -1, -1, 
                                          dtype=torch.float32, 
                                          device=binary_matrix.device)

        # Compute decimal values
        decimal_values = (binary_matrix * powers_of_two).sum(dim=1).long()
        return decimal_values
    
    def produce_loaders(self, selected_concepts=None, task_names=None):
        """Produces new DataLoaders with embeddings instead of raw images."""
        train_embeddings, train_concepts, train_labels = self._extract_embeddings(self.train_loader)
        val_embeddings, val_concepts, val_labels = self._extract_embeddings(self.val_loader)
        test_embeddings, test_concepts, test_labels = self._extract_embeddings(self.test_loader)

        batch_size = self.train_loader.batch_size

        train_loader = self._create_loader(train_embeddings, train_concepts, train_labels, batch_size)
        val_loader = self._create_loader(val_embeddings, val_concepts, val_labels, batch_size)
        test_loader = self._create_loader(test_embeddings, test_concepts, test_labels, batch_size)

        return train_loader, val_loader, test_loader


class TextEmbeddingExtractor:
    """
    Extracts text embeddings using a pre-trained backbone (e.g., Mistral) and
    produces DataLoaders containing the respective embeddings instead of the
    original text. It inherits from EmbeddingExtractor and overrides the
    _extract_embeddings method to handle text data.
    Args:
        train_loader (DataLoader): DataLoader for the training set, yielding (texts, concepts, targets).
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        device (str, optional): Device to run the model.
        task_names (list, optional): List of task names for multi-task settings. Default is None.
    Methods:
        _extract_embeddings(loader):
            Extracts embeddings, concepts, and labels from a given DataLoader.
    """
    def __init__(self,
                 cfg,
                 train_loader,
                 val_loader,
                 test_loader,
                 device='cuda',
                 extract_embeddings=None,
                 task_names=None):
        # Load a pre-trained text model (e.g., Mistral)
        self.cfg = cfg
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.extract_embeddings = extract_embeddings
        self.task_names = task_names

        self.model_name = cfg.text_backbone_name
        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, use_safetensors=True)

    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _extract_embeddings(self, loader):
        embeddings = []
        attention_masks = []
        labels = []
        input_ids = []
        token_type_ids = []
        concepts = []

        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                if self.extract_embeddings:
                    # Move inputs to device
                    input_ids_gpu = batch['x']["input_ids"].to(self.model.device).long()
                    attention_mask_gpu = batch['x']["attention_mask"].to(self.model.device).long()

                    token_type_ids_gpu = batch['x']["token_type_ids"].to(self.model.device).long()
                    outputs = self.model(
                        input_ids=input_ids_gpu,
                        token_type_ids=token_type_ids_gpu,
                        attention_mask=attention_mask_gpu
                    )
                    
                    if 'sentence-transformers' in self.model_name:
                        # NOTE: we decided to not use the [CLS] token representation for sentence-transformers
                        # but the concatenaion of the whole embeddings.

                        # emb = outputs.last_hidden_state  # shape: (B, L, D)
                        # Use the [CLS] token representation. This is useful to reduce the overall number of
                        # parameters of the model while preserving expressivity in the embeddings.
                        # emb = emb[:, 0, :]  # shape: (B, D)

                        # Extract the last hidden state
                        emb = outputs.last_hidden_state  # shape: (B, L, D)
                        # Flatten the embeddings
                        emb = emb.flatten(1).float()
                    else:
                        # Perform pooling
                        emb = self._mean_pooling(outputs, attention_mask_gpu)
                        # Normalize embeddings
                        emb = F.normalize(emb, p=2, dim=1)
                    
                    # Move to CPU immediately to free GPU memory for next batch
                    embeddings.append(emb.cpu())
                else:
                    # If the embedding is not produced, then the input of the model will be
                    # the raw text input (already on CPU from DataLoader).
                    if self.cfg.dataset.metadata.name == "mawps":
                        input_ids.append(batch['x']["input_ids"])
                        attention_masks.append(batch['x']["attention_mask"])
                    else:
                        input_ids.append(batch['x']["input_ids"])
                        attention_masks.append(batch['x']["attention_mask"])
                        token_type_ids.append(batch['x']["token_type_ids"])

                # append the remaining fields (already on CPU)
                concepts.append(batch['c'])
                if "label" in batch:
                    labels.append(batch["label"])
                else:
                    labels.append(batch['y'])

            # Stack everything (all already on CPU)
            if self.extract_embeddings:
                input = torch.cat(embeddings, dim=0)
            else:
                input_ids = torch.cat(input_ids, dim=0)
                attention_masks = torch.cat(attention_masks, dim=0)
                if self.cfg.dataset.metadata.name == "mawps":
                    input = {
                        "input_ids": input_ids,
                        "attention_mask": attention_masks
                    }
                else:
                    token_type_ids = torch.cat(token_type_ids, dim=0)
                    input = {
                        "input_ids": input_ids,
                        "attention_mask": attention_masks,
                        "token_type_ids": token_type_ids
                    }
            concepts = torch.cat(concepts, dim=0)
            labels = torch.cat(labels, dim=0) if labels else None

        return input, concepts, labels


    def _create_loader(self, x, c, y, batch_size):
        """Helper function to create a DataLoader from embeddings and labels."""
        if self.extract_embeddings:
            dataset = [{'x': _x.float(), 'c': _c, 'y': _y} for _x, _c, _y in tqdm(zip(x, c, y), total=len(x), desc="Creating dataset")]
        else:
            dataset = [{'x': {'input_ids': input_ids.long(), 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, 'c': _c, 'y': _y} 
                            for input_ids, attention_mask, token_type_ids, _c, _y in tqdm(zip(x['input_ids'], x['attention_mask'], x['token_type_ids'], c, y), total=len(x['input_ids']), desc="Creating dataset")]
        return DataLoader(dataset, batch_size=batch_size)

    def produce_loaders(self, selected_concepts=None, task_names=None):
        """Produces new DataLoaders with embeddings instead of raw text."""
        (train_embeddings, train_concepts, train_labels) = self._extract_embeddings(self.train_loader)
        (val_embeddings, val_concepts, val_labels) = self._extract_embeddings(self.val_loader)
        (test_embeddings, test_concepts, test_labels) = self._extract_embeddings(self.test_loader)

        batch_size = self.train_loader.batch_size

        train_loader = self._create_loader(train_embeddings, train_concepts, train_labels, batch_size) # be sure to shuffle the data prior to this step
        val_loader = self._create_loader(val_embeddings, val_concepts, val_labels, batch_size)
        test_loader = self._create_loader(test_embeddings, test_concepts, test_labels, batch_size)

        return train_loader, val_loader, test_loader

class TextEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, attention_mask, labels, input_ids):
        self.embeddings = embeddings
        self.attention_mask = attention_mask
        self.labels = labels
        self.input_ids = input_ids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        sample = {
            "embeddings": self.embeddings[idx],
            "attention_mask": self.attention_mask[idx],
            "concept": self.labels[idx],
            "task": self.input_ids[idx]
        }
        seq_len = sample["attention_mask"].shape[0]
        mask = sample["attention_mask"][:-1]

        # Create concept labels: first convert the concept to one-hot encoding
        concept_label = F.one_hot(sample["concept"], num_classes=self.labels.max()+1).float()
        # Shift the task labels to create next-token prediction labels
        concept_label = concept_label.repeat(seq_len, 1)[:-1, :].float()
        # Set to 0 where the attention mask is 0
        concept_label = torch.where(mask.unsqueeze(-1) == 0, -100, concept_label)

        # Create word labels: where the attention mask is 0, assign -100; otherwise, use the next token as the label
        word_label = torch.where(mask == 0, -100, sample["task"][1:])
        # Select all embeddings except the last one to align with the shifted labels
        # (the last token does not have a next token to predict)
        features = sample["embeddings"][:-1].float()

        assert features.shape[0] == concept_label.squeeze().shape[0],\
                "Features and concept labels must have the same sequence length"
        assert features.shape[0] == word_label.squeeze().shape[0],\
                "Features and word labels must have the same sequence length"

        return features, concept_label, word_label


class VideoEmbeddingExtractor:
    """
    Extracts video embeddings using frame-by-frame extraction with a pre-trained image backbone.
    Each video is processed frame by frame, and embeddings are concatenated with zero-padding
    to match the maximum number of frames across all videos.
    
    Args:
        cfg: Configuration object containing img_backbone_name
        video_paths: List or dict of video file paths to process
        annotations: Dict of annotations for each video
        device (str): Device to run the model on ('cuda' or 'cpu')
    """
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.img_backbone_name = cfg.img_backbone_name
        
        # Load the backbone model
        self.model, self.latent_dim, self.processor = self._load_backbone_model()
        
    def _load_backbone_model(self):
        """
        Load the image backbone model based on the configuration.
        Returns:
            model: Loaded backbone model.
            latent_dim: Dimension of the output embeddings.
            processor: Image processor (if needed for transformers).
        """
        processor = None
        
        if 'resnet' in self.img_backbone_name:
            if self.img_backbone_name == 'resnet18':
                model = resnet18(pretrained=True)
            elif self.img_backbone_name == 'resnet34':
                model = resnet34(pretrained=True)
            elif self.img_backbone_name == 'resnet50':
                model = resnet50(pretrained=True)
            elif self.img_backbone_name == 'resnet101':
                model = resnet101(pretrained=True)
            elif self.img_backbone_name == 'resnet152':
                model = resnet152(pretrained=True)
            else:
                raise ValueError(f"ResNet model {self.img_backbone_name} not recognized.")
            
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            latent_dim = model[-2][-1].bn2.num_features
            
        elif 'vit' in self.img_backbone_name or 'dino' in self.img_backbone_name:
            model = AutoModel.from_pretrained(self.img_backbone_name)
            latent_dim = model.config.hidden_size
            processor = AutoImageProcessor.from_pretrained(self.img_backbone_name)
            
        else:
            raise ValueError(f"Image backbone {self.img_backbone_name} not recognized.")
        
        model = model.to(self.device)
        model.eval()
        
        return model, latent_dim, processor
    
    def read_all_frames_from_video(self, video_path):
        """
        Read all frames from a video file.
        Args:
            video_path (str): Path to the video file.
        Returns:
            frames_list (list): List of frames as numpy arrays (H, W, 3).
        """
        container = av.open(video_path)
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame.to_ndarray(format="rgb24"))
        container.close()
        return frames_list
    
    def extract_frame_embedding(self, frame_np):
        """
        Extract embedding from a single frame.
        Args:
            frame_np (np.ndarray): Frame as numpy array (H, W, 3).
        Returns:
            embedding (np.ndarray): Extracted embedding.
        """
        # Convert numpy array to PIL Image
        frame_pil = Image.fromarray(frame_np)
        
        # Preprocess the frame
        if 'vit' in self.img_backbone_name or 'dino' in self.img_backbone_name:
            inputs = self.processor(images=frame_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            # For ResNet, we need to convert to tensor and normalize
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
            # Resize to 224x224
            frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frame_tensor = (frame_tensor - mean) / std
            inputs = frame_tensor.to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            if 'vit' in self.img_backbone_name or 'dino' in self.img_backbone_name:
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                # ResNet output
                outputs = self.model(inputs)
                embedding = outputs.flatten(start_dim=1).cpu().numpy()
        
        return embedding.squeeze()
    
    def extract_embedding_from_video(self, video_path):
        """
        Extract embeddings from all frames in a video file.
        Args:
            video_path (str): Path to the video file.
        Returns:
            embeddings (list): List of embeddings for each frame.
            num_frames (int): Number of frames in the video.
        """
        frames_list = self.read_all_frames_from_video(video_path)
        num_frames = len(frames_list)
        
        embeddings = []
        for frame in frames_list:
            embedding = self.extract_frame_embedding(frame)
            embeddings.append(embedding)
        
        return embeddings, num_frames
    
    def extract_embeddings_for_dataset(self, video_paths, output_file, dataset_dir):
        """
        Extract embeddings for all videos in the dataset and save them.
        
        Args:
            video_paths (dict): Dictionary mapping sample names to video paths
            output_file (str): Name of the output file
            dataset_dir (str or Path): Dataset directory
        
        Returns:
            dict: Metadata about the embeddings
        """
        dataset_dir = Path(dataset_dir)
        
        print(f"Extracting embeddings using {self.img_backbone_name}...")
        
        # First pass: find maximum number of frames
        print("First pass: determining maximum number of frames...")
        max_frames = 0
        frame_counts = {}
        
        for sample_name, video_path in tqdm(video_paths.items(), desc="Counting frames"):
            try:
                frames_list = self.read_all_frames_from_video(str(video_path))
                num_frames = len(frames_list)
                frame_counts[sample_name] = num_frames
                max_frames = max(max_frames, num_frames)
            except Exception as e:
                print(f"\nError reading {sample_name}: {e}")
                continue
        
        print(f"Maximum number of frames: {max_frames}")
        
        # Second pass: extract embeddings and pad
        embeddings = {}
        
        print(f"Second pass: extracting embeddings for {len(video_paths)} videos...")
        
        for sample_name, video_path in tqdm(video_paths.items(), desc="Extracting embeddings"):
            try:
                frame_embeddings, num_frames = self.extract_embedding_from_video(str(video_path))
                
                # Convert list of embeddings to numpy array
                frame_embeddings = np.stack(frame_embeddings)  # Shape: (num_frames, latent_dim)
                
                # Pad with zeros if necessary
                if num_frames < max_frames:
                    padding = np.zeros((max_frames - num_frames, self.latent_dim), dtype=frame_embeddings.dtype)
                    frame_embeddings = np.concatenate([frame_embeddings, padding], axis=0)
                
                # Flatten to create final embedding: (max_frames * latent_dim,)
                final_embedding = frame_embeddings.flatten()
                
                embeddings[sample_name] = final_embedding
                
            except Exception as e:
                print(f"\nError processing {sample_name}: {e}")
                continue
        
        # Save embeddings
        output_path = dataset_dir / output_file
        np.savez(output_path, **embeddings)
        
        # Save metadata about the embeddings
        metadata = {
            "img_backbone_name": self.img_backbone_name,
            "latent_dim": int(self.latent_dim),
            "max_frames": int(max_frames),
            "embedding_shape": [int(max_frames), int(self.latent_dim)],
            "flattened_embedding_dim": int(max_frames * self.latent_dim)
        }
        
        print(f"\nEmbeddings saved to {output_path}")
        print(f"Total embeddings extracted: {len(embeddings)}")
        print(f"Embedding shape per video: ({max_frames}, {self.latent_dim}) -> flattened to ({max_frames * self.latent_dim},)")
        
        return metadata
