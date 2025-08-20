import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModel
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from tqdm import tqdm
import torch.nn.functional as F

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
        elif 'vit' in self.img_backbone_name:
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
                images = batch['x']#.to(self.device)
                concepts = batch['c']#.to(self.device)
                targets = batch['y']#.to(self.device)
                bsz = images.shape[0]
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

                    if 'vit' in self.cfg.img_backbone_name:
                        outputs = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
                    else:
                        outputs = outputs.flatten(start_dim=1)
                    embeddings.append(outputs.cpu())
                else:
                    # If embeddings are not extracted, just append the images
                    embeddings.append(images.cpu())
                if self.celeba:
                    targets = self._batch_binary_to_decimal_torch(
                        torch.stack([targets[:,i] for i in range(len(self.task_names))], dim=1)
                    )
                labels.append(targets.cpu())
                concepts_list.append(concepts.cpu())
                
        # Concatenate all embeddings and labels
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
        self.model_name = cfg.text_backbone_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.extract_embeddings = extract_embeddings
        self.task_names = task_names

        self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)

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
            for batch in tqdm(loader, desc="Extracting embeddings"):
                if self.extract_embeddings:
                    outputs = self.model(
                        input_ids=batch['x']["input_ids"].to(self.model.device).long(),
                        token_type_ids=batch['x']["token_type_ids"].to(self.model.device).long(),
                        attention_mask=batch['x']["attention_mask"].to(self.model.device).long()
                    )
                    if 'sentence-transformers' not in self.model_name:
                        emb = outputs.last_hidden_state  # shape: (B, L, D)
                        # Use the [CLS] token representation. This is useful to reduce the overall number of
                        # parameters of the model while preserving expressivity in the embeddings.
                        emb = emb[:, 0, :]  # shape: (B, D)
                    else:
                        # Perform pooling
                        emb = self._mean_pooling(
                            outputs, 
                            batch['x']["attention_mask"].to(self.model.device).long()
                        )

                        # Normalize embeddings
                        emb = F.normalize(emb, p=2, dim=1)
                    embeddings.append(emb.cpu())
                else:
                    # If the embedding is not produced, then the input of the model will be
                    # the raw text input.
                    input_ids.append(batch['x']["input_ids"].cpu())
                    attention_masks.append(batch['x']["attention_mask"].cpu())
                    token_type_ids.append(batch['x']["token_type_ids"].cpu())

                # append the remaining fields
                concepts.append(batch['c'].cpu())
                if "label" in batch:
                    labels.append(batch["label"].cpu())
                else:
                    labels.append(batch['y'].cpu())

            # Stack everything
            if self.extract_embeddings:
                input = torch.cat(embeddings, dim=0)
            else:
                input_ids = torch.cat(input_ids, dim=0)
                attention_masks = torch.cat(attention_masks, dim=0)
                token_type_ids = torch.cat(token_type_ids, dim=0)
                input = {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "token_type_ids": token_type_ids
                }
            concepts = torch.cat(concepts, dim=0)
            labels = torch.cat(labels, dim=0) if labels else None

        return input, concepts, labels


    def _create_loader(self, x, c, y, batch_size, use_custom_format=False):
        """Helper function to create a DataLoader from embeddings and labels."""
        if use_custom_format:
            # TODO: implement a custom dataset format
            dataset = TextEmbeddingDataset(embeddings, attention_masks, labels, input_ids)
            return DataLoader(dataset, batch_size=batch_size)
        else:
            if self.extract_embeddings:
                dataset = [{'x': _x.float(), 'c': _c, 'y': _y} for _x, _c, _y in zip(x, c, y)]
            else:
                dataset = [{'x': {'input_ids': input_ids.long(), 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, 'c': _c, 'y': _y} 
                                for input_ids, attention_mask, token_type_ids, _c, _y in zip(x['input_ids'], x['attention_mask'], x['token_type_ids'], c, y)]
            return DataLoader(dataset, batch_size=batch_size)

    def produce_loaders(self, selected_concepts=None, task_names=None):
        """Produces new DataLoaders with embeddings instead of raw text."""
        (train_embeddings, train_concepts, train_labels) = self._extract_embeddings(self.train_loader)
        (val_embeddings, val_concepts, val_labels) = self._extract_embeddings(self.val_loader)
        (test_embeddings, test_concepts, test_labels) = self._extract_embeddings(self.test_loader)

        batch_size = self.train_loader.batch_size

        use_custom_format = True if self.cfg.dataset.metadata.name == 'sst2' else False

        train_loader = self._create_loader(train_embeddings, train_concepts, train_labels, batch_size, use_custom_format) # be sure to shuffle the data prior to this step
        val_loader = self._create_loader(val_embeddings, val_concepts, val_labels, batch_size, use_custom_format)
        test_loader = self._create_loader(test_embeddings, test_concepts, test_labels, batch_size, use_custom_format)

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
