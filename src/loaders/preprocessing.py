import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from transformers import ViTModel
from torchvision.models import resnet34
from tqdm import tqdm
import torch.nn.functional as F

class EmbeddingExtractor:
    def __init__(self, 
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 device='cuda', 
                 celeba=False, 
                 selected_concepts=None,
                 task_names=None):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.celeba = celeba
        self.selected_concepts = selected_concepts
        self.task_names = task_names

        # Load ViT model pre-trained on ImageNet
        #self.model = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')
        # Load ResNet34 model pre-trained on ImageNet
        self.model = resnet34(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _extract_embeddings(self, loader):
        """Helper function to extract embeddings for a given DataLoader."""
        embeddings = []
        concepts_list = []
        labels = []

        with torch.no_grad():
            #if not self.celeba:
            for images, concepts, targets in tqdm(loader):
                bsz = images.shape[0]
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
                # Get the [CLS] token representation
                #outputs = outputs.last_hidden_state[:, 0, :]
                outputs = outputs.flatten(start_dim=1)
                embeddings.append(outputs.cpu())
                if self.celeba:
                    targets = self.batch_binary_to_decimal_torch(torch.stack([targets[:,i] \
                                                                              for i in range(len(self.task_names))], dim=1))
                    labels.append(targets.cpu())
                else:
                    labels.append(targets.cpu())
                concepts_list.append(concepts.cpu())

            # else:
            #     for images, concepts, targets in tqdm(loader):
            #         images = images.to(self.device)
            #         # Extract embeddings
            #         outputs = self.model(images)
            #         # Get the [CLS] token representation
            #         #outputs = outputs.last_hidden_state[:, 0, :]
            #         outputs = outputs.flatten(start_dim=1)
            #         embeddings.append(outputs)
            #         concepts_list.append(concepts)
            #         labels.append(targets)
                
        # Concatenate all embeddings and labels
        embeddings = torch.cat(embeddings, dim=0)
        concepts = torch.cat(concepts_list, dim=0)
        labels = torch.cat(labels, dim=0)

        if len(labels.shape)>1:
            labels = labels.squeeze()

        return embeddings, concepts.float(), labels

    def _create_loader(self, embeddings, concepts, labels, batch_size):
        """Helper function to create a DataLoader from embeddings and labels."""
        dataset = TensorDataset(embeddings, concepts, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def batch_binary_to_decimal_torch(self, binary_matrix):
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
