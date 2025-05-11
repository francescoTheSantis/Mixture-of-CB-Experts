from torch_concepts.data import ToyDataset
import torch
from torch import nn
from torch_concepts.data.mnist import MNISTAddition
from torch_concepts.data.cub import CUBDataset
from torch_concepts.data.cub import SELECTED_CONCEPTS as cub_selected_concepts
from torch_concepts.data.cub import CONCEPT_SEMANTICS as cub_concept_semantics
from torch_concepts.data.cub import CLASS_NAMES as cub_class_names
from torch_concepts.data.cub import CONCEPT_GROUP_MAP
from torch_concepts.data.celeba import CelebADataset
from torch.utils.data import DataLoader, random_split
from env import DATA_PATH
from src.loaders.preprocessing import EmbeddingExtractor
import omegaconf
from torchvision import transforms

CUB_CONCEPT_NAMES = [x for i, x in enumerate(cub_concept_semantics) if i in cub_selected_concepts]

class loader(object):
    def __init__(self, 
                 name,
                 batch_size,
                 num_workers,
                 device,
                 selected_concepts=None,
                 class_attributes=None,
                 ):
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device[0] if isinstance(device, omegaconf.listconfig.ListConfig) else device
        self.selected_concepts = selected_concepts
        self.task_names = class_attributes
        self.concept_groups = None

    def get_names(self):
        # Get the concept names and task names
        if self.name in ['xor', 'trigonometry', 'dot', 'checkmark']:
            dataset = ToyDataset(self.name, size=1000, random_state=42)
            concept_names = dataset.concept_attr_names
            task_names = ['0', '1']
            concept_groups = None
        elif self.name in ['mnist_addition']:
            train_dataset = MNISTAddition(root=DATA_PATH, train=True)
            concept_names = train_dataset.concept_names
            task_names = train_dataset.task_names
            concept_groups = None
        elif self.name == 'cub':
            concept_names = CUB_CONCEPT_NAMES
            task_names = cub_class_names
            concept_groups = CONCEPT_GROUP_MAP
        elif self.name == 'celeba':
            concept_names = self.selected_concepts
            task_names = ["class_"+str(x) for x in range(2**len(self.task_names))]
            concept_groups = None
        else:
            raise ValueError(f"Dataset {self.name} not recognized.")
        
        return concept_names, task_names, concept_groups

    def load_data(self):
        # Load the data
        if self.name in ['xor', 'trigonometry', 'dot', 'checkmark']:
            dataset = ToyDataset(self.name, size=1000, random_state=42)
            # split the dataset
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [0.7, 0.1, 0.2],
                generator=torch.Generator().manual_seed(42)
            )
        elif self.name == 'mnist_addition':
            train_dataset = MNISTAddition(root=DATA_PATH, train=True)
            test_dataset = MNISTAddition(root=DATA_PATH, train=False)
            # Split the dataset into train, validation and test sets
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, 
                                              [train_size, val_size])
        elif self.name == 'cub':
            train_dataset = CUBDataset(root=DATA_PATH, split='train')
            val_dataset = CUBDataset(root=DATA_PATH, split='val')
            test_dataset = CUBDataset(root=DATA_PATH, split='test')
        elif self.name == 'celeba':
            celeba_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            train_dataset = CelebADataset(root=DATA_PATH, split='train', 
                                          class_attributes=self.task_names,
                                          transform=celeba_transform)
            test_dataset = CelebADataset(root=DATA_PATH, split='test', 
                                         class_attributes=self.task_names,
                                         transform=celeba_transform)
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, 
                                              [train_size, val_size])
        else:
            raise ValueError(f"Dataset {self.name} not recognized.")

        # create the dataloaders
        loaded_train = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True,
                                  num_workers=self.num_workers)
        loaded_val = DataLoader(val_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=False,
                                num_workers=self.num_workers)
        loaded_test = DataLoader(test_dataset, 
                                 batch_size=self.batch_size, 
                                 shuffle=False,
                                 num_workers=self.num_workers)  
        
        if self.name in ['cub', 'mnist_addition', 'celeba']:
            celeba_flag = True if self.name == 'celeba' else False
            E_extr = EmbeddingExtractor(loaded_train, 
                                        loaded_val, 
                                        loaded_test, 
                                        self.device,
                                        celeba_flag,
                                        self.selected_concepts,
                                        self.task_names,
                                        )
            loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()

        return loaded_train, loaded_val, loaded_test
