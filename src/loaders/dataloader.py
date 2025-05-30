from torch_concepts.data import ToyDataset
import torch
from torch import nn
from torch_concepts.data.mnist import MNISTAddition
from torch_concepts.data.cub import CUBDataset
from torch_concepts.data.cub import SELECTED_CONCEPTS as cub_selected_concepts
from torch_concepts.data.cub import CONCEPT_SEMANTICS as cub_concept_semantics
from torch_concepts.data.cub import CLASS_NAMES as cub_class_names
from torch_concepts.data.cub import CONCEPT_GROUP_MAP as cub_concept_groups
from torch_concepts.data.awa2 import AwA2Dataset
from torch_concepts.data.awa2 import CONCEPT_SEMANTICS as awa2_concept_semantics
from torch_concepts.data.awa2 import CLASS_NAMES as awa2_class_names
from torch_concepts.data.awa2 import CONCEPT_GROUPS as awa2_concept_groups
from torch_concepts.data.celeba import CelebADataset
from torch.utils.data import DataLoader, random_split
from env import DATA_PATH
from src.loaders.preprocessing import EmbeddingExtractor
import omegaconf
from torchvision import transforms
import os
import itertools

CUB_CONCEPT_NAMES = [x for i, x in enumerate(cub_concept_semantics) if i in cub_selected_concepts]

class loader(object):
    """
    Data loader class to manage loading, preprocessing, and batching of various datasets from PyC.

    Args:
        name (str): Name of the dataset to load.
        batch_size (int): Number of samples per batch in DataLoader.
        num_workers (int): Number of worker threads for loading data.
        device (str or ListConfig): Device identifier (e.g. 'cuda', 'cpu') or omegaconf ListConfig.
        selected_concepts (list, optional): Subset of concept names to use, if applicable.
        class_attributes (list, optional): List of class attribute names used for tasks (especially for CelebA).

    Attributes:
        name (str): Dataset name.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoader.
        device (str): Device to run on.
        selected_concepts (list): Selected concept names.
        task_names (list): Class attribute names used for task labels.
        concept_groups (dict or None): Mapping of concepts to groups, if available.
        transform (torchvision.transforms.Compose): Image preprocessing transformations.

    Methods:
        get_names():
            Returns concept names, task names, and concept groups corresponding to the dataset.
            Raises ValueError if dataset name is not recognized.

        load_data():
            Loads the dataset splits (train, validation, test) and returns DataLoaders.
            Applies transformations, splits data as needed, and optionally applies embedding extraction
            for certain datasets.
            Raises ValueError if dataset name is not recognized.
    """
    def __init__(self,
                 name,
                 batch_size,
                 num_workers,
                 device,
                 selected_concepts=None,
                 selected_concept_groups=None,
                 class_attributes=None,
                 extract_embeddings=True
                 ):
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device[0] if isinstance(device, omegaconf.listconfig.ListConfig) else device
        self.selected_concepts = selected_concepts
        self.selected_concept_groups = selected_concept_groups
        self.task_names = class_attributes
        self.concept_groups = None
        self.extract_embeddings = extract_embeddings

        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(            # Normalize using ImageNet stats
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        if self.selected_concept_groups != None and self.name == 'cub_incomplete':
            # Select the group that matches the selected group names
            self.incomplete_cub_groups = {k:v for k,v in cub_concept_groups.items() if k in selected_concept_groups}
            # Select the name of the concepts corresponding to the ones in the selected groups
            self.selected_concept_idxes = list(itertools.chain.from_iterable([x for x in self.incomplete_cub_groups.values()]))
            self.CUB_CONCEPT_NAMES = [x for i, x in enumerate(cub_concept_semantics) if i in self.selected_concept_idxes]
            # reset the indexes in the cub groups
            cnt = 0
            for k,v in self.incomplete_cub_groups.items():
                self.incomplete_cub_groups[k] = [x+cnt for x in list(range(len(self.incomplete_cub_groups[k])))]
                cnt += len(self.incomplete_cub_groups[k])

        if self.selected_concepts is not None and self.name == 'awa2_incomplete':
            # Select the indexes that matches the selected concept names
            self.selected_concept_idxes = [awa2_concept_semantics.index(x) for x in self.selected_concepts]
            # Filter the group according to the selected concepts
            self.incomplete_awa2_groups = {}
            for k, v in awa2_concept_groups.items():
                idxs = []
                for idx in self.selected_concept_idxes:
                    if idx in v:
                        idxs.append(idx)
                if len(idxs) > 0:
                    self.incomplete_awa2_groups[k] = idxs

            # reset the indexes in the cub groups
            cnt = 0
            for k,v in self.incomplete_awa2_groups.items():
                self.incomplete_awa2_groups[k] = [x+cnt for x in list(range(len(self.incomplete_awa2_groups[k])))]
                cnt += len(self.incomplete_awa2_groups[k])

    def get_names(self):
        # Get the concept names and task names
        if self.name in ['xor', 'trigonometry', 'dot', 'checkmark']:
            dataset = ToyDataset(self.name, size=1000, random_state=42)
            concept_names = dataset.concept_attr_names
            task_names = [self.name] #['0', '1']
            concept_groups = None
        elif self.name in ['or', 'xnor', 'nor']:
            dataset = ToyDataset('xor', size=1000, random_state=42)
            concept_names = dataset.concept_attr_names
            task_names = [self.name]
            concept_groups = None
        elif self.name in ['mnist_addition']:
            train_dataset = MNISTAddition(root=DATA_PATH, train=True)
            concept_names = train_dataset.concept_names
            task_names = train_dataset.task_names
            concept_groups = None
        elif self.name == 'cub':
            concept_names = CUB_CONCEPT_NAMES
            task_names = cub_class_names
            concept_groups = cub_concept_groups
        elif self.name == 'celeba':
            test_dataset = CelebADataset(root=DATA_PATH, split='test', 
                                         class_attributes=self.task_names,
                                         transform=self.transform)
            concept_names = test_dataset.concept_attr_names
            # delete test_dataset
            del test_dataset
            task_names = ["class_"+str(x) for x in range(2**len(self.task_names))]
            concept_groups = None
        elif self.name == 'awa2':
            concept_names = awa2_concept_semantics
            task_names = awa2_class_names
            concept_groups = awa2_concept_groups
        elif self.name == 'awa2_incomplete':
            concept_names = self.selected_concepts
            task_names = awa2_class_names
            concept_groups = self.incomplete_awa2_groups
        elif self.name == 'cub_incomplete':
            concept_names = self.CUB_CONCEPT_NAMES
            task_names = cub_class_names
            concept_groups = self.incomplete_cub_groups
        else:
            raise ValueError(f"Dataset {self.name} not recognized.")
        
        return concept_names, task_names, concept_groups

    def load_data(self):
        # Load the data
        if self.name in ['xor', 'trigonometry', 'dot', 'checkmark']:
            dataset = ToyDataset(self.name, size=1000, random_state=42)
            # split the dataset
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [0.7, 0.1, 0.2]
            )
        elif self.name in ['or', 'nor', 'xnor']:
            dataset = ToyDataset('xor', size=1000, random_state=42)
            # split the dataset
            if self.name == 'xnor':
                dataset.target_labels = 1 - dataset.target_labels
                assert torch.isclose(dataset.target_labels.mean(), torch.tensor(0.5), atol=0.1), \
                    "XNOR dataset not generated correctly"
                dataset.name = 'xnor'
                dataset.task_attr_names = 'xnor'
            elif self.name == 'nor':
                dataset.target_labels = ((dataset.data[:, 0] < 0.5).float() *
                                         (dataset.data[:, 1] < 0.5).float())
                assert torch.isclose(dataset.target_labels.mean(), torch.tensor(0.25), atol=0.1), \
                    "NOR dataset not generated correctly"
                dataset.name = 'nor'
                dataset.task_attr_names = 'nor'
            else:
                dataset.target_labels = torch.clip(((dataset.data[:, 0] > 0.5).float() +
                                         (dataset.data[:, 1] > 0.5).float()), 0, 1)
                assert torch.isclose(dataset.target_labels.mean(), torch.tensor(0.75), atol=0.1), \
                    "OR dataset not generated correctly"
                dataset.name = 'or'
                dataset.task_attr_names = 'or'
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [0.7, 0.1, 0.2]
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
        elif self.name == 'cub_incomplete':
            train_dataset = CUBDataset(root=DATA_PATH, split='train', selected_concepts=self.selected_concept_idxes)
            val_dataset = CUBDataset(root=DATA_PATH, split='val', selected_concepts=self.selected_concept_idxes)
            test_dataset = CUBDataset(root=DATA_PATH, split='test', selected_concepts=self.selected_concept_idxes)
        elif self.name == 'celeba':
            train_dataset = CelebADataset(root=DATA_PATH, split='train', 
                                          class_attributes=self.task_names,
                                          transform=self.transform,
                                          download=True)
            test_dataset = CelebADataset(root=DATA_PATH, split='test', 
                                         class_attributes=self.task_names,
                                         transform=self.transform,
                                         download=True)
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, 
                                              [train_size, val_size])
        elif self.name == 'awa2':
            path = os.path.join(DATA_PATH, 'Animals_with_Attributes2')
            train_dataset = AwA2Dataset(root=path, split='train')
            val_dataset = AwA2Dataset(root=path, split='val')
            test_dataset = AwA2Dataset(root=path, split='test')
        elif self.name == 'awa2_incomplete':
            path = os.path.join(DATA_PATH, 'Animals_with_Attributes2')
            train_dataset = AwA2Dataset(root=path, split='train', selected_concepts=self.selected_concept_idxes)
            val_dataset = AwA2Dataset(root=path, split='val', selected_concepts=self.selected_concept_idxes)
            test_dataset = AwA2Dataset(root=path, split='test', selected_concepts=self.selected_concept_idxes)
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
        
        if self.name in ['cub', 'mnist_addition', 'celeba', 'awa2',
                         'cub_incomplete', 'awa2_incomplete'] and \
            self.extract_embeddings:
            celeba_flag = True if self.name == 'celeba' else False
            E_extr = EmbeddingExtractor(loaded_train, 
                                        loaded_val, 
                                        loaded_test, 
                                        self.device,
                                        celeba_flag,
                                        self.task_names)
            loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()

        return loaded_train, loaded_val, loaded_test


if __name__ == '__main__':
    print("\nXNOR dataset")
    xnor_dataset = loader(
        name='xnor',
        batch_size=32,
        num_workers=4,
        device='cpu',
    ).load_data()[0]
    print(xnor_dataset.__iter__().__next__())

    print("\nNOR dataset")
    nor_dataset = loader(
        name='nor',
        batch_size=32,
        num_workers=4,
        device='cpu',
    ).load_data()[0]
    print(nor_dataset.__iter__().__next__())

    print("\nOR dataset")
    or_dataset = loader(
        name='or',
        batch_size=32,
        num_workers=4,
        device='cpu',
    ).load_data()[0]
    print(or_dataset.__iter__().__next__())
