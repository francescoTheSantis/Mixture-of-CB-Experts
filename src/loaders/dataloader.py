from torch_concepts.data import ToyDataset
import torch
from torch import nn
from torch_concepts.data.mnist import MNISTAddition
from env import DATASETS
from torch.utils.data import DataLoader, random_split

class loader(object):
    def __init__(self, 
                 name,
                 batch_size,
                 num_workers
                 ):
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load_data(self):
        # Load the data
        if self.name in ['xor', 'trigonometry', 'dot', 'checkmark']:
            dataset = ToyDataset(self.name, size=1000, random_state=42)
            concept_names = dataset.concept_attr_names
            task_names = dataset.task_attr_names
            # split the dataset
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [0.7, 0.1, 0.2],
                generator=torch.Generator().manual_seed(42)
            )
  
        elif self.name in ['mnist_addition']:
            train_dataset = MNISTAddition(root=DATASETS, train=True)
            concept_names = train_dataset.concept_names
            task_names = train_dataset.task_names
            test_dataset = MNISTAddition(root=DATASETS, train=False)

            # Split the dataset into train, validation and test sets
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, 
                                              [train_size, val_size])

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

        return loaded_train, loaded_val, loaded_test, concept_names, task_names
