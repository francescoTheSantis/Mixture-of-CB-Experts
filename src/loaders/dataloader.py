from torch_concepts.data import ToyDataset
import torch

class loader(object):
    def __init__(self, 
                 name: str,
                 batch_size: int = 32,
                 ):
        self.name = name
        self.batch_size = batch_size

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
        else:
            pass

        # create the dataloaders
        loaded_train = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        loaded_val = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        loaded_test = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)    

        return loaded_train, loaded_val, loaded_test, concept_names, task_names