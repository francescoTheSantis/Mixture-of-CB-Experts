from torch_concepts.data import ToyDataset
import torch
from torch_concepts.data.mnist import MNISTAddition
from src.loaders.datasets.cub import CUBDataset
from src.loaders.datasets.cub import SELECTED_CONCEPTS as cub_selected_concepts
from src.loaders.datasets.cub import CONCEPT_SEMANTICS as cub_concept_semantics
from src.loaders.datasets.cub import CLASS_NAMES as cub_class_names
from src.loaders.datasets.cub import CONCEPT_GROUP_MAP as cub_concept_groups
from src.loaders.datasets.awa2 import AwA2Dataset
from src.loaders.datasets.awa2 import CONCEPT_SEMANTICS as awa2_concept_semantics
from src.loaders.datasets.awa2 import CLASS_NAMES as awa2_class_names
from src.loaders.datasets.awa2 import CONCEPT_GROUPS as awa2_concept_groups
from torch_concepts.data.celeba import CelebADataset
from src.loaders.datasets.cebab import CEBaBDataset
from src.loaders.datasets.cifar.cifar10 import get_CIFAR10_CBM_dataloader
from src.loaders.datasets.cifar.cifar100 import get_CIFAR100_CBM_dataloader
from src.loaders.datasets.mnist_arithmetic import ArithmeticMNISTDataset
from src.loaders.datasets.mnist_arithmetic import CONCEPT_NAMES as mnist_arithmetic_concept_names
from src.loaders.datasets.pendulum import PendulumDataset
from src.loaders.datasets.pendulum import CONCEPT_NAMES as concept_names_pendulum
from src.loaders.datasets.pendulum import TASK_NAMES as task_names_pendulum
from src.loaders.datasets.dsprites import DSprites

from torch.utils.data import DataLoader, random_split
from env import DATA_PATH
from src.loaders.preprocessing import EmbeddingExtractor, \
    TextEmbeddingExtractor
import omegaconf
from torchvision import transforms
import os
import itertools
from src.utilities import get_type_from_name
import random
import pandas as pd

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text):
        self.encoded_text = encoded_text

    def __getitem__(self, idx):
        t = {key: torch.tensor(values[idx]) for key, values in
             self.encoded_text.items()}
        return t

    def __len__(self):
        return len(self.encoded_text['input_ids'])

CUB_CONCEPT_NAMES = [x for i, x in enumerate(cub_concept_semantics) if i in cub_selected_concepts]

class loader(object):
    """
    Data loader class to manage loading, preprocessing, and batching of various datasets from PyC.
    """
    def __init__(self,
                 name,
                 batch_size,
                 num_workers,
                 device,
                 selected_concepts=None,
                 selected_concept_groups=None,
                 concept_percentage=None,
                 class_attributes=None,
                 extract_embeddings=True,
                 data_path=None,
                 dataset_already_created=False,
                 formulas=None,
                 seed=42,
                 n_samples=None,
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
        self.concept_percentage = concept_percentage
        self.data_path = data_path
        self.dataset_already_created = dataset_already_created
        self.formulas = formulas
        self.seed = seed
        self.n_samples = n_samples

        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(            # Normalize using ImageNet stats
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        if self.name == 'cub' and self.concept_percentage is not None:
            # Randomly select the concepts according to the concept_percentage %
            selected_concepts = random.sample(CUB_CONCEPT_NAMES, int(len(CUB_CONCEPT_NAMES) * self.concept_percentage))
            # Select the indexes that matches the selected concept names
            self.selected_concept_idxes = [CUB_CONCEPT_NAMES.index(x) for x in selected_concepts]
            # Filter the group according to the selected concepts
            self.incomplete_cub_groups = {}
            for k, v in cub_concept_groups.items():
                idxs = []
                for idx in self.selected_concept_idxes:
                    if idx in v:
                        idxs.append(idx)
                if len(idxs) > 0:
                    self.incomplete_cub_groups[k] = idxs

            # Store in the same format as cub_incomplete
            self.CUB_CONCEPT_NAMES = selected_concepts

        # set the concepts for the incomplete version of cifar10/100 datasets
        if self.name in ['cifar10', 'cifar100']:

            # Read and set the concept and label names for cifar10/100
            with open(f"{DATA_PATH}{self.name}/{self.name}_filtered.txt", "r") as file:
                concept_list = [line.strip() for line in file]
            self.concept_names_cifar = concept_list
            with open(f"{DATA_PATH}{self.name}/{self.name}_classes.txt", "r") as file:
                task_label_list = [line.strip() for line in file]            
            self.task_names_cifar = task_label_list

            # Generate concept indexes 
            self.concept_idxs_cifar = {self.concept_names_cifar.index(x): x for x in self.concept_names_cifar}

            # Reduce the concept size if the percentage is not none
            if self.concept_percentage is not None:
                # generate random number from 0 to the number of concepts in the dataset
                filtered_idxs = random.sample(range(0, len(self.concept_names_cifar)), int(len(self.concept_names_cifar) * self.concept_percentage))
                # sort the indexes in ascending order
                filtered_idxs.sort()
                # Select the concepts
                self.concept_names_cifar = [self.concept_idxs_cifar[x] for x in filtered_idxs]
                # Update the idxs
                self.concept_idxs_cifar = {idx:name for idx, name in zip(filtered_idxs, self.concept_names_cifar)}

            self.concept_idxs_cifar = list(sorted(self.concept_idxs_cifar.keys()))

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
            dataset = ToyDataset(self.name, size=1000)
            concept_names = dataset.concept_attr_names
            task_names = [self.name] #['0', '1']
            concept_groups = None
        elif self.name in ['or', 'xnor', 'nor']:
            dataset = ToyDataset('xor', size=1000)
            concept_names = dataset.concept_attr_names
            task_names = [self.name]
            concept_groups = None
        elif self.name in ['mnist_addition']:
            train_dataset = MNISTAddition(root=DATA_PATH, train=True)
            concept_names = train_dataset.concept_names
            task_names = train_dataset.task_names
            concept_groups = None
        elif self.name in ['mnist_arithmetic', 'mnist_arithmetic_hard']:
            concept_names = mnist_arithmetic_concept_names
            task_names = ['Result']
            concept_groups = None
        elif self.name == 'cub' and self.concept_percentage is None:
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
        elif self.name == 'cub_incomplete' or (self.name == 'cub' and self.concept_percentage is not None):
            concept_names = self.CUB_CONCEPT_NAMES
            task_names = cub_class_names
            concept_groups = self.incomplete_cub_groups
        elif self.name == "sst2":
            from transformers import AutoTokenizer
            # For SST2, we use the concept names as the task names
            concept_names = ['negative', 'positive']
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1") # TODO: switch to Llama
            # get the list of all tokens
            task_names = [f"w_{i}" for i in range(len(tokenizer.get_vocab().keys()))] # simple keys were not working
            concept_groups = None
        elif self.name == "cebab":
            concept_names = ['food_negative', 'food_unknown', 'food_positive', 'ambiance_negative', 'ambiance_unknown', 'ambiance_positive', 'service_negative', 'service_unknown', 'service_positive', 'noise_negative', 'noise_unknown', 'noise_positive']
            task_names = ['review']
            concept_groups = {'food': [0,1,2], 'ambiance': [3,4,5], 'service': [6,7,8], 'noise': [9,10,11]}
        elif self.name in ['cifar10', 'cifar100']:
            concept_names = self.concept_names_cifar          
            task_names = self.task_names_cifar
            concept_groups = None
        elif self.name == "pendulum":
            concept_names = concept_names_pendulum
            task_names = task_names_pendulum
            concept_groups = None
        elif self.name in ["dsprites_simple", "dsprites_complex"]:
            concept_names = self.selected_concepts
            task_names = ['custom_target']
            concept_groups = None
        else:
            raise ValueError(f"Dataset {self.name} not recognized.")
        
        return concept_names, task_names, concept_groups

    def load_data(self, cfg=None):
        # Load the data
        if self.name in ['xor', 'trigonometry', 'dot', 'checkmark']:
            dataset = ToyDataset(self.name, size=1000, random_state=self.seed)
            # split the dataset
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [0.7, 0.1, 0.2]
            )
        elif self.name in ['or', 'nor', 'xnor']:
            dataset = ToyDataset('xor', size=1000, random_state=self.seed)
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
        elif self.name == 'mnist_arithmetic':
            train_dataset = ArithmeticMNISTDataset(mnist_root=DATA_PATH, train=True, num_samples=int(self.n_samples*0.7), img_size=224)
            val_dataset = ArithmeticMNISTDataset(mnist_root=DATA_PATH, train=True, num_samples=int(self.n_samples*0.1), img_size=224)
            test_dataset = ArithmeticMNISTDataset(mnist_root=DATA_PATH, train=False, num_samples=int(self.n_samples*0.2), img_size=224)
            # Get the equations x sample in the test-set
            operators = test_dataset.operator_list
            equations = []
            for i in range(len(operators)):
                equations.append(f"c0 {operators[i]} c1")
            # Create a dataframe and save it
            equations_df = pd.DataFrame(equations, columns=['equation'])
            equations_df.to_csv(f"{self.data_path}/mnist_arithmetic_equations.csv", index=False)
        elif self.name == 'mnist_arithmetic_hard':
            train_dataset = ArithmeticMNISTDataset(mnist_root=DATA_PATH, train=True, num_samples=int(self.n_samples*0.7), img_size=224, operators=('x', '/'))
            val_dataset = ArithmeticMNISTDataset(mnist_root=DATA_PATH, train=True, num_samples=int(self.n_samples*0.1), img_size=224, operators=('x', '/'))
            test_dataset = ArithmeticMNISTDataset(mnist_root=DATA_PATH, train=False, num_samples=int(self.n_samples*0.2), img_size=224, operators=('x', '/'))
            # Get the equations x sample in the test-set
            operators = test_dataset.operator_list
            equations = []
            for i in range(len(operators)):
                equations.append(f"c0 {operators[i]} c1")
            # Create a dataframe and save it
            equations_df = pd.DataFrame(equations, columns=['equation'])
            equations_df.to_csv(f"{self.data_path}/mnist_arithmetic_hard_equations.csv", index=False)            
        elif self.name == 'cub' and self.concept_percentage is None:
            train_dataset = CUBDataset(root=DATA_PATH, split='train')
            val_dataset = CUBDataset(root=DATA_PATH, split='val')
            test_dataset = CUBDataset(root=DATA_PATH, split='test')
        elif self.name == 'cub_incomplete' or (self.name == 'cub' and self.concept_percentage is not None):
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
        elif self.name == 'sst2':
            from datasets import load_dataset
            from transformers import AutoTokenizer

            train_dataset = load_dataset('SetFit/sst2', split='train')
            val_dataset = load_dataset('SetFit/sst2', split='validation')
            test_dataset = load_dataset('SetFit/sst2', split='test')
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            tokenizer.pad_token = tokenizer.eos_token
            encoded_datasets = []
            for dataset in [train_dataset, val_dataset, test_dataset]:
                encoded_dataset = dataset.map(
                    lambda e: tokenizer(e['text'], 
                                        padding=True,
                                        truncation=True,
                                        max_length=350), 
                                        batched=True,
                                        batch_size=len(dataset)
                )
                encoded_dataset = encoded_dataset.remove_columns(['text'])
                encoded_dataset = encoded_dataset.remove_columns(['label_text'])
                encoded_dataset = encoded_dataset[:len(encoded_dataset)]
                encoded_datasets.append(encoded_dataset)

            encoded_train_dataset, encoded_val_dataset, encoded_test_dataset = encoded_datasets

            train_dataset = TextDataset(encoded_train_dataset)
            val_dataset = TextDataset(encoded_val_dataset)
            test_dataset = TextDataset(encoded_test_dataset)
        elif self.name == "cebab":
            loader = CEBaBDataset(cfg.text_backbone_name, self.batch_size)
            loaded_train, loaded_val, loaded_test = loader.collator()
        elif self.name == 'cifar10':
            train_dataset, test_dataset = get_CIFAR10_CBM_dataloader(DATA_PATH, self.concept_idxs_cifar)
            # split the train in training and validation
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        elif self.name == 'cifar100':
            train_dataset, test_dataset = get_CIFAR100_CBM_dataloader(DATA_PATH, self.concept_idxs_cifar)
            # split the train in training and validation
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        elif self.name == "pendulum":
            loader = PendulumDataset(already_created=self.dataset_already_created)
            loaded_train, loaded_val, loaded_test = loader.collator()
        elif self.name in ["dsprites_simple", "dsprites_complex"]:
            assert self.formulas is not None, "Formulas must be provided for dsprites dataset"
            dsprites_dataset = DSprites(concepts=self.selected_concepts,
                                        formulas=self.formulas,
                                        split='train',
                                        num_samples=self.n_samples,
                                        random_seed=self.seed)
            # split the dataset into train, validation and test sets
            total_size = len(dsprites_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.1 * total_size)
            test_size = total_size - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(dsprites_dataset, [train_size, val_size, test_size])
        else:
            raise ValueError(f"Dataset {self.name} not recognized.")

        if get_type_from_name(self.name) != 'text':
            loaded_train = DataLoader(train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    persistent_workers=True if self.num_workers > 0 else False,
                                    pin_memory=True,
                                    collate_fn=self._custom_collate_fn)
            loaded_val = DataLoader(val_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    persistent_workers=True if self.num_workers > 0 else False,
                                    pin_memory=True,
                                    collate_fn=self._custom_collate_fn)
            loaded_test = DataLoader(test_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    persistent_workers=False if self.num_workers > 0 else False,
                                    collate_fn=self._custom_collate_fn)
        
        # We always modify the dataloaders since we want them to align with the batch that the engine is expecting.
        if get_type_from_name(self.name) == 'image' and self.extract_embeddings:
            celeba_flag = True if self.name == 'celeba' else False
            E_extr = EmbeddingExtractor(
                cfg,
                loaded_train, 
                loaded_val, 
                loaded_test, 
                self.device,
                celeba_flag,
                self.task_names,
                self.extract_embeddings
            )
            loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()
        elif get_type_from_name(self.name) == 'text':
            E_extr = TextEmbeddingExtractor(
                cfg,
                loaded_train,
                loaded_val,
                loaded_test,
                self.device,
                self.extract_embeddings
            )
            loaded_train, loaded_val, loaded_test = E_extr.produce_loaders()

        return loaded_train, loaded_val, loaded_test

    def _custom_collate_fn(self, batch):
        """
        Custom collate function to handle different data types in the batch.
        """
        batch_dict = {}
        for idx, key in enumerate(['x', 'c', 'y']):
            values = [item[idx] for item in batch]
            if isinstance(values[0], torch.Tensor):
                batch_dict[key] = torch.stack(values)
            else:
                batch_dict[key] = torch.tensor(values)
        return batch_dict

if __name__ == '__main__':
    print("\nSST2 dataset")
    sst2_datasets = loader(
        name='sst2',
        batch_size=10,
        num_workers=0,
        device='mps',
        extract_embeddings=False
    ).load_data()

    print([val.shape for val in sst2_datasets[0].__iter__().__next__().values()])

    E_extr = TextEmbeddingExtractor(sst2_datasets[0],
                                    sst2_datasets[1],
                                    sst2_datasets[2],
                                    device='mps',
                                    )
    train_loader, val_loader, test_loader = E_extr.produce_loaders()

    # we save the train, validation and test loaders
    data_path = os.path.join(DATA_PATH, 'stored_tensors', 'sst2')
    os.makedirs(data_path, exist_ok=True)
    torch.save(train_loader, os.path.join(data_path, 'train.pt'))
    torch.save(val_loader, os.path.join(data_path, 'val.pt'))
    torch.save(test_loader, os.path.join(data_path, 'test.pt'))


