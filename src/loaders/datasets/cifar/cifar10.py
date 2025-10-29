"""
Script adapted from: https://github.com/mvandenhi/SCBM/blob/main/datasets/cifar10_dataset.py
"""

import torch
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from env import HOME, DATA_PATH


def _check_cifar_data_exists(datapath, cifar_name="cifar10"):
    """Check if CIFAR concept labels exist, create them if not."""
    train_concepts_path = os.path.join(datapath, f"{cifar_name}_train_concept_labels.pt")
    test_concepts_path = os.path.join(datapath, f"{cifar_name}_test_concept_labels.pt")
    filtered_txt_path = os.path.join(datapath, f"{cifar_name}_filtered.txt")
    
    # Check if concept labels exist
    if os.path.exists(train_concepts_path) and os.path.exists(test_concepts_path):
        return True
    
    # Check if required text files exist
    if not os.path.exists(filtered_txt_path):
        raise FileNotFoundError(
            f"Required file '{cifar_name}_filtered.txt' not found in {datapath}. "
            f"Please download it from https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main "
            f"and place it in the {datapath} directory."
        )
    
    classes_txt_path = os.path.join(datapath, f"{cifar_name}_classes.txt")
    if not os.path.exists(classes_txt_path):
        raise FileNotFoundError(
            f"Required file '{cifar_name}_classes.txt' not found in {datapath}. "
            f"Please download it from https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main "
            f"and place it in the {datapath} directory."
        )
    
    return False


def _create_cifar_concepts(cifar_name="cifar10"):
    """Create CIFAR concept labels by running the creation script."""
    print(f"\n{'='*60}")
    print(f"{cifar_name.upper()} concept labels not found!")
    print(f"Generating concept labels automatically...")
    print(f"This may take several minutes depending on your GPU.")
    print(f"{'='*60}\n")
    
    # Import and run the creation function
    from .cifar_creation import main as create_cifar_main
    
    try:
        create_cifar_main(cifar_name)
        print(f"\n{'='*60}")
        print(f"{cifar_name.upper()} concept labels created successfully!")
        print(f"{'='*60}\n")
    except Exception as e:
        raise RuntimeError(
            f"Failed to create {cifar_name.upper()} concept labels. "
            f"Error: {str(e)}\n"
            f"You may need to run the creation script manually: "
            f"python src/loaders/datasets/cifar/cifar_creation.py"
        )


def get_CIFAR10_CBM_dataloader(datapath, selected_idxs=None):
    datapath = datapath + "cifar10/"
    
    # Check if concept labels exist, create if not
    if not _check_cifar_data_exists(datapath, "cifar10"):
        _create_cifar_concepts("cifar10")
    
    image_datasets = {
        "train": CIFAR10_CBM_dataloader(
            root=datapath,
            train=True,
            download=True,
            selected_idxs=selected_idxs
        ),
        "test": CIFAR10_CBM_dataloader(
            root=datapath,
            train=False,
            download=True,
            selected_idxs=selected_idxs
        ),
    }

    return image_datasets["train"], image_datasets["test"]


class CIFAR10_CBM_dataloader(datasets.CIFAR10):

    def __init__(self, selected_idxs, *args, **kwargs):
        super(CIFAR10_CBM_dataloader, self).__init__(*args, **kwargs)

        self.selected_idxs = selected_idxs
        
        if kwargs["train"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # implicitly divides by 255
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.concepts = (
                torch.load(kwargs["root"] + f"cifar10_train_concept_labels.pt") * 1
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # implicitly divides by 255
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            self.concepts = (
                torch.load(kwargs["root"] + f"cifar10_test_concept_labels.pt") * 1
            )

    def __getitem__(self, idx):
        X, target = super().__getitem__(idx)

        # from select.concept[idx], select only the columns identified by selected_idxs
        if self.selected_idxs is not None:
            concepts = self.concepts[idx, self.selected_idxs]

        return (X, concepts, torch.tensor(target))


if __name__ == "__main__":
    train_loader, test_loader = get_CIFAR10_CBM_dataloader(DATA_PATH)
    sample = next(iter(train_loader))
    for el in sample:
        print(el.shape)