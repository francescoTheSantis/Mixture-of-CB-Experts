"""
Script adapted from: https://github.com/mvandenhi/SCBM/blob/main/datasets/cifar10_dataset.py
"""

import torch
from torchvision import datasets, transforms
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from env import HOME, DATA_PATH


def get_CIFAR10_CBM_dataloader(datapath, selected_idxs=None):
    datapath = datapath + "cifar10/"
    image_datasets = {
        "train": CIFAR10_CBM_dataloader(
            root=datapath,
            train=True,
            download=False,
            selected_idxs=selected_idxs
        ),
        "test": CIFAR10_CBM_dataloader(
            root=datapath,
            train=False,
            download=False,
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