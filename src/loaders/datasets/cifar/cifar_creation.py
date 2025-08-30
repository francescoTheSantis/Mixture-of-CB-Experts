"""
Script adapted from: 
- https://github.com/mvandenhi/SCBM/blob/main/datasets/create_dataset_cifar.py

The files required to compute the concept labels have been downloaded form the Label-free-CBM repository: 
- https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main

Instructions to correctly create the CIFAR10 and CIFAR100 datasets:
    1. create a folder for each dataset.
    2. download the concept lists from the second link shown above and save them in the appropriate dataset folder.
    3. use the provided scripts to compute the concept labels for each dataset.
"""

import torchvision
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
from env import HOME, DATA_PATH


def main(cifar):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with open(f"{DATA_PATH}{cifar}/{cifar}_filtered.txt", "r") as file:
        # Read the contents of the file
        concept_list = [line.strip() for line in file]
    # Adding negated concepts
    neg_concept_list = ["not " + line for line in concept_list]
    pos_and_neg_concepts = concept_list + neg_concept_list


    ### GETTING TEXT EMBEDDINGS
    class CustomTransform:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, image):
            # Convert the image to a tensor
            # Use the processor to prepare the inputs
            inputs = self.processor(
                text=pos_and_neg_concepts, images=image, return_tensors="pt", padding=True
            )
            return inputs


    transform = CustomTransform(processor)


    # Load imagenet from folder
    if cifar == "cifar10":
        cifar_data = torchvision.datasets.CIFAR10(
            root=f"{DATA_PATH}/{cifar}", train=True, transform=transform, download=True
        )
    else:
        cifar_data = torchvision.datasets.CIFAR100(
            root=f"{DATA_PATH}/{cifar}", train=True, transform=transform, download=True
        )
    data_loader = torch.utils.data.DataLoader(
        cifar_data,
        batch_size=128,
        shuffle=False,
        num_workers=15,
        pin_memory=False,
    )
    model.to("cuda")

    ### Getting the text embeddings once s.t. they don't need to be recomputed every time.
    with torch.no_grad():
        for i, (inputs, target) in enumerate(data_loader):
            inputs.to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
            inputs["input_ids"] = inputs["input_ids"][0]
            inputs["attention_mask"] = inputs["attention_mask"][0]
            output = model(**inputs)
            text_embed = output.text_embeds.cpu()
            break


    ### GETTING IMAGE EMBEDDINGS
    class CustomTransform2:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, image):
            # Convert the image to a tensor
            # Use the processor to prepare the inputs
            inputs = self.processor(
                text=[""], images=image, return_tensors="pt", padding=True
            )
            return inputs


    transform = CustomTransform2(processor)

    for split in [False, True]:

        if cifar == "cifar10":
            cifar_data = torchvision.datasets.CIFAR10(
                root=f"{DATA_PATH}/{cifar}", train=split, transform=transform, download=True
            )
        else:
            cifar_data = torchvision.datasets.CIFAR100(
                root=f"{DATA_PATH}/{cifar}", train=split, transform=transform, download=True
            )
        data_loader = torch.utils.data.DataLoader(
            cifar_data,
            batch_size=128,
            shuffle=False,
            num_workers=15,
            pin_memory=True,
        )
        data_storage = []
        with torch.no_grad():
            for i, (inputs, target) in enumerate(tqdm(data_loader, desc=f"Processing {cifar} set")):
                inputs.to("cuda")
                inputs["pixel_values"] = inputs["pixel_values"].squeeze(1)
                inputs["input_ids"] = inputs["input_ids"][0]
                inputs["attention_mask"] = inputs["attention_mask"][0]
                output = model(**inputs)
                similarity = torch.nn.functional.cosine_similarity(
                    text_embed.unsqueeze(0).expand(output.image_embeds.shape[0], -1, -1),
                    output.image_embeds.unsqueeze(1)
                    .expand(-1, text_embed.shape[0], -1)
                    .cpu(),
                    dim=-1,
                    eps=1e-8,
                )
                # Split the similarity tensor into positive and negative concepts
                similarity_pos_neg = similarity.reshape(target.shape[0], 2, -1)
                concept_label = 1 - similarity_pos_neg.argmax(
                    dim=1
                )  # 0 if negated, 1 if positive
                data_storage.append(concept_label > 0)
        data_storage = torch.cat(data_storage)
        if split:
            name = "train"
        else:
            name = "test"
        torch.save(data_storage, f"{DATA_PATH}/{cifar}/{cifar}_{name}_concept_labels.pt")


if __name__ == "__main__":
    # create and save both datasets
    for cifar in ["cifar10", "cifar100"]:
        main(cifar)