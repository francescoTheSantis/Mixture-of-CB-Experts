import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from env import HOME, DATA_PATH

class MNISTExponential(Dataset):
    """
    MNIST dataset where the target is exp(digit).
    """
    def __init__(self, root=None, train=True, transform=None, download=True):
        if root is None:
            root = DATA_PATH
        if transform is None:
            transform = transforms.ToTensor()
        self.dataset = datasets.MNIST(root=root, train=train, transform=transform, download=download)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, digit = self.dataset[idx]
        target = np.exp(float(digit))
        return image, torch.tensor(digit, dtype=torch.long), torch.tensor(target, dtype=torch.float32)

def plot_mnist_exponential_samples(root, dataset, num_samples=10, figsize=(15, 3)):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        image, digit, target = dataset[i]
        img_array = image.squeeze(0).numpy()
        axes[i].imshow(img_array, cmap='gray')
        axes[i].set_title(f'Digit: {digit.item()}\nTarget: {target.item():.3f}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"{root}/mnist_exponential_samples.pdf")
    plt.show()

if __name__ == "__main__":
    root = f"{HOME}/figs/mnist_exponential"
    os.makedirs(root, exist_ok=True)

    dataset = MNISTExponential()
    plot_mnist_exponential_samples(root, dataset, num_samples=10)

    for i in range(20):
        image, digit, target = dataset[i]
        print(f"Sample {i}: Digit: {digit.item()}, Target: {target.item():.3f}")
