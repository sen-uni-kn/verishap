#!/usr/bin/env python3
# Copyright 2025 David Boetius
"""View images from datasets with keyboard navigation."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from .argument_parser import NumpyVisionDataset, NumpyVisionDataset2
from .datasets import NIHChestXrayDataset


class DatasetViewer:
    def __init__(self, dataset_name: str, train_set: bool):
        self.dataset_name = dataset_name
        self.raw_dataset, self.labels, self.label_names = self._load_dataset(train_set)
        self.current_index = 0
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _load_dataset(self, train_set: bool):
        """Load the dataset based on the dataset name."""
        dataset = self.dataset_name

        if dataset.lower() == "mnist":
            testset = torchvision.datasets.MNIST(
                ".datasets",
                train=train_set,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
            labels = testset.targets
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            label_names = [str(i) for i in range(10)]
            return testset, labels, label_names
        elif dataset.lower().replace("-", "").replace("_", "") == "fashionmnist":
            testset = torchvision.datasets.FashionMNIST(
                ".datasets",
                train=train_set,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
            labels = testset.targets
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            label_names = [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
            return testset, labels, label_names
        elif dataset.lower() == "chestmnist":
            path = Path(".datasets/medmnist")
            path.mkdir(parents=True, exist_ok=True)
            testset = medmnist.ChestMNIST(
                split="train" if train_set else "test",
                download=True,
                transform=torchvision.transforms.ToTensor(),
                root=path,
            )
            labels = testset.labels.squeeze()
            label_names = [
                "Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Pleural_Thickening",
                "Hernia",
            ]
            return testset, labels, label_names
        elif dataset.lower() == "octmnist":
            path = Path(".datasets/medmnist")
            path.mkdir(parents=True, exist_ok=True)
            testset = medmnist.OCTMNIST(
                split="train" if train_set else "test",
                download=True,
                transform=torchvision.transforms.ToTensor(),
                root=path,
            )
            labels = testset.labels.squeeze()
            label_names = ["CNV", "DME", "DRUSEN", "NORMAL"]
            return testset, labels, label_names
        elif dataset.lower() == "retinamnist":
            path = Path(".datasets/medmnist")
            path.mkdir(parents=True, exist_ok=True)
            testset = medmnist.RetinaMNIST(
                split="train" if train_set else "test",
                download=True,
                transform=torchvision.transforms.ToTensor(),
                root=path,
            )
            labels = testset.labels.squeeze()
            label_names = ["0", "1", "2", "3", "4"]
            return testset, labels, label_names
        elif dataset.lower() == "tissuemnist":
            path = Path(".datasets/medmnist")
            path.mkdir(parents=True, exist_ok=True)
            testset = medmnist.TissueMNIST(
                split="train" if train_set else "test",
                download=True,
                transform=torchvision.transforms.ToTensor(),
                root=path,
            )
            labels = testset.labels.squeeze()
            label_names = [
                "Collecting Duct",
                "Thick Ascending Limb",
                "Distal Convoluted Tubule",
                "Proximal Tubule",
                "Podocytes",
                "Glomerular endothelial cells",
                "Leukocytes",
                "Connecting Tubule",
            ]
            return testset, labels, label_names
        elif dataset.lower() == "cifar10":
            testset = torchvision.datasets.CIFAR10(
                ".datasets",
                train=train_set,
                download=True,
                transform=torchvision.transforms.ToTensor(),
            )
            labels = np.array(testset.targets)
            label_names = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
            return testset, labels, label_names
        elif dataset.lower() == "gtsrb":
            testset = torchvision.datasets.GTSRB(
                ".datasets",
                split="train" if train_set else "test",
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((32, 32)),
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            )
            labels = np.array([label for _, label in testset])
            label_names = [f"Sign {i}" for i in range(43)]
            return testset, labels, label_names
        elif dataset.lower().replace("-", "").replace("_", "") == "nihchestxray":
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.ToTensor(),
                ]
            )
            testset = NIHChestXrayDataset(
                split="train" if train_set else "test",
                transform=transform,
            )
            labels = np.array([label for _, label in testset])
            label_names = [
                "Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax",
                "Consolidation",
                "Edema",
                "Emphysema",
                "Fibrosis",
                "Pleural_Thickening",
                "Hernia",
                "No Finding",
            ]
            return testset, labels, label_names
        else:
            raise ValueError(
                f"Unknown dataset: {dataset}. Supported datasets: "
                "mnist, fashion-mnist, chestmnist, octmnist, retinamnist, "
                "tissuemnist, cifar10, gtsrb, nih-chest-xray"
            )

    def _display_image(self):
        """Display the current image."""
        self.ax.clear()

        # Get the image data directly from the raw dataset
        img, _ = self.raw_dataset[self.current_index]

        # Convert from tensor if needed
        if isinstance(img, torch.Tensor):
            img = img.numpy()

        # Get label
        label_idx = self.labels[self.current_index]
        if isinstance(label_idx, np.ndarray):
            label_idx = label_idx.item()
        label_name = self.label_names[label_idx]

        # Handle different image formats
        if img.shape[0] == 1:  # Grayscale with channel dimension
            img = img[0]
            cmap = "gray"
        elif img.shape[0] == 3:  # RGB with channels first
            img = np.transpose(img, (1, 2, 0))
            cmap = None
        else:
            cmap = None

        # Clip values to valid range for display
        img = np.clip(img, 0, 1)

        # Display the image
        self.ax.imshow(img, cmap=cmap)
        self.ax.set_title(
            f"{self.dataset_name} - Image {self.current_index}/{len(self.raw_dataset) - 1}\n"
            f"Label: {label_name} ({label_idx})\n"
            f"Use arrow keys (← →) to navigate, 'q' to quit"
        )
        self.ax.axis("off")
        self.fig.canvas.draw()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == "right" or event.key == "down":
            self.current_index = (self.current_index + 1) % len(self.raw_dataset)
            self._display_image()
        elif event.key == "left" or event.key == "up":
            self.current_index = (self.current_index - 1) % len(self.raw_dataset)
            self._display_image()
        elif event.key == "q" or event.key == "escape":
            plt.close(self.fig)

    def show(self):
        """Start the viewer."""
        print(f"Loading {self.dataset_name} dataset...")
        print(f"Total images: {len(self.raw_dataset)}")
        print("Controls:")
        print("  → or ↓  : Next image")
        print("  ← or ↑  : Previous image")
        print("  q or Esc: Quit")

        self._display_image()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="View images from datasets with keyboard navigation"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name (mnist, fashion-mnist, chestmnist, octmnist, "
        "retinamnist, tissuemnist, cifar10, gtsrb, nih-chest-xray)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting image index (default: 0)",
    )
    parser.add_argument(
        "--train-set",
        action="store_true",
        help="Show images from the training set. "
        "By default, shows images from the test set.",
    )

    args = parser.parse_args()

    viewer = DatasetViewer(args.dataset, args.train_set)
    viewer.current_index = args.start_index
    viewer.show()


if __name__ == "__main__":
    main()
