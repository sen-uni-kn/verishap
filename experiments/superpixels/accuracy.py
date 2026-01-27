# Copyright 2025 David Boetius
"""Compute training and test set accuracy for CNNs on CIFAR10/GTSRB."""

import argparse
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from ..models import CNN


def get_datasets(dataset_name: str) -> tuple:
    """Load training and test datasets."""
    if dataset_name.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            ".datasets",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        testset = torchvision.datasets.CIFAR10(
            ".datasets",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        shape = (3, 32, 32)
    elif dataset_name.lower() == "gtsrb":
        trainset = torchvision.datasets.GTSRB(
            ".datasets",
            split="train",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        testset = torchvision.datasets.GTSRB(
            ".datasets",
            split="test",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
        shape = (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return trainset, testset, shape


def compute_accuracy(model, state, dataloader, batch_size: int = 256) -> float:
    """Compute accuracy on a dataset."""
    model = eqx.nn.inference_mode(model)

    @partial(jax.vmap, axis_name="batch")
    def forward(x):
        y, _ = model(x, state)
        return y

    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.numpy()
        labels = labels.numpy()

        logits = forward(jnp.asarray(images))
        predictions = jnp.argmax(logits, axis=-1)
        correct += (predictions == labels).sum().item()
        total += len(labels)

    return correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Compute CNN accuracy on CIFAR10/GTSRB."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the model file (.eqx).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (cifar10 or gtsrb). If not specified, inferred from model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args()

    # Infer dataset from model name if not specified
    dataset_name = args.dataset
    if dataset_name is None:
        model_stem = args.model.stem.lower()
        if "cifar10" in model_stem:
            dataset_name = "cifar10"
        elif "gtsrb" in model_stem:
            dataset_name = "gtsrb"
        else:
            raise ValueError(
                "Could not infer dataset from model name. Please specify --dataset."
            )

    print(f"Loading model from {args.model}")
    model, state = CNN.load(args.model)

    print(f"Loading {dataset_name} dataset")
    trainset, testset, shape = get_datasets(dataset_name)

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        multiprocessing_context="forkserver",
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        multiprocessing_context="forkserver",
    )

    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")

    print("\nComputing training accuracy...")
    train_acc = compute_accuracy(model, state, train_loader, args.batch_size)
    print(f"Training accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")

    print("\nComputing test accuracy...")
    test_acc = compute_accuracy(model, state, test_loader, args.batch_size)
    print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    print(f"\nSummary:")
    print(f"  Train: {train_acc * 100:.2f}%")
    print(f"  Test:  {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
