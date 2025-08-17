# Copyright 2025 David Boetius
# Adapted from https://docs.kidger.site/equinox/examples/mnist/
import itertools as it
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torchvision
from jaxtyping import Array, Float, Int, PyTree
from optax.losses import softmax_cross_entropy_with_integer_labels as cross_entropy
from torch.utils.data import DataLoader, Dataset

from .models import CNN

# ==============================================================================
# Hyperparameters
# ==============================================================================

BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EPOCHS = 10
PRINT_EVERY = 100
SEED = 1708
OUT_FILE = "mnist-cnn.eqxparams"

if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)
    np.random.seed(SEED + 1)

    # ==============================================================================
    # Data Loading
    # ==============================================================================

    print("=" * 80)
    print("Downloading MNIST dataset...")

    trainset = torchvision.datasets.MNIST(
        ".datasets",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    testset = torchvision.datasets.MNIST(
        ".datasets",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    @dataclass
    class MNISTDataset:
        data: np.ndarray
        targets: np.ndarray

        def __init__(self, dataset: Dataset):
            data, targets = next(iter(DataLoader(dataset, batch_size=len(dataset))))
            self.data = data.numpy()
            self.targets = targets.numpy()

        def __len__(self):
            return len(self.data)


    print("Loading datasets into memory...")
    trainset = MNISTDataset(trainset)
    testset = MNISTDataset(testset)


    # ==============================================================================
    # Model
    # ==============================================================================

    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)

    print("=" * 80)
    print("Model:")
    print(model)

    # ==============================================================================
    # Training
    # ==============================================================================

    print("=" * 80)
    print("Training...")
    # MNIST fits into memory, so we don't use data loaders.


    @eqx.filter_jit
    def accuracy(
        model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    ) -> Float[Array, ""]:
        """This function takes as input the current model
        and computes the average accuracy on a batch.
        """
        pred_y = jax.vmap(model)(x)
        pred_y = jnp.argmax(pred_y, axis=1)
        return jnp.mean(y == pred_y)


    @eqx.filter_jit
    def loss(
        model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
    ) -> Float[Array, ""]:
        # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
        # a single input input image of shape (1, 28, 28).
        #
        # Therefore, we have to use jax.vmap, which in this case maps our model over the
        # leading (batch) axis.
        pred_y = jax.vmap(model)(x)
        return cross_entropy(pred_y, y).mean()


    def evaluate(model: CNN, dataset: Dataset) -> tuple[float, float]:
        """Computes average loss and accuracy over a dataset."""
        x, y = dataset.data, dataset.targets
        return loss(model, x, y), accuracy(model, x, y)


    def train(
        model: CNN,
        trainset: Dataset,
        testset: Dataset,
        optim: optax.GradientTransformation,
        epochs: int,
        print_every: int,
    ) -> CNN:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def train_step(
            model: CNN,
            opt_state: PyTree,
            x: Float[Array, "batch 1 28 28"],
            y: Int[Array, " batch"],
        ):
            loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        x, y = trainset.data, trainset.targets
        epoch_len = len(trainset) // BATCH_SIZE

        for epoch in range(epochs):
            perm = np.random.permutation(len(trainset))

            for i, train_idx in enumerate(it.batched(perm, BATCH_SIZE)):
                train_idx = np.array(train_idx)
                x_batch = x[train_idx]
                y_batch = y[train_idx]
                model, opt_state, train_loss = train_step(
                    model, opt_state, x_batch, y_batch
                )

                if (i % print_every) == 0 or (i == epoch_len - 1):
                    train_loss, train_accuracy = evaluate(model, trainset)
                    test_loss, test_accuracy = evaluate(model, testset)
                    progress = (i + 1) / epoch_len
                    print(
                        f"[{epoch + 1}/{epochs} {progress:4.0%}] "
                        f"train loss: {train_loss.item():.6f}, "
                        f"test loss: {test_loss.item():.6f}, "
                        f"train accuracy: {train_accuracy.item():.2%}, "
                        f"test accuracy: {test_accuracy.item():.2%}"
                    )
        return model


    optim = optax.adamw(LEARNING_RATE)
    train(model, trainset, testset, optim, EPOCHS, PRINT_EVERY)

    eqx.tree_serialise_leaves(OUT_FILE, model)
