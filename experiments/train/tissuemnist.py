# Copyright 2025 David Boetius
# Adapted from https://docs.kidger.site/equinox/examples/mnist/
import itertools as it
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import medmnist
import numpy as np
import optax
import torchvision
from jaxtyping import Array, Float, Int, PyTree
from optax.losses import softmax_cross_entropy_with_integer_labels as cross_entropy
from torch.utils.data import DataLoader, Dataset

from ..models import resnet18

# ==============================================================================
# Hyperparameters
# ==============================================================================

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 30
PRINT_EVERY = 100
SEED = 2011
OUT_FILE = "tissuemnist-resnet18.eqx"

if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)
    np.random.seed(SEED + 1)

    # ==============================================================================
    # Data Loading
    # ==============================================================================

    print("=" * 80)
    print("Downloading TissueMNIST dataset...")

    trainset = medmnist.TissueMNIST(
        root=".datasets/medmnist",
        split="train",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    testset = medmnist.TissueMNIST(
        root=".datasets/medmnist",
        split="test",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    # ==============================================================================
    # Model
    # ==============================================================================

    key, subkey = jax.random.split(key, 2)
    # model, state = eqx.nn.make_with_state(CNN)(
    #     (1, 28, 28),
    #     8,
    #     subkey,
    #     conv_layers=[{"channels": 8}, {"channels": 16}, {"channels": 32}],
    #     fc_in_sizes=(288, 64),
    # )
    model, state = eqx.nn.make_with_state(resnet18)(
        subkey, in_channels=1, num_classes=8
    )

    print("=" * 80)
    print("Model:")
    print(model)

    # ==============================================================================
    # Training
    # ==============================================================================

    print("=" * 80)
    print("Training...")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader2 = DataLoader(trainset, batch_size=10 * BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=10 * BATCH_SIZE, shuffle=False)

    @eqx.filter_jit
    def accuracy(
        model: PyTree,
        state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch t"],
    ) -> tuple[Float[Array, ""], PyTree]:
        """This function takes as input the current model
        and computes the average accuracy on a batch.
        """
        model = jax.vmap(
            model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        pred_y, state = model(x, state)
        pred_y = jnp.argmax(pred_y, axis=-1)
        acc = jnp.mean(y.squeeze() == pred_y)
        return acc, state

    @eqx.filter_jit
    def loss(
        model: PyTree,
        state: PyTree,
        x: Float[Array, " batch 1 28 28"],
        y: Int[Array, " batch t"],
    ) -> tuple[Float[Array, ""], PyTree]:
        # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
        # a single input input image of shape (1, 28, 28).
        #
        # Therefore, we have to use jax.vmap, which in this case maps our model over the
        # leading (batch) axis.
        model = jax.vmap(
            model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        pred_y, state = model(x, state)
        loss = cross_entropy(pred_y, y.squeeze()).mean()
        return loss, state

    def evaluate(model: PyTree, state: PyTree, loader: DataLoader) -> tuple[float, float]:
        """Computes average loss and accuracy over a dataset."""
        inference_model = eqx.nn.inference_mode(model)
        loss_val = 0.0
        acc_val = 0.0
        for x, y in loader:
            x, y = x.numpy(), y.numpy()
            loss_val += loss(inference_model, state, x, y)[0]
            acc_val += accuracy(inference_model, state, x, y)[0]
        return loss_val / len(loader), acc_val / len(loader)

    def train(
        model: PyTree,
        state: PyTree,
        trainset: Dataset,
        testset: Dataset,
        optim: optax.GradientTransformation,
        epochs: int,
        print_every: int,
    ) -> PyTree:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def train_step(
            model: PyTree,
            state: PyTree,
            opt_state: PyTree,
            x: Float[Array, "batch 1 28 28"],
            y: Int[Array, " batch"],
        ):
            (loss_value, state), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
                model, state, x, y
            )
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, state, opt_state, loss_value

        epoch_len = len(trainset) // BATCH_SIZE

        for epoch in range(epochs):
            for i, (x_batch, y_batch) in enumerate(iter(train_loader)):
                x_batch, y_batch = x_batch.numpy(), y_batch.numpy()
                model, state, opt_state, train_loss = train_step(
                    model, state, opt_state, x_batch, y_batch
                )

                if (i % print_every) == 0 or (i == epoch_len - 1):
                    train_loss, train_accuracy = evaluate(model, state, train_loader2)
                    test_loss, test_accuracy = evaluate(model, state, test_loader)
                    progress = (i + 1) / epoch_len
                    print(
                        f"[{epoch + 1}/{epochs} {progress:4.0%}] "
                        f"train loss: {train_loss.item():.6f}, "
                        f"test loss: {test_loss.item():.6f}, "
                        f"train accuracy: {train_accuracy.item():.2%}, "
                        f"test accuracy: {test_accuracy.item():.2%}"
                    )
        return model, state

    optim = optax.adamw(LEARNING_RATE)
    model, state = train(model, state, trainset, testset, optim, EPOCHS, PRINT_EVERY)

    model.save(state, OUT_FILE)
