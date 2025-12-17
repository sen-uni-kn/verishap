# Copyright 2025 David Boetius
# Adapted from https://docs.kidger.site/equinox/examples/mnist/
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torchvision
from jaxtyping import Array, Float, Int, PyTree
from optax.losses import softmax_cross_entropy_with_integer_labels as cross_entropy
from torch.utils.data import DataLoader

from ..models import CNN

# ==============================================================================
# Hyperparameters
# ==============================================================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10
PRINT_EVERY = 100
SEED = 1939
OUT_FILE = "cifar10-cnn.eqxparams"
MODEL_CLS = partial(
    CNN,
    (3, 32, 32),
    10,
    conv_layers=[{"channels": 4}, {"channels": 8}],
    fc_in_sizes=(512, 64),
)

if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)
    np.random.seed(SEED + 1)

    # ==============================================================================
    # Data Loading
    # ==============================================================================

    print("=" * 80)
    print("Downloading CIFAR10 dataset...")

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

    # ==============================================================================
    # Model
    # ==============================================================================

    key, subkey = jax.random.split(key, 2)
    model, state = eqx.nn.make_with_state(MODEL_CLS)(subkey)

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

    # def map_labels(labels: Int[Array, " batch"]) -> Bool[Array, " batch"]:
    #     """Transforms 10 CIFAR10 labels into animal vs. vehicle labels.

    #     Returns:
    #         - 0 if the label is an animal label
    #         - 1 if the label is a vehicle label
    #     """
    #     return (labels == 0) | (labels == 1) | (labels == 8) | (labels == 9)

    @eqx.filter_jit
    def accuracy(
        model: MODEL_CLS,
        state: PyTree,
        x: Float[Array, "batch 3 32 32"],
        y: Int[Array, " batch"],
    ) -> tuple[Float[Array, ""], PyTree]:
        """This function takes as input the current model
        and computes the average accuracy on a batch.
        """
        model = jax.vmap(
            model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        pred_y, state = model(x, state)
        # Animals vs. vehicles
        # pred_y = pred_y >= 0
        # acc = jnp.mean(map_labels(y) == pred_y)
        # return acc, state
        pred_y = jnp.argmax(pred_y, axis=1)
        acc = jnp.mean(y == pred_y)
        return acc, state

    @eqx.filter_jit
    def loss(
        model: MODEL_CLS,
        state: PyTree,
        x: Float[Array, "batch 3 32 32"],
        y: Int[Array, " batch"],
    ) -> tuple[Float[Array, ""], PyTree]:
        # y = map_labels(y)
        model = jax.vmap(
            model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        pred_y, state = model(x, state)
        # loss = sigmoid_binary_cross_entropy(pred_y, y).mean()
        loss = cross_entropy(pred_y, y).mean()
        return loss, state

    def evaluate(
        model: MODEL_CLS, state: PyTree, loader: DataLoader
    ) -> tuple[float, float]:
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
        model: MODEL_CLS,
        state: PyTree,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optim: optax.GradientTransformation,
        epochs: int,
        print_every: int,
    ) -> MODEL_CLS:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def train_step(
            model: MODEL_CLS,
            state: PyTree,
            opt_state: PyTree,
            x: Float[Array, "batch 3 32 32"],
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

        epoch_len = len(train_loader)

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
    model, state = train(
        model, state, train_loader, test_loader, optim, EPOCHS, PRINT_EVERY
    )

    type(model).save(model, state, OUT_FILE)
