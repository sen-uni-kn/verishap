# Copyright 2025 David Boetius
# Adapted from https://docs.kidger.site/equinox/examples/mnist/
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torchvision
from jaxtyping import Array, Float, Int, PyTree
from optax.losses import sigmoid_binary_cross_entropy as binary_cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets import NIHChestXrayDataset
from ..models import CNN

# ==============================================================================
# Hyperparameters
# ==============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 50
PRINT_EVERY = 100
SEED = 1243
OUT_FILE = "nih-chest-xray-cnn.eqxparams"
MODEL_CLS = partial(
    CNN,
    (1, 100, 100),
    1,
    conv_layers=[{"channels": 16}, {"channels": 32}, {"channels": 64}, {"channels": 128}],
    fc_in_sizes=(4608, 256),
)


# Turn multi-class labels into finding/no-finding binary labels.
def target_transform(x: torch.Tensor) -> torch.Tensor:
    return (x[..., :-1] > 0.0).any(dim=-1)


if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)
    np.random.seed(SEED + 1)

    # ==============================================================================
    # Data Loading
    # ==============================================================================

    print("=" * 80)
    print("Obtaining NIH Chest X-ray dataset...")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((100, 100)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
        ]
    )

    trainset = NIHChestXrayDataset(
        split="train",
        transform=transform,
        target_transform=target_transform,
    )
    testset = NIHChestXrayDataset(
        split="test",
        transform=transform,
        target_transform=target_transform,
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

    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=20,
        multiprocessing_context="forkserver",
    )
    train_loader2 = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=20,
        multiprocessing_context="forkserver",
    )
    test_loader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        multiprocessing_context="forkserver",
    )

    @eqx.filter_jit
    def loss(
        model: MODEL_CLS,
        state: PyTree,
        x: Float[Array, "batch 1 32 32"],
        y: Int[Array, " batch"],
    ) -> tuple[Float[Array, ""], PyTree]:
        model = jax.vmap(
            model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
        pred_y, state = model(x, state)
        loss = binary_cross_entropy(pred_y, y).mean()
        accuracy = (y == (pred_y >= 0.0)).mean()
        return loss, (accuracy, state)

    def evaluate(
        model: MODEL_CLS, state: PyTree, loader: DataLoader
    ) -> tuple[float, float]:
        """Computes average loss and accuracy over a dataset."""
        inference_model = eqx.nn.inference_mode(model)
        loss_val = 0.0
        acc_val = 0.0
        for x, y in tqdm(loader):
            x, y = x.numpy(), y.numpy()
            loss_, (acc, state) = loss(inference_model, state, x, y)
            loss_val += loss_
            acc_val += acc
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
            x: Float[Array, "batch 1 32 32"],
            y: Int[Array, " batch"],
        ):
            (loss_value, (accuracy, state)), grads = eqx.filter_value_and_grad(
                loss, has_aux=True
            )(model, state, x, y)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, state, opt_state, loss_value, accuracy

        epoch_len = len(train_loader)

        losses, accuracies = [], []
        for epoch in range(epochs):
            for i, (x_batch, y_batch) in enumerate(iter(train_loader)):
                x_batch, y_batch = x_batch.numpy(), y_batch.numpy()
                model, state, opt_state, train_loss, train_accuracy = train_step(
                    model, state, opt_state, x_batch, y_batch
                )
                losses.append(train_loss)
                accuracies.append(train_accuracy)

                if (i % print_every) == 0:
                    loss_ = np.mean(losses)
                    accuracy_ = np.mean(accuracies)
                    progress = (i + 1) / epoch_len
                    print(
                        f"[{epoch + 1}/{epochs} {progress:4.0%}] "
                        f"avg loss: {loss_.item():.6f}, "
                        f"avg accuracy: {accuracy_.item():.2%}"
                    )
                    losses, accuracies = [], []

            test_loss, test_accuracy = evaluate(model, state, test_loader)
            progress = (i + 1) / epoch_len
            print(
                f"[{epoch + 1}/{epochs} {progress:4.0%}] "
                f"test loss: {test_loss.item():.6f}, "
                f"test accuracy: {test_accuracy.item():.2%}"
            )
        return model, state

    optim = optax.adamw(LEARNING_RATE)
    model, state = train(
        model, state, train_loader, test_loader, optim, EPOCHS, PRINT_EVERY
    )

    type(model).save(model, state, OUT_FILE)
