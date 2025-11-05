# Copyright 2025 David Boetius
# Adapted from https://docs.kidger.site/equinox/examples/mnist/
import argparse
import itertools as it
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, Int, PyTree
from optax.losses import softmax_cross_entropy_with_integer_labels as cross_entropy
from sklearn.model_selection import train_test_split

from .models import MLP
from .utils import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Whether the dataset is a regression task. "
        "Otherwise, classification is assumed.",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2810)
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()

    dataset = args.dataset
    is_classification = not args.regression
    hidden_dim = args.hidden_dim
    hidden_layers = args.hidden_layers
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    seed = args.seed
    print_every = args.print_every
    out_file = f"{dataset}-mlp-{hidden_dim}x{hidden_layers}.eqx"

    key = jax.random.PRNGKey(seed)
    np.random.seed(seed + 1)

    # ==============================================================================
    # Data Loading
    # ==============================================================================

    print("=" * 80)
    print(f"Obtaining {dataset} dataset...")
    data, targets = load_dataset(dataset)

    data_mean, data_std = data.mean(axis=0), data.std(axis=0)
    targets_mean, targets_std = targets.mean(axis=0), targets.std(axis=0)

    input_dim = data.shape[-1]
    if is_classification:
        assert targets.dtype in (np.int32, np.int64, np.bool_), (
            "Unsupported target dtype for classification: " + str(targets.dtype)
        )
        targets = targets.astype(np.int32)
        output_dim = len(np.unique(targets))
    else:
        output_dim = targets.shape[-1] if len(targets.shape) > 1 else 1
        targets = targets.reshape(-1, output_dim)

    train_data, test_data, train_targets, test_targets = train_test_split(
        data, targets, test_size=0.2, random_state=seed + 2
    )

    @dataclass
    class Dataset:
        data: np.ndarray
        targets: np.ndarray

        def __len__(self):
            return len(self.data)

    trainset = Dataset(train_data, train_targets)
    testset = Dataset(test_data, test_targets)

    print("Trainset: ", trainset.data.shape, trainset.targets.shape)
    print("Testset: ", testset.data.shape, testset.targets.shape)

    # ==============================================================================
    # Model
    # ==============================================================================

    key, subkey = jax.random.split(key, 2)
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        key=subkey,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        input_norm_stats=(data_mean, data_std),
        output_norm_stats=None if is_classification else (targets_mean, targets_std),
    )

    print("=" * 80)
    print("Model:")
    print(model)

    # ==============================================================================
    # Training
    # ==============================================================================

    print("=" * 80)
    print("Training...")

    @eqx.filter_jit
    def eval_classification(
        model: MLP,
        x: Float[Array, "batch n"],
        y: Int[Array, " batch"],
    ) -> Float[Array, ""]:
        """This function takes as input the current model
        and computes the average accuracy on a batch.
        """
        model = jax.vmap(model, axis_name="batch", in_axes=0, out_axes=0)
        pred_y = model(x)
        pred_y = jnp.argmax(pred_y, axis=1)
        return jnp.mean(y == pred_y)

    # @eqx.filter_jit
    def eval_regression(
        model: MLP,
        x: Float[Array, "batch n"],
        y: Int[Array, " batch"],
    ) -> Float[Array, ""]:
        """This function takes as input the current model
        and computes the root mean squared error, as well as R² on a batch.
        """
        model = jax.vmap(model, axis_name="batch", in_axes=0, out_axes=0)
        pred_y = model(x)
        mse = jnp.mean((pred_y - y) ** 2)
        rmse = jnp.sqrt(mse)
        r2 = 1 - mse / jnp.var(y)
        return rmse, r2

    @eqx.filter_jit
    def loss(
        model: MLP,
        x: Float[Array, "batch n"],
        y: Int[Array, " batch"],
    ) -> Float[Array, ""]:
        model = jax.vmap(model, axis_name="batch", in_axes=0, out_axes=0)
        pred_y = model(x)
        if is_classification:
            return cross_entropy(pred_y, y).mean()
        else:
            return ((pred_y - y) ** 2).mean()

    def evaluate(model: MLP, dataset: Dataset) -> tuple[float, float]:
        """Computes average loss and other statistics over a dataset."""
        inference_model = eqx.nn.inference_mode(model)
        x, y = dataset.data, dataset.targets
        loss_val = loss(inference_model, x, y)
        if is_classification:
            acc_val = eval_classification(inference_model, x, y)
            return loss_val, acc_val
        else:
            rmse_val, r2_val = eval_regression(inference_model, x, y)
            return loss_val, rmse_val, r2_val

    def train(
        model: MLP,
        trainset: Dataset,
        testset: Dataset,
        optim: optax.GradientTransformation,
        epochs: int,
        print_every: int,
    ) -> MLP:
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def train_step(
            model: MLP,
            opt_state: PyTree,
            x: Float[Array, "batch n"],
            y: Int[Array, " batch"],
        ):
            loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        x, y = trainset.data, trainset.targets
        epoch_len = len(trainset) // batch_size

        for epoch in range(epochs):
            perm = np.random.permutation(len(trainset))

            for i, train_idx in enumerate(it.batched(perm, batch_size)):
                train_idx = np.array(train_idx)
                x_batch = x[train_idx]
                y_batch = y[train_idx]
                model, opt_state, train_loss = train_step(
                    model, opt_state, x_batch, y_batch
                )

                if (i % print_every) == 0 or (i == epoch_len - 1):
                    progress = (i + 1) / epoch_len
                    if is_classification:
                        train_loss, train_accuracy = evaluate(model, trainset)
                        test_loss, test_accuracy = evaluate(model, testset)
                        print(
                            f"[{epoch + 1}/{epochs} {progress:4.0%}] "
                            f"train loss: {train_loss.item():.6f}, "
                            f"test loss: {test_loss.item():.6f}, "
                            f"train accuracy: {train_accuracy.item():.2%}, "
                            f"test accuracy: {test_accuracy.item():.2%}"
                        )
                    else:
                        train_loss, train_rmse, train_r_squared = evaluate(
                            model, trainset
                        )
                        test_loss, test_rmse, test_r_squared = evaluate(model, testset)
                        print(
                            f"[{epoch + 1}/{epochs} {progress:4.0%}] "
                            f"train loss: {train_loss.item():.6f}, "
                            f"test loss: {test_loss.item():.6f}, "
                            f"train RMSE: {train_rmse.item():.6f}, "
                            f"test RMSE: {test_rmse.item():.6f}, "
                            f"train R²: {train_r_squared.item():.6f}, "
                            f"test R²: {test_r_squared.item():.6f}"
                        )
        return model

    optim = optax.adamw(learning_rate)
    model = train(model, trainset, testset, optim, epochs, print_every)

    info_dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
    }
    if is_classification:
        train_loss, train_accuracy = evaluate(model, trainset)
        test_loss, test_accuracy = evaluate(model, testset)
        info_dict["train_loss"] = train_loss.item()
        info_dict["train_accuracy"] = train_accuracy.item()
        info_dict["test_loss"] = test_loss.item()
        info_dict["test_accuracy"] = test_accuracy.item()
    else:
        train_loss, train_rmse, train_r_squared = evaluate(model, trainset)
        test_loss, test_rmse, test_r_squared = evaluate(model, testset)
        info_dict["train_loss"] = train_loss.item()
        info_dict["train_rmse"] = train_rmse.item()
        info_dict["train_r_squared"] = train_r_squared.item()
        info_dict["test_loss"] = test_loss.item()
        info_dict["test_rmse"] = test_rmse.item()
        info_dict["test_r_squared"] = test_r_squared.item()

    model.save(out_file, extra_info=info_dict)
