#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from torchvision.datasets import EMNIST


def load_data() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    - Loads the EMNIST dataset (emnist-bymerge, 42 labels)
    - Normalizes image data to [0, 1]

    Returns: Training set images, training set labels, test set images,
        test set labels
    """

    data_train = EMNIST(".datasets", split="bymerge", train=True, download=True)
    data_test = EMNIST(".datasets", split="bymerge", train=False, download=True)

    x_train = data_train.data / 255.0
    x_test = data_test.data / 255.0
    y_train = data_train.targets
    y_test = data_test.targets

    x_train = x_train.numpy()
    x_test = x_test.numpy()
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()


class CNN(nn.Module):
    """
    - Input of dimension (1,28,28,1)
    - Conv. Layer: 8 filters of 3x3 size + ReLU
    - Halving of spatial dimension with 2x2 pooling window
    - Flatten for fully connected layer
    - Fully connected layer (42 units) + ReLU
    - log-softmax to produce probabilities for 42 outcomes
    """

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, 28, 28, 1)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=47)(x)
        return x


# Training procedure in jax framework:
#
# - Defining training state object (random parameters, optimizer, LR)
# - Defining training step function:
#     - Pass inputs through function
#     - Compute loss (cross entropy using log_softmax outputs + l2_regularization)
#     - Compute gradients
#     - update params in training state via gradients
#
# - Evaluation function (forward pass with trained model)


def create_train_state(rng, learning_rate, model, input_shape):
    params = model.init(rng, jnp.ones(input_shape))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def l2_loss(x, lambda_):
    return lambda_ * jnp.mean(x**2)


@jax.jit
def train_step(state, batch, lambda_):
    def loss_fn(params):
        scores = state.apply_fn({"params": params}, batch["images"])
        one_hot = jax.nn.one_hot(batch["labels"], 47)
        logits = nn.log_softmax(scores)
        cross_entropy_loss = -jnp.mean(one_hot * logits)
        l2_reg_loss = sum(l2_loss(w, lambda_) for w in jax.tree.leaves(params))
        loss = cross_entropy_loss + l2_reg_loss
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, batch):
    scores = state.apply_fn({"params": state.params}, batch["images"])
    return scores


# Initializing the model and training state
# Run training loop
# Evaluate on test data (accuracy 88.55%)

learning_rate = 0.001
input_shape = (1, 28, 28, 1)

model = CNN()
rng_key = jax.random.PRNGKey(0)
rng_key, subkey = jax.random.split(rng_key)
state = create_train_state(subkey, learning_rate, model, input_shape)


# Training loop
num_epochs = 2
batch_size = 64
lambda_ = 0.001

epoch_len = len(x_train) // batch_size
log_frequency = epoch_len // 100

print("checkpoint")

for epoch in range(num_epochs):
    # reshuffle the training set every epoch
    rng_key, subkey = jax.random.split(rng_key)
    train_idx = jax.random.permutation(subkey, len(x_train))
    for i in range(epoch_len):
        batch_images = x_train[train_idx[i * batch_size : (i + 1) * batch_size]]
        batch_labels = y_train[train_idx[i * batch_size : (i + 1) * batch_size]]
        batch = {"images": batch_images, "labels": batch_labels}
        state, loss = train_step(state, batch, lambda_)

        if i % log_frequency == log_frequency - 1:
            print(f"[Epoch {epoch + 1}, {100 * i / epoch_len:3.0f}%] Loss: {loss:.6f}")

# Evaluation
preds = []
for i in range(0, len(x_test), batch_size):
    batch_images = x_test[i : i + batch_size]
    batch = {"images": batch_images, "labels": None}
    scores = eval_step(state, batch)
    preds.extend(jnp.argmax(scores, axis=-1))

accuracy = jnp.mean(jnp.array(preds) == y_test)
print(f"Test set accuracy: {accuracy * 100:.2f}%")


def save_model(state, filename):
    arrays, _ = jax.tree.flatten(state)
    np.savez(filename, *[np.asarray(a) for a in arrays])


save_model(state, "../emnist_conv_flax.pkl")
