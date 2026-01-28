#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.

import pickle

import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

"""
Define model architecture
Load parameters and reconstruct model
Test on some random inputs
"""


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
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
        x = nn.log_softmax(x)
        return x


# Load trained parameters
def load_model(filename):
    with open(filename, "rb") as f:
        params = pickle.load(f)
    return params


# Create a new train state with the loaded parameters
def create_train_state_with_params(rng, model, params, learning_rate):
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({"params": state.params}, batch["images"])
    return logits


# Initialize model and new RNG key
model = CNN()
rng = jax.random.PRNGKey(0)
learning_rate = 0.001


# Load the model parameters
loaded_params = load_model("../emnist_conv_flax.pkl")
loaded_state = create_train_state_with_params(rng, model, loaded_params, learning_rate)


# Generate random noise input and print predictions
batch_size = 1
input_shape = (5, 28, 28, 1)

rng = jax.random.PRNGKey(42)
x_test_noise = jax.random.normal(rng, input_shape)


preds = []
for _ in range(0, batch_size, batch_size):
    batch = {"images": x_test_noise, "labels": None}
    logits = eval_step(loaded_state, batch)
    preds.extend(np.argmax(logits, axis=-1))

print(preds)
