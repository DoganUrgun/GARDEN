"""Feed-forward MAPPO networks for fully observed classic Overcooked."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class _CNN(nn.Module):
    """The convolutional encoder from JaxMARL's Overcooked PPO reference."""

    height: int
    width: int
    channels: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = inputs.astype(jnp.float32).reshape(
            (inputs.shape[0], self.height, self.width, self.channels)
        )
        for kernel in ((5, 5), (3, 3), (3, 3)):
            x = nn.Conv(
                features=32,
                kernel_size=kernel,
                padding="SAME",
                kernel_init=nn.initializers.orthogonal(np.sqrt(2)),
                bias_init=nn.initializers.zeros_init(),
            )(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        return nn.relu(x)


class ActorCNN(nn.Module):
    action_dim: int
    height: int
    width: int

    @nn.compact
    def __call__(self, hidden: jax.Array, inputs: jax.Array):
        x = _CNN(height=self.height, width=self.width, channels=26)(inputs)
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(x)
        return hidden, logits


class CentralCriticCNN(nn.Module):
    num_agents: int
    height: int
    width: int

    @nn.compact
    def __call__(self, hidden: jax.Array, inputs: jax.Array):
        # Each agent sees the complete map but agent channels are reordered.
        # Channel-wise concatenation therefore gives the centralized critic a
        # stable joint representation without exposing hidden environment data.
        x = _CNN(height=self.height, width=self.width, channels=26 * self.num_agents)(inputs)
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.relu(x)
        values = nn.Dense(
            self.num_agents,
            kernel_init=nn.initializers.orthogonal(1.0),
        )(x)
        return hidden, values

