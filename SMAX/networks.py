"""Masked recurrent actor matching JaxMARL's SMAX MAPPO reference shape."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class ActorRNN(nn.Module):
    action_dim: int
    hidden_size: int

    @nn.compact
    def __call__(self, hidden: jax.Array, inputs: jax.Array):
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(
            inputs.astype(jnp.float32)
        )
        x = nn.relu(x)
        hidden, x = nn.GRUCell(features=self.hidden_size)(hidden, x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.orthogonal(2.0))(x)
        x = nn.relu(x)
        return hidden, nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
