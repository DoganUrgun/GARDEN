"""Small recurrent components used by the SMAX trainer."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class CentralCriticRNN(nn.Module):
    """Centralized recurrent critic returning one value per allied agent."""

    num_agents: int
    hidden_size: int

    @nn.compact
    def __call__(self, hidden: jax.Array, inputs: jax.Array):
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(
            inputs.astype(jnp.float32)
        )
        x = nn.tanh(nn.LayerNorm()(x))
        hidden, x = nn.GRUCell(features=self.hidden_size)(hidden, x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        x = nn.tanh(x)
        values = nn.Dense(self.num_agents, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return hidden, values


def zero_hidden(batch_size: int, hidden_size: int) -> jax.Array:
    return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)
