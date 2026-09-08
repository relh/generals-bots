"""Size-independent actor/critic with a single pass and optional castle builds."""
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

from generals.core.action import compute_valid_move_mask_obs


def features(obs):
    x = obs.as_tensor().astype(jnp.float32)
    for channel in (0, 9, 10, 11, 12):
        x = x.at[channel].set(jnp.log1p(jnp.maximum(x[channel], 0)) / 8.0)
    return x.at[13].set(jnp.minimum(x[13] / 1200.0, 2.0))


def build_costs(obs):
    offsets = jnp.arange(-6, 7)
    kernel = jnp.maximum(0, 14 - 2 * (jnp.abs(offsets[:, None]) + jnp.abs(offsets[None, :])))
    own_structures = ((obs.generals | obs.castles) & obs.owned_cells).astype(jnp.float32)
    return 35 + convolve2d(own_structures, kernel, mode="same")


def action_mask(obs, build_enabled=False):
    # Hidden structures may be mountains OR castles. Do not inspect hidden truth:
    # an attempted move there is allowed, and may be rejected by the simulator.
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1)
    builds = (obs.owned_cells & ~obs.generals & ~obs.castles
              & (obs.armies >= build_costs(obs))) if build_enabled else jnp.zeros_like(obs.owned_cells)
    return jnp.concatenate([moves.reshape(-1), moves.reshape(-1), builds.reshape(-1), jnp.ones(1, bool)])


def decode_action(index, shape):
    h, w = shape
    cells = h * w
    channel, position = index // cells, index % cells
    action = jnp.array([jnp.where(channel == 8, 2, 0), position // w, position % w,
                        channel % 4, (channel >= 4) & (channel < 8)], dtype=jnp.int32)
    return jnp.where(index == 9 * cells, jnp.array([1, 0, 0, 0, 0], jnp.int32), action)


def encode_action(action, shape):
    h, w = shape
    channel = jnp.where(action[0] == 2, 8, action[3] + 4 * action[4])
    return jnp.where(action[0] == 1, 9 * h * w, channel * h * w + action[1] * w + action[2])


class SpatialPolicy(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    actor: eqx.nn.Conv2d
    critic: eqx.nn.Linear
    pass_head: eqx.nn.Linear

    def __init__(self, key, width=32):
        keys = jax.random.split(key, 6)
        self.conv1 = eqx.nn.Conv2d(14, width, 3, padding=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(width, width, 3, padding=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(width, width, 3, padding=1, key=keys[2])
        self.actor = eqx.nn.Conv2d(width, 9, 1, key=keys[3])
        self.critic = eqx.nn.Linear(2 * width, 1, key=keys[4])
        self.pass_head = eqx.nn.Linear(2 * width, 1, key=keys[5])

    def __call__(self, x, mask):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(x + self.conv2(x))
        x = jax.nn.relu(x + self.conv3(x))
        pooled = jnp.concatenate([x.mean((1, 2)), x.max((1, 2))])
        logits = jnp.concatenate([self.actor(x).reshape(-1), self.pass_head(pooled)])
        return jnp.where(mask, logits, -1e9), self.critic(pooled)[0]

    def act(self, obs, key, build_enabled=False, deterministic=False):
        logits, _ = self(features(obs), action_mask(obs, build_enabled))
        index = jnp.argmax(logits) if deterministic else jax.random.categorical(key, logits)
        return decode_action(index, obs.armies.shape)
