import numpy as np
import jax
from jax import tree_util


def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(abs(x))


def symexp(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * (np.exp(np.abs(x)) - 1)

def device_put(x, device=None):
    if device is None:
        return jax.device_put(x, device=device)
    return tree_util.tree_map(
        lambda leaf: jax.device_put(leaf, device.reshape([1 for _ in range(leaf.ndim)])),
        x
    )
