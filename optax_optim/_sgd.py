# ---------------------------------------------------------------------------
#  Scale‑by‑Adam: lightweight Optax transformation mirroring classic Adam
# ---------------------------------------------------------------------------
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base, numerics, utils
import chex
from utils import prune_tree
# ---------------------------------------------------------------------------
# 1.  Optimiser state container
# ---------------------------------------------------------------------------
class ScaleByMomentumState(NamedTuple):
    """State for the Adam preconditioner transformation."""
    count: jax.Array  # Step counter for bias correction
    mu:    object

def scale_by_momentum(
    b1: float = 0.9,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    static=False

) -> base.GradientTransformation:

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    return ScaleByMomentumState(mu=mu, count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    count_inc = numerics.safe_increment(state.count)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    mu = otu.tree_cast(mu, mu_dtype)

    return mu_hat, ScaleByMomentumState(mu=mu, count=count_inc) if not static else state

  return base.GradientTransformation(init_fn, update_fn)