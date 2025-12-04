# ---------------------------------------------------------------------------
#  Scale‑by‑Adam: lightweight Optax transformation mirroring classic Adam
# ---------------------------------------------------------------------------
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base, numerics, utils
import chex
# ---------------------------------------------------------------------------
# 1.  Optimiser state container
# ---------------------------------------------------------------------------
class ScaleByAdamState(NamedTuple):
    """State for the Adam preconditioner transformation."""

    count: jax.Array  # Step counter for bias correction
    mu:    object     # Exponential moving average of gradients
    nu:    object     # Exponential moving average of squared gradients


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    scale_eps: bool = False,
    B = None,
    N = None,
    din = None,
    dout = None,
    static=False,
) -> base.GradientTransformation:
  r"""Rescale updates according to the Adam algorithm.

  See :func:`optax.adam` for more details.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum. The variant of Adam with
      Nesterov momentum is described in [Dozat 2016]

  Returns:
    A :class:`optax.GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)
  if scale_eps:
    # batch scaling
    eps = eps / (B**0.5)  # eps ~ sqrt(||E[g^2]||_F) ~ 1/sqrt(B)

  if din is None or dout is None:
    raise ValueError("scale_by_adam requires din/dout trees")
  if scale_eps and N is None:
    raise ValueError("scale_by_adam with scale_eps=True requires an N tree")
  
  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params

    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(mu, b1, numerics.safe_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc),
      )
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
    # unclear why. Other Nadam implementations also omit the extra b2 factor.
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    if scale_eps:
        updates = jax.tree.map(
            lambda m, v, i, o, n: None if m is None else m / (jnp.sqrt(v) + eps/o/n), # eps ~ 1/dout; depth scaling 1/N
            mu_hat,
            nu_hat,
            din,
            dout,
            N,
            is_leaf=lambda x: x is None,
        )
    else:
        updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu) if not static else state

  return base.GradientTransformation(init_fn, update_fn)
