import functools
from typing import NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax.transforms import _accumulation
from optax.transforms import _adding

def static_scale_by_identity() -> base.GradientTransformation:
  def init_fn(_):
    raise Exception("Cannot initialize static preconditioner")

  # Take opt_state as a placeholder
  def update_fn(updates, _, params=None):
    del params
    return updates, None

  return base.GradientTransformation(init_fn, update_fn)


def static_scale_by_adam(
    b2: float=0.95,
    eps: float = 1e-8,
    scale_eps: bool = False,
    N = None,
    B = None,
) -> base.GradientTransformation:
  
  if scale_eps:
    # batch scaling
    eps = eps / (B**0.5)  # eps ~ sqrt(||E[g^2]||_F) ~ 1/sqrt(B)


  def init_fn(_):
    raise Exception("Cannot initialize static preconditioner")

  def update_fn(updates, state, params=None):
    del params
    nu_hat = otu.tree_bias_correction(state.nu, b2, state.count)
    if scale_eps:
      updates = jax.tree.map(
          lambda m, v, din, dout, n: None if m is None else m / (jnp.sqrt(v) + eps/dout/n), # eps ~ 1/dout; depth scaling 1/N
          updates,
          nu_hat,
          state.din,
          state.dout,
          N,
          is_leaf=lambda x: x is None,
      )
    else:
        updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            updates,
            nu_hat,
            is_leaf=lambda x: x is None,
        )

    return updates, None

  return base.GradientTransformation(init_fn, update_fn)

def static_scale_by_shampoo(grafting) -> base.GradientTransformation:

  def init_fn(_):
    raise Exception("Cannot initialize static preconditioner")

  def update_fn(updates, state, params=None):
    del params

    updates = jax.tree.map(lambda g, l, r: l @ g @ r, updates, state.L_inv, state.R_inv)

    if grafting:
      updates = jax.tree.map(lambda a, s: a*s, state.alpha, updates)

    return updates, None


  return base.GradientTransformation(init_fn, update_fn)

def _apply_preconditioner(
    grad: jnp.ndarray,
    mu_hat: jnp.ndarray,
    *,
    eps: float = 1e-12,
    choose_side: str = "auto",
    scale_eps = False,
    din = 1,
    dout = 1,
) -> jnp.ndarray:
    """Apply a Muon preconditioner to a single gradient tensor.

    Parameters
    ----------
    grad:
        The gradient update of shape ``(d,)`` or ``(d, 1)``.
    mu_hat:
        Second‑moment estimate of shape ``(m, n)`` such that
        ``mu_hat = U @ diag(s) @ V.T``.
    eps:
        Jitter for numerical stability.
    choose_side: {"u", "v", "auto"}
        Which symmetric factor to use for the preconditioner:

            * ``"u"`` → :math:`P = U\,\Sigma^{-1}\,U^T`
            * ``"v"`` → :math:`P = V\,\Sigma^{-1}\,V^T`
            * ``"auto"`` → pick the cheaper of the two; i.e. when
              ``m <= n`` use *U*, else use *V*.
    """
    if scale_eps:
       eps *= (dout/din) ** 0.5
    # Ensure 2‑D shapes
    assert mu_hat.ndim == 2, "Expect 2D parameter"

    # Add a small diagonal term for stability before the SVD.
    mu_hat_stable = mu_hat + eps * jnp.eye(mu_hat.shape[0], dtype=mu_hat.dtype)

    # Full SVD on the stabilised matrix.
    # We request compact matrices because this tends to be faster and uses less memory.
    U, s, Vh = jnp.linalg.svd(mu_hat_stable, full_matrices=False)

    # Inverse singular values – clip to avoid exploding values if s is tiny.
    s_inv = jnp.where(s > eps, 1.0 / s, 1.0 / eps)

    # Decide which side gives a better‑conditioned transformation.
    if choose_side == "auto":
        use_u = mu_hat.shape[0] <= mu_hat.shape[1]
    else:
        use_u = choose_side == "u"

    # Apply P to grad → P·g = U Σ⁻¹ Uᵀ g  or  V Σ⁻¹ Vᵀ g.
    if use_u:
        z = U.T @ grad          # Uᵀ g
        z = z * s_inv           # Σ⁻¹ (Uᵀ g)
        return U @ z            # U Σ⁻¹ Uᵀ g
    else:
        V = Vh.T
        z = grad @   V          # Vᵀ g
        z = z * s_inv           # Σ⁻¹ (Vᵀ g)
        return z @ V.T          # V Σ⁻¹ Vᵀ g


def static_scale_by_muon(
    beta: float,
    scale_eps: bool,
    *,
    eps: float = 1e-12,
    choose_side: str = "auto",
) -> base.GradientTransformation:
    """Static Muon preconditioner.

    Assumes you chain it *after* an optimizer that maintains
    ``state.mu`` (the exponential second moment) and ``state.count_inc``.

    For each matrix parameter the update is transformed as

    .. math::

        g \leftarrow P g, \qquad
        P = \begin{cases}
            U\,\Sigma^{-1}\,U^T & (m \le n) \\
            V\,\Sigma^{-1}\,V^T & (m > n)
        \end{cases}

    where ``mu_hat = U Σ Vᵀ`` is the bias‑corrected second moment.
    """

    def init_fn(_):
        raise ValueError(
            "static_scale_by_muon is stateless – initialise the preceding Muon "
            "optimizer and chain this transformation afterwards."
        )

    def update_fn(
        updates,  # pytree of gradients
        state,  # expecting .mu and .count_inc in the preceding transform's state
        params = None,
    ):
        del params  # Not needed

        # Bias‑correct `mu` following the Muon paper.
        mu_hat = otu.tree_bias_correction(state.mu, beta, state.count_inc)

        # Apply the preconditioner element‑wise through the pytree.
        scaled_updates = jax.tree_util.tree_map(
            lambda g, m, i, o: _apply_preconditioner(g, m, eps=eps, scale_eps=scale_eps, choose_side=choose_side, din=i, dout=o),
            updates,
            mu_hat,
            state.din,
            state.dout
        )

        return scaled_updates, None  # no internal state

    return base.GradientTransformation(init_fn, update_fn)