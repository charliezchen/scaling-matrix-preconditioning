# ---------------------------------------------------------------------------
#  Shampoo variant with fast compile:  batched inverse‑roots + lax.cond gate
# ---------------------------------------------------------------------------
import functools
from typing import NamedTuple, Optional, List, Tuple

import jax
import jax.numpy as jnp
import chex
from optax import tree_utils as otu
from optax._src import base, numerics
from optax_optim._google_shampoo import matrix_inverse_pth_root

# ---------------------------------------------------------------------------
# 1.  Low‑level inverse‑pth‑root routine (one JIT per matrix shape)
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("p", "rel_eps"))
def _inv_root(mat: jnp.ndarray, p: float, eps: jnp.ndarray, rel_eps: bool) -> jnp.ndarray:
    """Return (mat + eps·I)^(-p) using an iterative method."""
    return matrix_inverse_pth_root(mat, 1/p, ridge_epsilon=eps, rel_eps=rel_eps)[0] # (res_mat, error)

@functools.partial(jax.jit, static_argnames=("p",))
def _inv_root_eigh(mat: jnp.ndarray, p: float, eps: jnp.ndarray) -> jnp.ndarray:
    """Return (mat + eps·I)^(-p) using an iterative method."""
    eigval, eigvec = jnp.linalg.eigh(mat + eps * jnp.eye(mat.shape[0], dtype=mat.dtype))
    eigval = jnp.maximum(eigval, eps)
    return eigvec @ jnp.diag(eigval ** (-p)) @ eigvec.T

# ---------------------------------------------------------------------------
# 2.  Optimiser state container
# ---------------------------------------------------------------------------
class ScaleByShampooState(NamedTuple):
    count: jax.Array
    L:     object
    R:     object
    L_inv: object
    R_inv: object
    mu:    object
    nu:    object


# ---------------------------------------------------------------------------
# 3.  Public Optax transformation
# ---------------------------------------------------------------------------
def scale_by_shampoo(
        *,
        b1: float = 0.9,
        b2: float = 0.999,
        freq: int = 1,
        kl: float = 0.25,
        kr: float = 0.25,
        adam_eps: float = 1e-8,
        matrix_eps: float = 1e-6,
        grafting: bool = False,
        scale_eps: bool = False,
        B = None,
        N = None,
        static = False,
        eigh = False,
        rel_eps = False,
        block_size: int = 0,
        nb_in = None,
        nb_out = None,
        din = None,
        dout = None,
        max_precond_dim: int = 0,
    ) -> base.GradientTransformation:
    """Shampoo preconditioner with compile-time optimisations."""

    if din is None or dout is None or nb_in is None or nb_out is None:
        raise ValueError("scale_by_shampoo requires din/dout/nb_in/nb_out trees")
    if N is None:
        raise ValueError("scale_by_shampoo requires N tree")

    # ------------------------- init ----------------------------------------
    if rel_eps:
        assert not eigh, "EIGH is not supported with relative eps"
    if scale_eps:
        # batch scaling
        if not rel_eps:
            matrix_eps = matrix_eps / B #/ (N ** 2) # eps ~ E[G^2] ~ 1/B;
        else:
            matrix_eps = matrix_eps # since the newton iteration uses relative eps
        adam_eps = adam_eps / (B**0.5) #/ N # eps ~ sqrt(E[G^2]) ~ 1/sqrt(B);
        graft_eps = adam_eps * (B**(kl + kr - 1/2)) #* (N ** (2*(kl + kr) - 1)) # eps ~ sqrt(E[G^2]) ~ 1/sqrt(B)
    else:
        matrix_eps = matrix_eps
        adam_eps = adam_eps
        graft_eps = adam_eps
    # ---- blocking helpers (2D only) ----
    def _compute_block_slices(m: int, n: int, bsz: int) -> Tuple[List[slice], List[slice]]:
        if bsz <= 0 or (m <= bsz and n <= bsz):
            return [slice(0, m)], [slice(0, n)]
        def _mk_slices(dim):
            if bsz <= 0 or dim <= bsz:
                return [slice(0, dim)]
            starts = list(range(0, dim, bsz))
            return [slice(s, min(s + bsz, dim)) for s in starts]
        return _mk_slices(m), _mk_slices(n)

    def _compute_block_slices_with_gate(m: int, n: int, bsz: int, left_id: bool, right_id: bool) -> Tuple[List[slice], List[slice]]:
        rs, cs = _compute_block_slices(m, n, bsz)
        if left_id:
            rs = [slice(0, m)]
        if right_id:
            cs = [slice(0, n)]
        return rs, cs

    def _stack_blocks(x: jnp.ndarray, row_slices: List[slice], col_slices: List[slice], row_bsz: int, col_bsz: int) -> jnp.ndarray:
        blocks = []
        for rs in row_slices:
            for cs in col_slices:
                blk = x[rs, cs]
                br, bc = blk.shape
                pr = (0, row_bsz - br)
                pc = (0, col_bsz - bc)
                blk = jnp.pad(blk, (pr, pc))
                blocks.append(blk)
        return jnp.stack(blocks, axis=0)

    def _merge_stacked_blocks(stacked: jnp.ndarray, m: int, n: int,
                              row_slices: List[slice], col_slices: List[slice], row_bsz: int, col_bsz: int) -> jnp.ndarray:
        out = jnp.zeros((m, n), dtype=stacked.dtype)
        idx = 0
        for rs in row_slices:
            br = rs.stop - rs.start
            for cs in col_slices:
                bc = cs.stop - cs.start
                blk = stacked[idx, :br, :bc]
                out = out.at[rs, cs].set(blk)
                idx += 1
        return out

    def init_fn(params):
        # Some callers (e.g. optax.partition/multi_transform) wrap params with
        # optax's MaskedNode for leaves not belonging to this transform. Those
        # nodes are tuple-like with no elements and do not have shape/dtype.
        # We therefore guard all leaf computations to gracefully passthrough
        # masked leaves.

        def _has_array_attrs(x):
            return hasattr(x, "shape") and hasattr(x, "dtype")

        # Always-stacked state per leaf
        def _init_leaf(w, din_i, dout_i, n_i, nb_in_i, nb_out_i):
            # Passthrough masked nodes
            if not _has_array_attrs(w):
                return w
            m, n = w.shape
            left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
            right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
            rs, cs = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            Bn = len(rs) * len(cs)
            row_bs = (block_size if (block_size > 0 and len(rs) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cs) > 1) else n)
            if scale_eps:
                epsL = matrix_eps * (din_i / dout_i) / (n_i ** 2) / (nb_in_i * nb_out_i)
                epsR = matrix_eps * (din_i / dout_i) / (n_i ** 2) / (nb_in_i * nb_out_i)
                L = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=w.dtype) * epsL, (Bn, row_bs, row_bs))
                R = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=w.dtype) * epsR, (Bn, col_bs, col_bs))
                L_inv = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=w.dtype) / epsL, (Bn, row_bs, row_bs))
                R_inv = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=w.dtype) / epsR, (Bn, col_bs, col_bs))
            else:
                L = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=w.dtype) * matrix_eps, (Bn, row_bs, row_bs))
                R = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=w.dtype) * matrix_eps, (Bn, col_bs, col_bs))
                L_inv = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=w.dtype) / matrix_eps, (Bn, row_bs, row_bs))
                R_inv = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=w.dtype) / matrix_eps, (Bn, col_bs, col_bs))
            # if left_id:
            #     L = jax.lax.with_sharding_constraint(L, jax.sharding.PartitionSpec(None, "data"))
            #     L_inv = jax.lax.with_sharding_constraint(L_inv, jax.sharding.PartitionSpec(None, "data"))
            # if right_id:
            #     R = jax.lax.with_sharding_constraint(R, jax.sharding.PartitionSpec(None, "data"))
            #     R_inv = jax.lax.with_sharding_constraint(R_inv, jax.sharding.PartitionSpec(None, "data"))

            return L, R, L_inv, R_inv
        stacked = jax.tree.map(_init_leaf, params, din, dout, N, nb_in, nb_out)
        is_tuple = lambda x: isinstance(x, tuple)

        def _pick_or_pass(t, i):
            # For real 4-tuples return element; for masked (0-tuple) passthrough
            return t[i] if isinstance(t, tuple) and len(t) > i else t

        L     = jax.tree_util.tree_map(lambda t: _pick_or_pass(t, 0), stacked, is_leaf=is_tuple)
        R     = jax.tree_util.tree_map(lambda t: _pick_or_pass(t, 1), stacked, is_leaf=is_tuple)
        L_inv = jax.tree_util.tree_map(lambda t: _pick_or_pass(t, 2), stacked, is_leaf=is_tuple)
        R_inv = jax.tree_util.tree_map(lambda t: _pick_or_pass(t, 3), stacked, is_leaf=is_tuple)

        # Build zeros like params but keep masked leaves as-is
        def _zeros_or_pass(x):
            return jnp.zeros_like(x) if _has_array_attrs(x) else x
        mu_zeros = jax.tree.map(_zeros_or_pass, params)
        nu_zeros = jax.tree.map(_zeros_or_pass, params) if grafting else None
        return ScaleByShampooState(
            count=jnp.zeros([], jnp.int32),
            L=L,
            R=R,
            L_inv=L_inv,
            R_inv=R_inv,
            mu=mu_zeros,
            nu=nu_zeros,
        )

    # ------------------------- update --------------------------------------
    def update_fn(updates, state, params=None):
        del params

        # Shampoo statistics with blocking
        def _update_stats(g, L_i, R_i):
            # Skip masked leaves
            if not (hasattr(g, "shape") and hasattr(g, "dtype")):
                return (L_i, R_i)
            m, n = g.shape
            left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
            right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
            rs, cs = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            row_bs = (block_size if (block_size > 0 and len(rs) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cs) > 1) else n)
            need_L = L_i is not None
            need_R = R_i is not None
            if not (need_L or need_R):
                return (None, None)
            g_stk = _stack_blocks(g, rs, cs, row_bs, col_bs)
            g_t = jnp.swapaxes(g_stk, -1, -2)
            L_new = (b2 * L_i + (1.0 - b2) * (g_stk @ g_t)) if need_L else None
            R_new = (b2 * R_i + (1.0 - b2) * (g_t @ g_stk)) if need_R else None
            return L_new, R_new
        LR_new = jax.tree.map(_update_stats, updates, state.L, state.R)
        is_pair = lambda x: isinstance(x, tuple) and len(x) == 2
        L = jax.tree_util.tree_map(lambda t: t[0] if isinstance(t, tuple) and len(t) > 0 else t, LR_new, is_leaf=is_pair)
        R = jax.tree_util.tree_map(lambda t: t[1] if isinstance(t, tuple) and len(t) > 1 else t, LR_new, is_leaf=is_pair)

        # Bias‑corrected moments
        should_recompute = (state.count % freq) == 0
        count_inc = numerics.safe_increment(state.count)
        # Bias correction that skips masked leaves
        def _bias_correct(moment):
            bc = 1 - (b2 ** count_inc)
            def _f(t):
                return t / bc.astype(t.dtype) if hasattr(t, "dtype") else t
            return jax.tree.map(_f, moment)
        L_hat = _bias_correct(L)
        R_hat = _bias_correct(R)

        # -- lazily recompute inverse roots every `freq` steps ----------------
        def _leaf_inv(mat, din_i, dout_i, Ni, p, nb_in_i, nb_out_i):
            if mat is None:
                return None
            if not (hasattr(mat, "shape") and hasattr(mat, "dtype")):
                return mat
            Bi = int(mat.shape[0])
            if scale_eps and not rel_eps:
                eps_leaf = matrix_eps * (din_i / dout_i) / (Ni ** 2) / (nb_in_i * nb_out_i)
            else:
                eps_leaf = matrix_eps
            eps_vec = jnp.repeat(jnp.asarray(eps_leaf, dtype=mat.dtype)[None], Bi, axis=0)
            kernel = (lambda M, e: _inv_root_eigh(M, p, e)) if eigh else (lambda M, e: _inv_root(M, p, e, rel_eps))
            return jax.vmap(kernel)(mat, eps_vec)

        def_tree_N = N

        def _recompute(_):
            is_none = lambda x: x is None
            Linv = jax.tree.map(
                lambda M, din_i, dout_i, Ni, nbi, nbo: None if M is None else _leaf_inv(M, din_i, dout_i, Ni, kl, nbi, nbo),
                L_hat, din, dout, def_tree_N, nb_in, nb_out,
                is_leaf=is_none,
            )
            Rinv = jax.tree.map(
                lambda M, din_i, dout_i, Ni, nbi, nbo: None if M is None else _leaf_inv(M, din_i, dout_i, Ni, kr, nbi, nbo),
                R_hat, din, dout, def_tree_N, nb_in, nb_out,
                is_leaf=is_none,
            )
            return Linv, Rinv

        L_inv, R_inv = jax.lax.cond(
            should_recompute,
            _recompute,                              # true branch
            lambda _: (state.L_inv, state.R_inv),    # false branch
            operand=None
        )
        # First moment update that skips masked leaves
        def _update_moment(upd, mom):
            if hasattr(upd, "dtype"):
                return (1 - b1) * upd + b1 * mom
            return mom
        mu = jax.tree.map(_update_moment, updates, state.mu)
        def _bias_correct_mu(moment):
            bc = 1 - (b1 ** count_inc)
            def _f(t):
                return t / bc.astype(t.dtype) if hasattr(t, "dtype") else t
            return jax.tree.map(_f, moment)
        mu_hat = _bias_correct_mu(mu)
        # Apply preconditioner in blocks then merge
        def _apply_precond(mu_i, Linv_i, Rinv_i, w):
            if not (hasattr(mu_i, "shape") and hasattr(mu_i, "dtype")):
                return mu_i
            m, n = mu_i.shape
            left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
            right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
            rs, cs = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            row_bs = (block_size if (block_size > 0 and len(rs) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cs) > 1) else n)
            mu_stk = _stack_blocks(mu_i, rs, cs, row_bs, col_bs)
            if Linv_i is None and Rinv_i is None:
                upd_stk = mu_stk
            elif Linv_i is None:
                upd_stk = mu_stk @ Rinv_i
            elif Rinv_i is None:
                upd_stk = Linv_i @ mu_stk
            else:
                upd_stk = Linv_i @ mu_stk @ Rinv_i
            return _merge_stacked_blocks(upd_stk, m, n, rs, cs, row_bs, col_bs)
        shampoo_updates = jax.tree.map(_apply_precond, mu_hat, L_inv, R_inv, updates)

        # Optional AdaGrad/Adam‑style grafting --------------------------------
        if grafting:
            nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
            nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
            if scale_eps:
                is_none = lambda x: x is None
                adam_updates = jax.tree.map(
                    lambda m, v, din, dout, n: (
                        m if not (hasattr(m, "dtype") and hasattr(v, "dtype"))
                        else m / (jnp.sqrt(v) + adam_eps / dout / n)
                    ),
                    mu_hat, nu_hat, din, dout, def_tree_N,
                    is_leaf=is_none,
                )
                # graft eps ~ sqrt(dout/din) / lr_shampoo; depth scaling N^(2(kl+kr)-1)
                kl_active = jax.tree.map(
                    lambda x: 0 if (hasattr(x, "shape") and x.shape[0] > max_precond_dim) else kl,
                    updates,
                )
                kr_active = jax.tree.map(
                    lambda x: 0 if (hasattr(x, "shape") and x.shape[1] > max_precond_dim) else kr,
                    updates,
                )
                E_active = jax.tree.map(lambda x, y: x + y, kl_active, kr_active)
                # N ** (1 - 2*E) * (dout / din) ** (1 - E) * (nb_in * nb_out) ** -E
                shampoo_lr = jax.tree.map(
                    lambda n, E, din, dout, nbi, nbo: n ** (1 - 2*E) * (dout / din) ** (1 - E) * (nbi * nbo) ** -E,
                    def_tree_N, E_active, din, dout, nb_in, nb_out,
                    is_leaf=is_none,
                )
                final_updates = jax.tree.map(
                    lambda a, s, slr, din, dout: (
                        a if not (hasattr(a, "dtype") and hasattr(s, "dtype"))
                        else jnp.linalg.norm(a) / (jnp.linalg.norm(s) + graft_eps * ((dout / din) ** 0.5) / slr) * s
                    ),
                    adam_updates, shampoo_updates, shampoo_lr, din, dout,
                    is_leaf=is_none,
                )
            else:
                adam_updates = jax.tree.map(lambda m, v: m / (jnp.sqrt(v) + adam_eps),
                                       mu_hat, nu_hat)
                final_updates = jax.tree.map(
                    lambda a, s: jnp.linalg.norm(a) / (jnp.linalg.norm(s) + graft_eps) * s,
                    adam_updates, shampoo_updates
                )
        else:
            nu = state.nu
            final_updates = shampoo_updates

        new_state = ScaleByShampooState(count_inc, L, R, L_inv, R_inv, mu, nu)
        return final_updates, new_state if not static else state

    return base.GradientTransformation(init_fn, update_fn)
