from typing import NamedTuple, Optional, List, Tuple
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from optax._src import numerics
from optax import GradientTransformation, Updates

# v2-compatible utility
from utils import prune_tree


class SOAPState(NamedTuple):
    count: jnp.ndarray          # step counter (int32)
    mu: Updates                 # first moment (same shapes as grads)
    nu: Updates                 # second moment in *projected* basis (same shapes as grads)
    L: Updates                  # left Gram EMA per leaf: g @ g.T
    R: Updates                  # right Gram EMA per leaf: g.T @ g
    LQ: Updates                 # left basis (columns are eigenvectors)
    RQ: Updates                 # right basis
    din: object                 # per-leaf D_in scaling (float scalar leaves)
    dout: object                # per-leaf D_out scaling (float scalar leaves)
    lam: object                 # per-leaf max eigenvalue for denom eps
    nb_in: object               # per-leaf normalized number of input blocks
    nb_out: object              # per-leaf normalized number of output blocks


def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    adam_eps: float = 1e-8,
    matrix_eps: float = 1e-8,
    freq: int = 10,
    scale_eps: bool = False,      # controls BOTH denom and eigen regularization scaling
    base_shapes=None,             # tree of reference shapes (as in v2)
    B=None,                       # unused (API parity)
    N=None,                       # tree of per-leaf batch sizes; if None, defaults to 1.0
    eigh: bool = False,           # if True, keep using EIGH after warmup; else switch to QR
    eigh_warmup_steps: int = 50,        # steps to force EIGH before switching to QR when eigh=False
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    atan2: bool = False,         # if True, use arctan2(m, sqrt(v)) instead of Adam-style ratio
    rel_eps: bool = False,       # if True, use relative denom eps: adam_eps * max_eig per leaf
    block_size: int = 0,         # if > 0, partition each 2D leaf into blocks of this size
    nb_in=None,                  # tree of normalized input block counts (from train.py)
    nb_out=None,                 # tree of normalized output block counts (from train.py)
    max_precond_dim: int = 0,    # if > 0, treat axes larger than this as identity (one-sided)
    bf16_momentum: bool = False,  # if True, keep first moment (mu) state in bf16
) -> GradientTransformation:
    """
    Simplified v1 SOAP for 2D weights (no blocking/merging/stacked layers/sharding),
    with din/dout/N scaling in BOTH the denominator and the eigen regularization used for bases.

    Step timing matches v1 in this regime:
      • Step 1: initialize L/R from grads (EMA from zero), bases from EIG with diag shift, return zero updates.
      • Steps >1: update uses *stale* bases; if (count % freq == 0) recompute bases AFTER the update
        via a QR power step on (P + eps_eig I), and reorder ν to align with new bases.
      • Bias correction applied outside the ratio: update *= sqrt(1 - b2^t) / (1 - b1^t)
    """

    def _assert_2d_tree(params):
        def _check(p):
            if p.ndim != 2:
                raise ValueError(
                    f"Simplified v1 expects all leaves to be 2D, got shape {p.shape}."
                )
        jtu.tree_map(_check, params)

    def _zeros_like_params(params):
        return otu.tree_zeros_like(params)

    def _matmul(a, b):
        return jnp.matmul(a, b, precision=precision)

    def _eig_eps_for_leaf(eps_base, din, dout, n, nb_in, nb_out, dtype):
        base = jnp.asarray(eps_base, dtype=dtype)
        if scale_eps:
            return base * (din / dout) / (n ** 2) / (nb_in * nb_out)
        return base

    def _scale_with_eps(m, v, din, dout, n, nb_in, nb_out, lam_max, left_id: bool, right_id: bool):
        denom = jnp.sqrt(v)
        if rel_eps:
            eps = adam_eps * jnp.maximum(lam_max ** 0.5, 1e-8)
        else:
            if scale_eps:
                # left is din, right is dout
                # omitting depth scaling for one-sided since emb params are not in the bulk anyways
                if right_id: # in only
                    eps = adam_eps * (din / nb_in) ** 0.5 / dout
                elif left_id: # out only
                    eps = adam_eps * (dout * nb_out) ** -0.5
                else:
                    eps = adam_eps * (jnp.sqrt(din / dout / (nb_in * nb_out)) / n)
            else:
                eps = adam_eps
        denom = denom + eps
        return m / denom

    def _scale_with_atan2(m, v):
        # Adam-atan2 variant: no eps, direct arctan2 on elementwise (m, sqrt(v))
        return jnp.arctan2(m, jnp.sqrt(v))

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
        """Pad each block to (row_bsz, col_bsz) and stack along axis 0 -> [B, row_bsz, col_bsz]."""
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
        """Merge [B, bsz, bsz] back into (m, n), cropping padding per block."""
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

    def _qr_recompute_with_reg_batched(P: jnp.ndarray, Q: jnp.ndarray, nu_b: jnp.ndarray,
                                       eps_reg: jnp.ndarray, axis: int):
        """
        Batched QR recompute with regularization.
        Inputs shapes: P,Q,nu_b: [B, d, d]; eps_reg: scalar (broadcast).
        Returns: (Q_new [B,d,d], nu_reordered [B,d,d], lam_max_per_block [B])
        """
        B, d, _ = P.shape
        eye = jnp.eye(d, dtype=P.dtype)
        P_reg = P + eps_reg * eye  # broadcast over batch

        def per_block(Pi, Qi, nui):
            est = jnp.diag(_matmul(_matmul(Qi.T, Pi), Qi))
            sort_idx = jnp.argsort(est, descending=True)
            Qi_sorted = Qi[:, sort_idx]
            nui_sorted = jnp.take(nui, sort_idx, axis=axis)
            Q_new, _ = jnp.linalg.qr(_matmul(Pi, Qi_sorted))
            lam = jnp.max(est)
            return Q_new, nui_sorted, lam

        Q_new, nu_sorted, lam = jax.vmap(per_block, in_axes=(0, 0, 0))(P_reg, Q, nu_b)
        return Q_new, nu_sorted, lam

    def _qr_recompute_k_steps(P: jnp.ndarray, Q: jnp.ndarray, nu_b: jnp.ndarray,
                              eps_reg: jnp.ndarray, axis: int, k: int):
        """Run k QR power steps with regularization; returns (Q_k, nu_reordered_k, lam_k)."""
        B = P.shape[0]
        zero_lam = jnp.zeros((B,), dtype=P.dtype)

        def body(i, carry):
            Qc, nuc, _lamc = carry
            Qn, nu_n, lam = _qr_recompute_with_reg_batched(P, Qc, nuc, eps_reg, axis)
            return (Qn, nu_n, lam)

        return jax.lax.fori_loop(0, k, body, (Q, nu_b, zero_lam))

    # Removed EIGH path entirely: QR-only basis updates.

    def init_fn(params: Updates) -> SOAPState:
        _assert_2d_tree(params)

        mu = _zeros_like_params(params)
        if bf16_momentum:
            mu = jtu.tree_map(lambda x: x.astype(jnp.bfloat16), mu)
        nu = _zeros_like_params(params)

        def _init_LR_Q_nu(p):
            m, n = p.shape
            left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
            right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
            rslices, cslices = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            B = len(rslices) * len(cslices)
            # choose per-axis block sizes to avoid padding if no splitting
            row_bs = (block_size if (block_size > 0 and len(rslices) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cslices) > 1) else n)

            L  = None if left_id  else jnp.zeros((B, row_bs, row_bs), dtype=p.dtype)
            R  = None if right_id else jnp.zeros((B, col_bs, col_bs), dtype=p.dtype)
            LQ = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=p.dtype), (B, row_bs, row_bs))
            RQ = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=p.dtype), (B, col_bs, col_bs))
            nu = jnp.zeros((B, row_bs, col_bs), dtype=p.dtype)
            return L, R, LQ, RQ, nu

        # Initialize per-leaf structures, optionally blocked
        L  = jtu.tree_map(lambda p: _init_LR_Q_nu(p)[0], params)
        R  = jtu.tree_map(lambda p: _init_LR_Q_nu(p)[1], params)
        LQ = jtu.tree_map(lambda p: _init_LR_Q_nu(p)[2], params)
        RQ = jtu.tree_map(lambda p: _init_LR_Q_nu(p)[3], params)
        nu = jtu.tree_map(lambda p: _init_LR_Q_nu(p)[4], params)

        # -- din/dout from base_shapes (v2-compatible) --
        shapes = jax.tree.map(lambda p: jnp.array(p.shape), params)
        pruned_base_shapes = prune_tree(base_shapes, shapes) if base_shapes is not None else shapes
        din = jax.tree.map(lambda s, b: (s[0] / b[0]).astype(jnp.float32), shapes, pruned_base_shapes)
        dout = jax.tree.map(lambda s, b: (s[1] / b[1]).astype(jnp.float32), shapes, pruned_base_shapes)
        # -- normalized block counts from train.py (if provided) --
        pruned_nb_in  = prune_tree(nb_in, params)  if nb_in  is not None else jtu.tree_map(lambda p: jnp.asarray(1.0, dtype=p.dtype), params)
        pruned_nb_out = prune_tree(nb_out, params) if nb_out is not None else jtu.tree_map(lambda p: jnp.asarray(1.0, dtype=p.dtype), params)

        lam = jtu.tree_map(lambda p: jnp.asarray(0.0, dtype=p.dtype), params)
        return SOAPState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            nu=nu,
            L=L,
            R=R,
            LQ=LQ,
            RQ=RQ,
            din=din,
            dout=dout,
            lam=lam,
            nb_in=pruned_nb_in,
            nb_out=pruned_nb_out,
        )

    def update_fn(updates: Updates, state: SOAPState, params: Optional[Updates] = None):
        del params  # unused

        # Per-leaf N (default = 1.0)
        pruned_N = prune_tree(N, updates) if (N is not None) else jtu.tree_map(
            lambda g: jnp.asarray(1.0, dtype=g.dtype), updates
        )

        # Increment count (v1 timing)
        count_inc = numerics.safe_increment(state.count)

        # Single update path: update preconditioners/bases before projection; supports blocking
        def _update_step():
            # Recompute every step during warmup; otherwise at provided frequency.
            in_warmup = count_inc <= eigh_warmup_steps
            do_recompute = jnp.logical_or(in_warmup, jnp.logical_or(count_inc == 1, (count_inc % freq) == 0))

            # First moment on raw grads
            mu_new_tree = otu.tree_update_moment(updates, state.mu, b1, 1)
            if bf16_momentum:
                mu_new_tree = jtu.tree_map(lambda x: x.astype(jnp.bfloat16), mu_new_tree)

            # Precompute stacked L_new and R_new for all leaves
            def _build_LR(g, L_i, R_i):
                m, n = g.shape
                left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
                right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
                rslices, cslices = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
                row_bs = (block_size if (block_size > 0 and len(rslices) > 1) else m)
                col_bs = (block_size if (block_size > 0 and len(cslices) > 1) else n)
                g_stk = _stack_blocks(g, rslices, cslices, row_bs, col_bs)
                g_t = jnp.swapaxes(g_stk, -1, -2)
                left_id = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
                right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
                L_new = L_i if left_id else (b2 * L_i + (1.0 - b2) * _matmul(g_stk, g_t))
                R_new = R_i if right_id else (b2 * R_i + (1.0 - b2) * _matmul(g_t, g_stk))
                return (L_new, R_new)

            LR_tree = jtu.tree_map(_build_LR, updates, state.L, state.R)
            is_pair = lambda x: isinstance(x, tuple)
            L_new_tree = jtu.tree_map(lambda t: t[0], LR_tree, is_leaf=is_pair)
            R_new_tree = jtu.tree_map(lambda t: t[1], LR_tree, is_leaf=is_pair)

            # Build eps trees matching L_new_tree/R_new_tree
            L_eps_tree = jtu.tree_map(
                lambda P, din, dout, n, nb_in, nb_out, g: (
                    None if P is None else _eig_eps_for_leaf(matrix_eps, din, dout, n, nb_in, nb_out, g.dtype)
                ),
                L_new_tree, state.din, state.dout, pruned_N, state.nb_in, state.nb_out, updates,
                is_leaf=lambda x: x is None,
            )
            R_eps_tree = jtu.tree_map(
                lambda P, din, dout, n, nb_in, nb_out, g: (
                    None if P is None else _eig_eps_for_leaf(matrix_eps, din, dout, n, nb_in, nb_out, g.dtype)
                ),
                R_new_tree, state.din, state.dout, pruned_N, state.nb_in, state.nb_out, updates,
                is_leaf=lambda x: x is None,
            )

            # No EIGH branch; QR-only path below handles recompute.

            def _leaf_update(g, mu_new, L_new, R_new, LQ_i, RQ_i, nu_i, din_i, dout_i, N_i, nb_in_i, nb_out_i, lam_i):
                m, n = g.shape
                left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
                right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)
                rslices, cslices = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
                # Always use stacked path with leading dim B (1 when unblocked)
                # If no splitting on an axis, don't pad beyond the true dim to preserve equivalence.
                row_bs = (block_size if (block_size > 0 and len(rslices) > 1) else m)
                col_bs = (block_size if (block_size > 0 and len(cslices) > 1) else n)

                g_stk = _stack_blocks(g, rslices, cslices, row_bs, col_bs)
                mu_stk = _stack_blocks(mu_new, rslices, cslices, row_bs, col_bs)
                # State tensors are already stacked with leading B in init
                LQ_stk, RQ_stk, nu_stk = LQ_i, RQ_i, nu_i

                # Precompute eps
                L_eps = _eig_eps_for_leaf(matrix_eps, din_i, dout_i, N_i, nb_in_i, nb_out_i, g_stk.dtype)
                R_eps = _eig_eps_for_leaf(matrix_eps, din_i, dout_i, N_i, nb_in_i, nb_out_i, g_stk.dtype)

                def _recompute(_):
                    left_id  = (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim)
                    right_id = (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim)

                    def _do_warmup(_):
                        if left_id:
                            LQn, nu_mid, lamL = (None, nu_stk, jnp.asarray(0.0, dtype=g_stk.dtype))
                        else:
                            baseLQ = LQ_stk if LQ_stk is not None else jnp.broadcast_to(jnp.eye(row_bs, dtype=g_stk.dtype), (L_new.shape[0], row_bs, row_bs))
                            LQn, nu_mid, lamL = _qr_recompute_k_steps(L_new, baseLQ, nu_stk, L_eps, axis=0, k=10)
                        if right_id:
                            RQn, nu_prev2, lamR = (None, nu_mid, jnp.asarray(0.0, dtype=g_stk.dtype))
                        else:
                            baseRQ = RQ_stk if RQ_stk is not None else jnp.broadcast_to(jnp.eye(col_bs, dtype=g_stk.dtype), (R_new.shape[0], col_bs, col_bs))
                            RQn, nu_prev2, lamR = _qr_recompute_k_steps(R_new, baseRQ, nu_mid, R_eps, axis=1, k=10)
                        lamv = jnp.maximum(jnp.max(lamL), jnp.max(lamR))
                        return (LQn, RQn, nu_prev2, lamv)

                    def _do_regular(_):
                        if left_id:
                            LQn, nu_mid, lamL = (None, nu_stk, jnp.asarray(0.0, dtype=g_stk.dtype))
                        else:
                            baseLQ = LQ_stk if LQ_stk is not None else jnp.broadcast_to(jnp.eye(row_bs, dtype=g_stk.dtype), (L_new.shape[0], row_bs, row_bs))
                            LQn, nu_mid, lamL = _qr_recompute_with_reg_batched(L_new, baseLQ, nu_stk, L_eps, axis=0)
                        if right_id:
                            RQn, nu_prev2, lamR = (None, nu_mid, jnp.asarray(0.0, dtype=g_stk.dtype))
                        else:
                            baseRQ = RQ_stk if RQ_stk is not None else jnp.broadcast_to(jnp.eye(col_bs, dtype=g_stk.dtype), (R_new.shape[0], col_bs, col_bs))
                            RQn, nu_prev2, lamR = _qr_recompute_with_reg_batched(R_new, baseRQ, nu_mid, R_eps, axis=1)
                        lamv = jnp.maximum(jnp.max(lamL), jnp.max(lamR))
                        return (LQn, RQn, nu_prev2, lamv)

                    return jax.lax.cond(count_inc <= eigh_warmup_steps, _do_warmup, _do_regular, operand=None)

                def _no_recompute(_):
                    return (LQ_stk, RQ_stk, nu_stk, lam_i)

                LQ_new, RQ_new, nu_prev, lam_prev = jax.lax.cond(do_recompute, _recompute, _no_recompute, operand=None)

                # Project with only active sides; avoid materializing identities
                if left_id and right_id:
                    g_proj = g_stk
                elif left_id:
                    g_proj = _matmul(g_stk, RQ_new)
                elif right_id:
                    g_proj = _matmul(jnp.swapaxes(LQ_new, -1, -2), g_stk)
                else:
                    g_proj = _matmul(_matmul(jnp.swapaxes(LQ_new, -1, -2), g_stk), RQ_new)
                nu_new = b2 * nu_prev + (1.0 - b2) * (g_proj ** 2)
                if left_id and right_id:
                    m_proj = mu_stk
                elif left_id:
                    m_proj = _matmul(mu_stk, RQ_new)
                elif right_id:
                    m_proj = _matmul(jnp.swapaxes(LQ_new, -1, -2), mu_stk)
                else:
                    m_proj = _matmul(_matmul(jnp.swapaxes(LQ_new, -1, -2), mu_stk), RQ_new)
                upd_proj = _scale_with_atan2(m_proj, nu_new) if atan2 else _scale_with_eps(m_proj, nu_new, din_i, dout_i, N_i, nb_in_i, nb_out_i, lam_prev, left_id, right_id)
                if left_id and right_id:
                    upd_stk = upd_proj
                elif left_id:
                    upd_stk = _matmul(upd_proj, jnp.swapaxes(RQ_new, -1, -2))
                elif right_id:
                    upd_stk = _matmul(LQ_new, upd_proj)
                else:
                    upd_stk = _matmul(_matmul(LQ_new, upd_proj), jnp.swapaxes(RQ_new, -1, -2))
                upd_leaf = _merge_stacked_blocks(upd_stk, m, n, rslices, cslices, row_bs, col_bs)

                # State is always stacked; keep shapes as-is
                L_new_rest  = L_new
                R_new_rest  = R_new
                LQ_new_rest = LQ_new
                RQ_new_rest = RQ_new
                nu_new_rest = nu_new

                return (upd_leaf, nu_new_rest, L_new_rest, R_new_rest, LQ_new_rest, RQ_new_rest, lam_prev)

            results = jtu.tree_map(
                _leaf_update,
                updates, mu_new_tree, L_new_tree, R_new_tree, state.LQ, state.RQ, state.nu,
                state.din, state.dout, pruned_N, state.nb_in, state.nb_out, state.lam,
            )

            is_tuple = lambda x: isinstance(x, tuple)
            upd_tree   = jtu.tree_map(lambda t: t[0], results, is_leaf=is_tuple)
            nu_new_tr  = jtu.tree_map(lambda t: t[1], results, is_leaf=is_tuple)
            L_new_tr   = jtu.tree_map(lambda t: t[2], results, is_leaf=is_tuple)
            R_new_tr   = jtu.tree_map(lambda t: t[3], results, is_leaf=is_tuple)
            LQ_new_tr  = jtu.tree_map(lambda t: t[4], results, is_leaf=is_tuple)
            RQ_new_tr  = jtu.tree_map(lambda t: t[5], results, is_leaf=is_tuple)
            lam_new_tr = jtu.tree_map(lambda t: t[6], results, is_leaf=is_tuple)

            bc1 = 1.0 - (b1 ** count_inc)
            bc2 = 1.0 - (b2 ** count_inc)
            upd_tree = jtu.tree_map(lambda x: x * (jnp.sqrt(bc2) / bc1), upd_tree)

            new_state = SOAPState(
                count=count_inc,
                mu=mu_new_tree,
                nu=nu_new_tr,
                L=L_new_tr,
                R=R_new_tr,
                LQ=LQ_new_tr,
                RQ=RQ_new_tr,
                din=state.din,
                dout=state.dout,
                lam=lam_new_tr,
                nb_in=state.nb_in,
                nb_out=state.nb_out,
            )
            return upd_tree, new_state

        return _update_step()

    return optax.GradientTransformation(init_fn, update_fn)