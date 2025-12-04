from typing import NamedTuple, Optional, List, Tuple, Dict, Any
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
    nu: Updates                 # second moment in projected basis (same shapes as grads)
    L: Updates                  # left Gram EMA per leaf: g @ g.T (stacked over blocks)
    R: Updates                  # right Gram EMA per leaf: g.T @ g (stacked over blocks)
    LQ: Updates                 # left basis (stacked over blocks)
    RQ: Updates                 # right basis (stacked over blocks)
    din: object                 # per-leaf D_in scaling (float scalar leaves)
    dout: object                # per-leaf D_out scaling (float scalar leaves)
    # lam: object                 # per-leaf max eigenvalue for denom eps
    nb_in: object               # per-leaf normalized number of input blocks
    nb_out: object              # per-leaf normalized number of output blocks


def scale_by_soap_bucketed(
    b1: float = 0.95,
    b2: float = 0.95,
    adam_eps: float = 1e-8,
    matrix_eps: float = 1e-8,
    freq: int = 10,
    scale_eps: bool = False,      # controls BOTH denom and eigen regularization scaling
    base_shapes=None,             # tree of reference shapes (as in v2)
    B=None,                       # unused (API parity)
    N=None,                       # tree of per-leaf batch sizes; if None, defaults to 1.0
    eigh: bool = False,           # ignored (QR-only), kept for API compatibility
    eigh_warmup_steps: int = 50,  # steps to force QR-k steps during warmup
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    rel_eps: bool = False,       # if True, use relative denom eps: adam_eps * max_eig per leaf
    block_size: int = 0,         # if > 0, partition each 2D leaf into blocks of this size
    nb_in=None,                  # tree of normalized input block counts (from train.py)
    nb_out=None,                 # tree of normalized output block counts (from train.py)
    max_precond_dim: int = 1000000,     # if > 0, treat axes larger than this as identity (one-sided)
    skip_precond_dim: int = 0,     # if > 0, treat axes smaller than this as identity (one-sided)
    bf16_momentum: bool = False,  # if True, keep first moment (mu) state in bf16
    bucket_by_full_block_shape: bool = True,  # if True, bucket by both row/col sizes
    atan2: bool = False,         # if True, use Adam-atan2 variant: no eps, direct arctan2 on elementwise (m, sqrt(v))
    param_names=None,            # tree of string names aligned with params (optional; used for debug recorder)
    readout_eps_mult: float = 1.0, # if > 0, multiply eps by this factor for readout layers
) -> GradientTransformation:
    """
    SOAP optimizer with QR recompute bucketed across leaves.

    This mirrors optax_optim._soap.scale_by_soap but hoists the QR recompute
    out of the per-leaf map and batches it across all leaves with compatible
    block sizes, reducing compile graph size.
    If `bucket_by_full_block_shape` is True we additionally separate buckets by
    their column dimension so that large readout blocks do not force smaller
    leaves to be padded up to the largest width.
    """

    def _assert_2d_tree(params):
        def _check(p):
            if p.ndim != 2:
                raise ValueError(
                    f"SOAP expects all leaves to be 2D, got shape {p.shape}."
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

    def _scale_with_atan2(m, v):
        # Adam-atan2 variant: no eps, direct arctan2 on elementwise (m, sqrt(v))
        return jnp.arctan2(m, jnp.sqrt(v))

    def _scale_with_eps(m, v, din, dout, n, nb_in, nb_out, left_id: bool, right_id: bool):
        denom = jnp.sqrt(v)
        if rel_eps:
            eps = jnp.maximum(denom[:, 0, 0] * adam_eps, 1e-10)
        else:
            if scale_eps:
                # left is din, right is dout
                if right_id:  # in only
                    eps = adam_eps * (din / nb_in) ** 0.5 / dout
                elif left_id:  # out only
                    eps = adam_eps * (dout * nb_out) ** -0.5
                else:
                    eps = adam_eps * (jnp.sqrt(din / dout / (nb_in * nb_out)) / n)
            else:
                eps = adam_eps
        eps = jax.lax.cond(dout == 1, lambda _: eps * readout_eps_mult, lambda _: eps, operand=None)
        denom = denom + eps
        return m / denom

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

    # ---- batched QR helpers (support per-block eps) ----
    def _qr_recompute_with_reg_batched(P: jnp.ndarray, Q: jnp.ndarray, nu_b: jnp.ndarray,
                                       eps_reg: jnp.ndarray, axis: int):
        """
        Batched QR recompute with regularization.
        Inputs shapes: P,Q: [B, d, d]; nu_b: [B, d, k]; eps_reg: [B] or scalar.
        Returns: (Q_new [B,d,d], nu_reordered [B,d,k], lam_max_per_block [B])
        """
        Bsz, d, _ = P.shape
        eye = jnp.eye(d, dtype=P.dtype)
        if eps_reg.ndim == 0:
            eps_b = jnp.broadcast_to(eps_reg, (Bsz,))
        else:
            eps_b = eps_reg
        # P_reg = P + eps_b[:, None, None] * eye  # [B,d,d]

        def per_block(Pi, Qi, nui, eps):
            # Pi = Pi + eps * jnp.eye(d, dtype=P.dtype)
            # est = jnp.diag(_matmul(_matmul(Qi.T, Pi), Qi))

            # est_j = q_j^T P_reg q_j
            Pi_reg = Pi + eps * eye
            PQ = _matmul(Pi_reg, Qi)
            est = jnp.sum(Qi * PQ, axis=0)

            sort_idx = jnp.argsort(est, descending=True)
            # Qi_sorted = Qi[:, sort_idx]
            nui_sorted = jnp.take(nui, sort_idx, axis=axis)
            PQ_sorted  = PQ[:, sort_idx]

            Q_new, _ = jnp.linalg.qr(PQ_sorted)
            lam = jnp.max(est)
            return Q_new, nui_sorted, lam

        # Use lax.map to reduce peak memory vs vmap
        def per_block_tup(t):
            Pi, Qi, nui, eps = t
            return per_block(Pi, Qi, nui, eps)
        Q_new, nu_sorted, lam = jax.vmap(per_block, in_axes=(0, 0, 0, 0))(P, Q, nu_b, eps_b)
        # Q_new, nu_sorted, lam = jax.lax.map(per_block_tup, (P, Q, nu_b, eps_b))
        return Q_new, nu_sorted, lam

    def _qr_recompute_k_steps(P: jnp.ndarray, Q: jnp.ndarray, nu_b: jnp.ndarray,
                              eps_reg: jnp.ndarray, axis: int, k: int):
        """Run k QR power steps with regularization; returns (Q_k, nu_reordered_k, lam_k)."""
        Bsz = P.shape[0]
        zero_lam = jnp.zeros((Bsz,), dtype=P.dtype)

        def body(i, carry):
            Qc, nuc, _lamc = carry
            Qn, nu_n, lam = _qr_recompute_with_reg_batched(P, Qc, nuc, eps_reg, axis)
            return (Qn, nu_n, lam)

        return jax.lax.fori_loop(0, k, body, (Q, nu_b, zero_lam))

    # ---- batched EIGH helper (for warmup if eigh=True) ----
    def _eigh_recompute_with_reg_batched(P: jnp.ndarray, nu_b: jnp.ndarray,
                                         eps_reg: jnp.ndarray, axis: int):
        """
        Batched eigh with regularization. Computes eigenbasis of P_reg = P + eps*I,
        sorts by descending eigenvalue, and reorders nu accordingly along `axis`.
        Inputs:
          P: [B, d, d]
          nu_b: [B, d, k]
          eps_reg: [B] or scalar
          axis: 0 (rows/left) or 1 (cols/right)
        Returns: (Q_new [B,d,d], nu_sorted [B,d,k], lam_max [B])
        """
        Bsz, d, _ = P.shape
        eye = jnp.eye(d, dtype=P.dtype)[None, :, :]
        if eps_reg.ndim == 0:
            eps_b = jnp.broadcast_to(eps_reg, (Bsz,))
        else:
            eps_b = eps_reg
        P_reg = P + eps_b[:, None, None] * eye
        # Batched eigh; returns ascending eigenvalues
        w, V = jnp.linalg.eigh(P_reg)
        # Sort descending: reverse along eigen-dimension
        w_desc = w[..., ::-1]
        V_desc = V[..., :, ::-1]
        # Keep nu as-is in the eigh path (no reindexing)
        nu_sorted = nu_b
        lam = w_desc[..., 0]
        return V_desc, nu_sorted, lam

    def init_fn(params: Updates) -> SOAPState:
        _assert_2d_tree(params)

        mu = _zeros_like_params(params)
        if bf16_momentum:
            mu = jtu.tree_map(lambda x: x.astype(jnp.bfloat16), mu)
        nu = _zeros_like_params(params)

        def _init_LR_Q_nu(p):
            m, n = p.shape
            left_id  = m > max_precond_dim or m == skip_precond_dim
            right_id = n > max_precond_dim or n == skip_precond_dim
            rslices, cslices = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            Bblocks = len(rslices) * len(cslices)
            row_bs = (block_size if (block_size > 0 and len(rslices) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cslices) > 1) else n)

            L  = None if left_id  else jnp.zeros((Bblocks, row_bs, row_bs), dtype=p.dtype)
            R  = None if right_id else jnp.zeros((Bblocks, col_bs, col_bs), dtype=p.dtype)
            LQ = None if left_id  else jnp.broadcast_to(jnp.eye(row_bs, dtype=p.dtype), (Bblocks, row_bs, row_bs))
            RQ = None if right_id else jnp.broadcast_to(jnp.eye(col_bs, dtype=p.dtype), (Bblocks, col_bs, col_bs))
            nu = jnp.zeros((Bblocks, row_bs, col_bs), dtype=p.dtype)
            return L, R, LQ, RQ, nu
        init_tree = jtu.tree_map(_init_LR_Q_nu, params)
        L = jtu.tree_map(lambda p: p[0], init_tree, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 5)
        R = jtu.tree_map(lambda p: p[1], init_tree, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 5)
        LQ = jtu.tree_map(lambda p: p[2], init_tree, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 5)
        RQ = jtu.tree_map(lambda p: p[3], init_tree, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 5)
        nu = jtu.tree_map(lambda p: p[4], init_tree, is_leaf=lambda x: isinstance(x, tuple) and len(x) == 5)

        shapes = jax.tree.map(lambda p: jnp.array(p.shape), params)
        pruned_base_shapes = prune_tree(base_shapes, shapes) if base_shapes is not None else shapes
        din = jax.tree.map(lambda s, b: (s[0] / b[0]).astype(jnp.float32), shapes, pruned_base_shapes)
        dout = jax.tree.map(lambda s, b: (s[1] / b[1]).astype(jnp.float32), shapes, pruned_base_shapes)
        pruned_nb_in  = prune_tree(nb_in, params)  if nb_in  is not None else jtu.tree_map(lambda p: jnp.asarray(1.0, dtype=p.dtype), params)
        pruned_nb_out = prune_tree(nb_out, params) if nb_out is not None else jtu.tree_map(lambda p: jnp.asarray(1.0, dtype=p.dtype), params)

        # lam = jtu.tree_map(lambda p: jnp.asarray(0.0, dtype=p.dtype), params)
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
            # lam=lam,
            nb_in=pruned_nb_in,
            nb_out=pruned_nb_out,
        )

    def update_fn(updates: Updates, state: SOAPState, params: Optional[Updates] = None):
        del params

        pruned_N = prune_tree(N, updates) if (N is not None) else jtu.tree_map(
            lambda g: jnp.asarray(1.0, dtype=g.dtype), updates
        )

        count_inc = numerics.safe_increment(state.count)

    
        in_warmup = count_inc <= eigh_warmup_steps
        do_recompute = jnp.logical_or(in_warmup, jnp.logical_or(count_inc == 1, (count_inc % freq) == 0))
        # do_recompute = False # TODO: debug purpose

        # First moment on raw grads
        mu_new_tree = otu.tree_update_moment(updates, state.mu, b1, 1)
        if bf16_momentum:
            mu_new_tree = jtu.tree_map(lambda x: x.astype(jnp.bfloat16), mu_new_tree)

        # Compute L_new and R_new (EMAs) per leaf; also collect per-leaf meta
        def _build_LR_meta(g, L_i, R_i):
            m, n = g.shape
            left_id  = m > max_precond_dim or m == skip_precond_dim
            right_id = n > max_precond_dim or n == skip_precond_dim
            rslices, cslices = _compute_block_slices_with_gate(m, n, block_size, left_id, right_id)
            row_bs = (block_size if (block_size > 0 and len(rslices) > 1) else m)
            col_bs = (block_size if (block_size > 0 and len(cslices) > 1) else n)
            g_stk = _stack_blocks(g, rslices, cslices, row_bs, col_bs)
            g_t = jnp.swapaxes(g_stk, -1, -2)
            L_new = L_i if left_id else (b2 * L_i + (1.0 - b2) * _matmul(g_stk, g_t))
            R_new = R_i if right_id else (b2 * R_i + (1.0 - b2) * _matmul(g_t, g_stk))
            meta = (m, n, row_bs, col_bs, len(rslices) * len(cslices), left_id, right_id, rslices, cslices)
            return (L_new, R_new, meta)

        LR_meta_tree = jtu.tree_map(_build_LR_meta, updates, state.L, state.R)
        # MaskedNode is also a tuple (zero length), so guard on the expected triple shape.
        is_triple = lambda x: isinstance(x, tuple) and len(x) == 3
        L_new_tree = jtu.tree_map(lambda t: t[0], LR_meta_tree, is_leaf=is_triple)
        R_new_tree = jtu.tree_map(lambda t: t[1], LR_meta_tree, is_leaf=is_triple)
        meta_tree  = jtu.tree_map(lambda t: t[2], LR_meta_tree, is_leaf=is_triple)

        # Eps trees
        L_eps_tree = jtu.tree_map(
            lambda P, din, dout, n, nb_i, nb_o, g: (
                None if P is None else _eig_eps_for_leaf(matrix_eps, din, dout, n, nb_i, nb_o, g.dtype)
            ),
            L_new_tree, state.din, state.dout, pruned_N, state.nb_in, state.nb_out, updates,
            is_leaf=lambda x: x is None,
        )
        R_eps_tree = jtu.tree_map(
            lambda P, din, dout, n, nb_i, nb_o, g: (
                None if P is None else _eig_eps_for_leaf(matrix_eps, din, dout, n, nb_i, nb_o, g.dtype)
            ),
            R_new_tree, state.din, state.dout, pruned_N, state.nb_in, state.nb_out, updates,
            is_leaf=lambda x: x is None,
        )

        # Flatten all per-leaf structures for bucketing
        upd_leaves, treedef = jtu.tree_flatten(updates)
        # Ensure aligned flattening order across all trees
        struct = jtu.tree_structure(updates)
        L_new_leaves = struct.flatten_up_to(L_new_tree)
        R_new_leaves = struct.flatten_up_to(R_new_tree)
        LQ_leaves = struct.flatten_up_to(state.LQ)
        RQ_leaves = struct.flatten_up_to(state.RQ)
        nu_leaves = struct.flatten_up_to(state.nu)
        meta_leaves = struct.flatten_up_to(meta_tree)
        # lam_prev_leaves = struct.flatten_up_to(state.lam)
        L_eps_leaves = struct.flatten_up_to(L_eps_tree)
        R_eps_leaves = struct.flatten_up_to(R_eps_tree)
        mu_leaves = struct.flatten_up_to(mu_new_tree)
        din_leaves = struct.flatten_up_to(state.din)
        dout_leaves = struct.flatten_up_to(state.dout)
        N_leaves = struct.flatten_up_to(pruned_N)
        nb_in_leaves = struct.flatten_up_to(state.nb_in)
        nb_out_leaves = struct.flatten_up_to(state.nb_out)

        n_leaves = len(upd_leaves)

        # Prepare default outputs placeholders (functional style)
        def _recompute_all(_):
            LQ_new_leaves: List[Any] = [None] * n_leaves
            RQ_new_leaves: List[Any] = [None] * n_leaves
            nu_mid_leaves: List[Any] = [None] * n_leaves
            # lamL_max_per_leaf: List[Any] = [jnp.asarray(0.0, dtype=jnp.float32)] * n_leaves
            # lamR_max_per_leaf: List[Any] = [jnp.asarray(0.0, dtype=jnp.float32)] * n_leaves
            # ---- LEFT SIDE BUCKETING ----
            buckets: Dict[tuple, Dict[str, Any]] = {}
            # key: (row_bs, col_bs, dtype) when `bucket_by_full_block_shape` is True
            for i in range(n_leaves):
                P = L_new_leaves[i]
                if P is None:
                    # left identity: keep as-is
                    LQ_new_leaves[i] = None
                    nu_mid_leaves[i] = nu_leaves[i]
                    # lamL_max_per_leaf[i] = jnp.asarray(0.0, dtype=nu_leaves[i].dtype)
                    continue
                # meta
                m, n, row_bs, col_bs, Bblocks, left_id, right_id, _, _ = meta_leaves[i]
                assert not left_id
                # Base Q (may be None in state -> identity)
                baseLQ = LQ_leaves[i]
                if baseLQ is None:
                    baseLQ = jnp.broadcast_to(jnp.eye(row_bs, dtype=P.dtype), (P.shape[0], row_bs, row_bs))
                nu_i = nu_leaves[i]
                eps_i = L_eps_leaves[i]
                if bucket_by_full_block_shape:
                    key = (int(row_bs), int(col_bs), str(P.dtype))
                else:
                    key = (int(row_bs), str(P.dtype))
                if key not in buckets:
                    buckets[key] = {
                        'P_list': [], 'Q_list': [], 'nu_list': [], 'eps_list': [],
                        'indices': [], 'col_max': 0,
                    }
                buckets[key]['P_list'].append(P)
                buckets[key]['Q_list'].append(baseLQ)
                buckets[key]['nu_list'].append(nu_i)
                buckets[key]['eps_list'].append(eps_i)
                buckets[key]['indices'].append(i)
                buckets[key]['col_max'] = max(buckets[key]['col_max'], nu_i.shape[2])

            # process buckets
            for key, b in buckets.items():
                P_cat = jnp.concatenate(b['P_list'], axis=0)
                Q_cat = jnp.concatenate(b['Q_list'], axis=0)
                # pad nu to common col size
                col_max = b['col_max']
                nu_padded = []
                sizes = []
                for nu_i in b['nu_list']:
                    pad = col_max - nu_i.shape[2]
                    if pad:
                        nu_i = jnp.pad(nu_i, ((0, 0), (0, 0), (0, pad)))
                    nu_padded.append(nu_i)
                    sizes.append(nu_i.shape[0])
                nu_cat = jnp.concatenate(nu_padded, axis=0)
                eps_vec = jnp.concatenate([
                    jnp.broadcast_to(jnp.asarray(eps, dtype=P_cat.dtype), (sz,)) for eps, sz in zip(b['eps_list'], sizes)
                ], axis=0)

                def _do_warmup(_):
                    if eigh:
                        return _eigh_recompute_with_reg_batched(P_cat, nu_cat, eps_vec, axis=0)
                    else:
                        return _qr_recompute_k_steps(P_cat, Q_cat, nu_cat, eps_vec, axis=0, k=10)
                def _do_regular(_):
                    return _qr_recompute_with_reg_batched(P_cat, Q_cat, nu_cat, eps_vec, axis=0)
                Q_new_cat, nu_sorted_cat, lam_cat = jax.lax.cond(in_warmup, _do_warmup, _do_regular, operand=None)

                # scatter back
                offset = 0
                for idx, sz in zip(b['indices'], sizes):
                    LQ_new_leaves[idx] = Q_new_cat[offset:offset+sz]
                    # crop back to original col size
                    col_bs = nu_leaves[idx].shape[2]
                    nu_mid_leaves[idx] = nu_sorted_cat[offset:offset+sz, :, :col_bs]
                    # lamL_max_per_leaf[idx] = jnp.max(lam_cat[offset:offset+sz])
                    offset += sz

            # ---- RIGHT SIDE BUCKETING ----
            buckets_r: Dict[tuple, Dict[str, Any]] = {}
            for i in range(n_leaves):
                P = R_new_leaves[i]
                if P is None:
                    RQ_new_leaves[i] = None
                    # lamR_max_per_leaf[i] = jnp.asarray(0.0, dtype=nu_leaves[i].dtype)
                    continue
                # meta
                m, n, row_bs, col_bs, Bblocks, left_id, right_id, _, _ = meta_leaves[i]
                assert not right_id
                baseRQ = RQ_leaves[i]
                if baseRQ is None:
                    baseRQ = jnp.broadcast_to(jnp.eye(col_bs, dtype=P.dtype), (P.shape[0], col_bs, col_bs))
                nu_i = nu_mid_leaves[i]  # already left-reordered
                eps_i = R_eps_leaves[i]
                if bucket_by_full_block_shape:
                    key = (int(row_bs), int(col_bs), str(P.dtype))
                else:
                    key = (int(col_bs), str(P.dtype))
                if key not in buckets_r:
                    buckets_r[key] = {
                        'P_list': [], 'Q_list': [], 'nu_list': [], 'eps_list': [],
                        'indices': [], 'row_max': 0,
                    }
                buckets_r[key]['P_list'].append(P)
                buckets_r[key]['Q_list'].append(baseRQ)
                buckets_r[key]['nu_list'].append(nu_i)
                buckets_r[key]['eps_list'].append(eps_i)
                buckets_r[key]['indices'].append(i)
                buckets_r[key]['row_max'] = max(buckets_r[key]['row_max'], nu_i.shape[1])

            for key, b in buckets_r.items():
                P_cat = jnp.concatenate(b['P_list'], axis=0)
                Q_cat = jnp.concatenate(b['Q_list'], axis=0)
                row_max = b['row_max']
                nu_padded = []
                sizes = []
                for nu_i in b['nu_list']:
                    pad = row_max - nu_i.shape[1]
                    if pad:
                        nu_i = jnp.pad(nu_i, ((0, 0), (0, pad), (0, 0)))
                    nu_padded.append(nu_i)
                    sizes.append(nu_i.shape[0])
                nu_cat = jnp.concatenate(nu_padded, axis=0)
                eps_vec = jnp.concatenate([
                    jnp.broadcast_to(jnp.asarray(eps, dtype=P_cat.dtype), (sz,)) for eps, sz in zip(b['eps_list'], sizes)
                ], axis=0)

                def _do_warmup_r(_):
                    if eigh:
                        return _eigh_recompute_with_reg_batched(P_cat, nu_cat, eps_vec, axis=1)
                    else:
                        return _qr_recompute_k_steps(P_cat, Q_cat, nu_cat, eps_vec, axis=1, k=10)
                def _do_regular_r(_):
                    return _qr_recompute_with_reg_batched(P_cat, Q_cat, nu_cat, eps_vec, axis=1)
                Q_new_cat, nu_sorted_cat, lam_cat = jax.lax.cond(in_warmup, _do_warmup_r, _do_regular_r, operand=None)

                offset = 0
                for idx, sz in zip(b['indices'], sizes):
                    RQ_new_leaves[idx] = Q_new_cat[offset:offset+sz]
                    row_bs = nu_leaves[idx].shape[1]
                    nu_mid = nu_sorted_cat[offset:offset+sz, :row_bs, :]
                    nu_mid_leaves[idx] = nu_mid  # final nu_prev (reordered)
                    # lamR_max_per_leaf[idx] = jnp.max(lam_cat[offset:offset+sz])
                    offset += sz

            return (LQ_new_leaves, RQ_new_leaves, nu_mid_leaves)

        def _no_recompute_all(_):
            # carry over bases and nu
            LQ_new_leaves = list(LQ_leaves)
            RQ_new_leaves = list(RQ_leaves)
            nu_mid_leaves = list(nu_leaves)
            # lamL_max_per_leaf = [jnp.asarray(0.0, dtype=nu_leaves[i].dtype) for i in range(n_leaves)]
            # lamR_max_per_leaf = [jnp.asarray(0.0, dtype=nu_leaves[i].dtype) for i in range(n_leaves)]
            return (LQ_new_leaves, RQ_new_leaves, nu_mid_leaves)

        # Hoisted cond over recompute
        LQ_new_leaves, RQ_new_leaves, nu_mid_leaves, = jax.lax.cond(
            do_recompute, _recompute_all, _no_recompute_all, operand=None
        )

        # Now compute per-leaf projection, nu update, and final updates
        upd_leaves_out: List[Any] = [None] * n_leaves
        # L_new_leaves_out: List[Any] = [None] * n_leaves
        # R_new_leaves_out: List[Any] = [None] * n_leaves
        LQ_new_leaves_out: List[Any] = [None] * n_leaves
        RQ_new_leaves_out: List[Any] = [None] * n_leaves
        nu_new_leaves_out: List[Any] = [None] * n_leaves
        lam_new_leaves_out: List[Any] = [None] * n_leaves

        for i in range(n_leaves):
            g = upd_leaves[i]
            mu_i = mu_leaves[i]
            # L_new_i = L_new_leaves[i]
            # R_new_i = R_new_leaves[i]
            LQ_i = LQ_new_leaves[i]
            RQ_i = RQ_new_leaves[i]
            nu_prev = nu_mid_leaves[i]
            din_i = din_leaves[i]
            dout_i = dout_leaves[i]
            N_i = N_leaves[i]
            nb_in_i = nb_in_leaves[i]
            nb_out_i = nb_out_leaves[i]
            # lam_prev_old = lam_prev_leaves[i]

            # meta
            m, n, row_bs, col_bs, Bblocks, left_id, right_id, rslices, cslices = meta_leaves[i]
            # stack grads and mu for projection
            g_stk = _stack_blocks(g, rslices, cslices, row_bs, col_bs)
            mu_stk = _stack_blocks(mu_i, rslices, cslices, row_bs, col_bs)

            # Compute lam per leaf
            # lamL = lamL_max_per_leaf[i]
            # lamR = lamR_max_per_leaf[i]
            # lamL_a = jnp.asarray(lamL)
            # lamR_a = jnp.asarray(lamR)
            # lam_v = jax.lax.select(lamL_a >= lamR_a, lamL_a, lamR_a)
            # If no recompute, keep previous lam
            # lam_used = jax.lax.cond(do_recompute, lambda _: lam_v, lambda _: lam_prev_old, operand=None)

            # Project
            if (max_precond_dim is not None and max_precond_dim > 0 and m > max_precond_dim):
                left_id = True
            if (max_precond_dim is not None and max_precond_dim > 0 and n > max_precond_dim):
                right_id = True

            if left_id and right_id:
                g_proj = g_stk
            elif left_id:
                g_proj = _matmul(g_stk, RQ_i)
            elif right_id:
                g_proj = _matmul(jnp.swapaxes(LQ_i, -1, -2), g_stk)
            else:
                g_proj = _matmul(_matmul(jnp.swapaxes(LQ_i, -1, -2), g_stk), RQ_i)
            nu_new = b2 * nu_prev + (1.0 - b2) * (g_proj ** 2)

            if left_id and right_id:
                m_proj = mu_stk
            elif left_id:
                m_proj = _matmul(mu_stk, RQ_i)
            elif right_id:
                m_proj = _matmul(jnp.swapaxes(LQ_i, -1, -2), mu_stk)
            else:
                m_proj = _matmul(_matmul(jnp.swapaxes(LQ_i, -1, -2), mu_stk), RQ_i)
            
            upd_proj = _scale_with_atan2(m_proj, nu_new) if atan2 else _scale_with_eps(m_proj, nu_new, din_i, dout_i, N_i, nb_in_i, nb_out_i, left_id, right_id)

            if left_id and right_id:
                upd_stk = upd_proj
            elif left_id:
                upd_stk = _matmul(upd_proj, jnp.swapaxes(RQ_i, -1, -2))
            elif right_id:
                upd_stk = _matmul(LQ_i, upd_proj)
            else:
                upd_stk = _matmul(_matmul(LQ_i, upd_proj), jnp.swapaxes(RQ_i, -1, -2))

            upd_leaf = _merge_stacked_blocks(upd_stk, m, n, rslices, cslices, row_bs, col_bs)

            upd_leaves_out[i] = upd_leaf
            # L_new_leaves_out[i] = L_new_i
            # R_new_leaves_out[i] = R_new_i
            LQ_new_leaves_out[i] = LQ_i
            RQ_new_leaves_out[i] = RQ_i
            nu_new_leaves_out[i] = nu_new
            # lam_new_leaves_out[i] = lam_used

        upd_tree = jtu.tree_unflatten(treedef, upd_leaves_out)
        nu_new_tr = jtu.tree_unflatten(treedef, nu_new_leaves_out)
        L_new_tr = jtu.tree_unflatten(treedef, L_new_leaves)
        R_new_tr = jtu.tree_unflatten(treedef, R_new_leaves)
        LQ_new_tr = jtu.tree_unflatten(treedef, LQ_new_leaves_out)
        RQ_new_tr = jtu.tree_unflatten(treedef, RQ_new_leaves_out)
        # lam_new_tr = jtu.tree_unflatten(treedef, lam_new_leaves_out)

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
            # lam=lam_new_tr,
            nb_in=state.nb_in,
            nb_out=state.nb_out,
        )
        return upd_tree, new_state


    return optax.GradientTransformation(init_fn, update_fn)
