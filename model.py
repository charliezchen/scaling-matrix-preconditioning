import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from omegaconf.dictconfig import DictConfig
from rope import apply_rope
from causal_flash_attention import causal_flash_attention
from jax.sharding import PartitionSpec as P

class TransformerDecoder(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.embed = nnx.Embed(num_embeddings=cfg.V, features=cfg.D, embedding_init=fsdp_init('embedding_in', cfg), dtype=cfg.dtype, rngs=rngs)
    self.blocks = [TransformerBlock(cfg, rngs) for _ in range(cfg.N)]
    self.out_ln = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.readout = nnx.Linear(in_features=cfg.D, out_features=cfg.V, use_bias=False, kernel_init=fsdp_init('embedding_out', cfg), rngs=rngs, dtype=cfg.dtype)
    self.gradient_checkpointing = cfg.gradient_checkpointing

    self.rope = cfg.rope
    if not cfg.rope:
        self.pos_embed = nnx.Embed(num_embeddings=cfg.L, features=cfg.D, embedding_init=fsdp_init('embedding_in', cfg), rngs=rngs, dtype=cfg.dtype)
  
  @nnx.jit
  def get_features(self, x):
    # Token + positional embedding
    h = self.embed(x)  # [B, S, D]
    if not self.rope:
      h += self.pos_embed(jnp.arange(x.shape[1])[None, ...])
    for block in self.blocks:
      block = nnx.remat(block) if self.gradient_checkpointing else block
      h = block(h)
    return h
  
  @nnx.jit
  def get_features_and_logits(self, x):
    h = self.get_features(x)
    return h, self.readout(self.out_ln(h))

  @nnx.jit
  def __call__(self, x):  # [B, S]
    h = self.get_features(x)
    h = self.out_ln(h)
    return self.readout(h)  # [B, S, O] where O is either cfg.O or cfg.V


class Attention(nnx.Module):
  """Custom multi-headed attention implementation with D x D projection matrices."""
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.num_heads = cfg.D // cfg.dh
    self.head_dim = cfg.dh
    self.scale = (1 / self.head_dim) ** 0.5
    # D x D projection matrices
    self.query_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_qkv_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.key_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_qkv_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.value_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_qkv_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.output_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_out_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    
    # Layer normalization for query-key normalization
    self.q_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.k_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)

    # Check and store flash attention availability
    self.use_flash_attn = cfg.use_flash_attn

    # self.attention = lambda q, k, v: jax.nn.dot_product_attention(q, k, v, is_causal=True)

    self.rope = cfg.rope

  def __call__(self, x): # [B, S, D]
    B, S, D = x.shape
    H = self.num_heads
    
    q = self.query_proj(x) # [B, S, D]
    k = self.key_proj(x) # [B, S, D]
    v = self.value_proj(x) # [B, S, D]
    
    # Fused reshape and transpose
    q = q.reshape(B, S, H, -1) # [B, S, H, D/H]
    k = k.reshape(B, S, H, -1) # [B, S, H, D/H]
    v = v.reshape(B, S, H, -1) # [B, S, H, D/H]
    
    q = self.q_norm(q)
    k = self.k_norm(k)

    # position embedding
    if self.rope:
        position = jnp.arange(S)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])

    # attention
    # attention switch
    if self.use_flash_attn:
      # Use pure JAX causal flash attention kernel
      q_flash = jnp.swapaxes(q, 1, 2)  # [B, H, S, D/H]
      k_flash = jnp.swapaxes(k, 1, 2)
      v_flash = jnp.swapaxes(v, 1, 2)
      out = causal_flash_attention(q_flash, k_flash, v_flash)  # [B, H, S, D/H]
      out = jnp.swapaxes(out, 1, 2)  # [B, S, H, D/H]
    else:
      # Use the standard JAX attention
      out = jax.nn.dot_product_attention(q, k, v, is_causal=True) # [B, S, H, D/H]


    # output projection
    out = self.output_proj(out.reshape(B, S, -1))
    return out


class TransformerBlock(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.ln1 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    # Use our custom multi-headed attention implementation
    self.attn = Attention(cfg, rngs)
    self.ln2 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.mlp = Mlp(cfg, rngs)
    self.branch_multiplier = 1 / (cfg.N / cfg.base_N) if cfg.depth_mup else 1
    
  def __call__(self, x):  # [B, S, D]
    # Pre-layernorm attention block
    h = self.ln1(x)

    # Attention and residual connection
    x = x + self.attn(h) * self.branch_multiplier
    
    # Pre-layernorm MLP block
    return x + self.mlp(self.ln2(x)) * self.branch_multiplier


class Mlp(nnx.Module):
  """Multilayer perceptron."""
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(in_features=cfg.D, out_features=cfg.mlp_expansion*cfg.D, use_bias=False, kernel_init=fsdp_init('mlp_in', cfg), dtype=cfg.dtype, rngs=rngs)
    self.fc2 = nnx.Linear(in_features=cfg.mlp_expansion*cfg.D, out_features=cfg.D, use_bias=False, kernel_init=fsdp_init('mlp_out', cfg), dtype=cfg.dtype, rngs=rngs)

    self.swiglu = cfg.swiglu
    if cfg.swiglu:
        self.fc3 = nnx.Linear(in_features=cfg.D, out_features=cfg.mlp_expansion*cfg.D, use_bias=False, kernel_init=fsdp_init('mlp_in', cfg), dtype=cfg.dtype, rngs=rngs)
    
  def __call__(self, x):  # [B, S, D]
    # SwiGLU
    if self.swiglu:
        h = jax.nn.swish(self.fc1(x)) * self.fc3(x)       # [B, S, F]
    else:
        h = jax.nn.gelu(self.fc1(x))
    return self.fc2(h)  # [B, S, D]


def fsdp_init(layer_type: str, cfg: DictConfig):
  """Initialize weights with optional FSDP partitioning."""
  partition_fn = nnx.with_partitioning
  kernel_init = jax.nn.initializers.normal(stddev=cfg.init_std_mult*jnp.sqrt(1.0/cfg.D))
  embed_init = jax.nn.initializers.normal(stddev=cfg.init_std_mult*cfg.embed_init_std)
  zero_init = jax.nn.initializers.zeros
  if cfg.fsdp_enabled:
     fsdp_axis = "data"
  else:
     fsdp_axis = None
  match layer_type:
    case "embedding_in":  # [V, D]
      return partition_fn(embed_init, ("model", fsdp_axis))
    case "embedding_out":  # [D, V]
      return partition_fn(zero_init, (fsdp_axis, "model"))
    case "attn_qkv_proj":  # [D, D]
      return partition_fn(kernel_init, (fsdp_axis, "model"))
    case "attn_out_proj":  # [D, D]
      return partition_fn(zero_init, ("model", fsdp_axis))
    case "mlp_in": # [D, F]
      return partition_fn(kernel_init, (fsdp_axis, 'model'))
    case "mlp_out": # [D, F]
      return partition_fn(zero_init, ('model', fsdp_axis))
    case _:
      raise ValueError(f"unrecognized layer type: {layer_type}")

# Deprecated
def create_sharded_model(c: DictConfig, mesh: Mesh, seed: int):
  """
  initialize sharded model without putting it on a single device
  https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html
  """

  @nnx.jit
  def initialize_sharded_model():
    model = TransformerDecoder(c, rngs=nnx.Rngs(seed)) # unsharded at this moment
    state = nnx.state(model) # the model's state, a pure pytree
    pspecs = nnx.get_partition_spec(state) # get annotations from state
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state) # the model is sharded now
    return model

  with mesh:
    model = initialize_sharded_model()

  return model

def create_pspec_tree(state):
    """
    Build a pspec tree that is compatible with nnx state trees.

    nnx.get_partition_spec emits specs for every node, including MaskedNode
    placeholders used by optax.multi_transform. We also patch 3D tensors
    (blocking) to add a leading None so the block dimension remains unsharded.
    """
    all_pspec = nnx.get_partition_spec(state)
    path_and_spec, _ = jax.tree.flatten_with_path(all_pspec)
    pspec_map = {path: spec for (path, spec) in path_and_spec}

    path_and_state, tree_def = jax.tree.flatten_with_path(state)
    def _safe_shard(p, v):
       if hasattr(v, "shape") and len(v.shape) == 3:
          return P(None, *pspec_map[p])
       # Don't shard scalars and vectors
       # Spectral normalization has vectors
       elif not hasattr(v, "shape") or len(v.shape) <= 1:
          return P()
       else:
          return pspec_map[p]
    specs = [
        _safe_shard(p, v)
        for (p, v) in path_and_state
    ]
    return jax.tree.unflatten(tree_def, specs)

def initialize_sharded_optimizer_state(c: DictConfig, tx, mesh: Mesh, seed: int):
    """Return sharded optimizer state without keeping full objects alive."""
    @nnx.jit
    def _init_state():
        model = TransformerDecoder(c, rngs=nnx.Rngs(seed))
        optimizer = nnx.Optimizer(model, tx)
        state = nnx.state(optimizer)

        # Build pspec tree using shared helper (filters MaskedNode and fixes 3D specs)
        pspec = create_pspec_tree(state)

        # Print the sharding spec of all states
        if False:
            arrays1, _ = jax.tree.flatten_with_path(pspec)
            arrays2, _ = jax.tree.flatten_with_path(state)
            for (path1, p), (path2, v) in zip(arrays1, arrays2):
                key_path1 = jax.tree_util.keystr(path1, simple=True, separator='/')
                key_path2 = jax.tree_util.keystr(path2, simple=True, separator='/')
                print(f"[key path 1]: {key_path1} {p}")
                print(f"[key path 2]: {key_path2} {v.shape if hasattr(v, 'shape') else 'None'}")
            print("Length of two pytrees:", len(arrays1), len(arrays2))
        
        state = jax.lax.with_sharding_constraint(state, pspec)

        # jax.lax.with_sharding_constraint allows uneven sharding, but I
        # don't want it. So manually double check the shape.
        local_shapes = jax.tree.map(lambda s, spec: NamedSharding(mesh, spec).shard_shape(s.shape), state, pspec)

        return state

    with mesh:
        state = _init_state()
        # emb_state = state.embed.embedding.value  # adjust path to your structure
        # print(f"[Embed state sharding]: {emb_state.sharding}")
        return state
