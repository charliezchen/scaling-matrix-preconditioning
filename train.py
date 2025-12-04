import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import data, utils
import model as transformer
import mlp
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig
import pickle
from math import prod, ceil
from functools import partial
from optax_optim import scale_by_shampoo, scale_by_muon, scale_by_adam, scale_by_momentum, scale_by_soap, scale_by_soap_bucketed
import orbax.checkpoint as ocp
from utils import get_vm_region, bucket_exists, get_flop_per_token, device_hardware_flops, clean_gcs_path, _broadcast_str_from_host0
import time


@nnx.jit
def eval_features(features, prev_features):
  delta = features - prev_features
  return {'h': jnp.sqrt(jnp.mean(features**2)), 'dh': jnp.sqrt(jnp.mean(delta**2))}

@nnx.jit
def eval_features_and_logits(features, logits, prev_features, prev_logits):
  delta_features = features - prev_features
  delta_logits = logits - prev_logits
  return {'h': jnp.sqrt(jnp.mean(features**2)), 'dh': jnp.sqrt(jnp.mean(delta_features**2)),
          'f': jnp.sqrt(jnp.mean(logits**2)), 'df': jnp.sqrt(jnp.mean(delta_logits**2))}

def train_and_evaluate(cfg: DictConfig):
  # set seed
  np.random.seed(cfg.seed)

  if not cfg.single_host:
    jax.distributed.initialize()

  fsdp_dim = jax.device_count() / cfg.model.tp_dim
  mesh = jax.make_mesh((fsdp_dim,cfg.model.tp_dim), ('data', 'model'))
  if cfg.model.tp_dim > 1:
     V_pad = ceil(cfg.model.V / 128) * 128
     print(f"Model sharding dim is {cfg.model.tp_dim} > 1")
     print(f"Pad vocab dim from {cfg.model.V} to {V_pad}")
     cfg.model.V = V_pad


  # datasets
  tokens_per_step = cfg.opt.B * cfg.model.L
  # ── gradient-accumulation parameters ─────────────────────────────
  if cfg.opt.B_max is None:
    B_micro = cfg.opt.B
  else:
    B_micro = min(cfg.opt.B, cfg.opt.B_max)                 # per-device batch
    assert cfg.opt.B % B_micro == 0, \
        "`cfg.opt.B` must be an integer multiple of `cfg.opt.B_max`"
  accum_steps = cfg.opt.B // B_micro                     # micro-batches / update

  B_eval = min(cfg.T_eval // cfg.model.L, B_micro); tokens_per_step_eval = B_eval * cfg.model.L
  data_sharding = NamedSharding(mesh, P('data'))
  get_batch_train, total_train_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_micro, cfg.shuffle_data, 'train', data_sharding)
  try:
    get_batch_test, total_test_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_eval, cfg.shuffle_data, 'test', data_sharding)
  except:
    get_batch_test, total_test_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_eval, cfg.shuffle_data, 'val', data_sharding)
  print(f"Total train tokens: {total_train_tokens/1e9:.2g}B")
  print(f"Total test tokens: {total_test_tokens/1e9:.2g}B")

  # Process optimizer-specific overrides via "|" in opt.name
  if "|" in cfg.opt.name:
      parts = cfg.opt.name.split("|")
      cfg.opt.name = parts[0]
      if len(parts) > 1: cfg.opt.lr *= float(parts[1]) if float(parts[1]) > 0 else 2 ** float(parts[1])
      if len(parts) > 2: cfg.opt.embed_lr_mult *= float(parts[2])
      if len(parts) > 3: cfg.opt.readout_lr_mult *= float(parts[3])
      if len(parts) > 4: cfg.token_per_param = float(parts[4]) if float(parts[4]) > 0 else None


  
  # Create the base model and extract its shapes
  cfg.model.depth_mup = cfg.opt.depth_mup # to determine branch multiplier
  if cfg.model.aspect_ratio is not None:
    cfg.model.D = cfg.model.N * cfg.model.aspect_ratio
    print(f'Setting D to {cfg.model.N} * {cfg.model.aspect_ratio} = {cfg.model.D}')
  if cfg.arch == 'transformer':
    def instantiate_model(model_cfg):
      return transformer.TransformerDecoder(model_cfg, rngs=nnx.Rngs(cfg.seed))
  elif cfg.arch == 'mlp':
    def instantiate_model(model_cfg):
      return mlp.MLPRegression(model_cfg, rngs=nnx.Rngs(params=jax.random.key(cfg.seed)))
  else:
    raise ValueError(f'Unknown model type: {cfg.arch}')
  base_model_cfg = cfg.model.copy(); base_model_cfg.D = cfg.model.base_D
  
  abstract_base_model = nnx.eval_shape(lambda: instantiate_model(base_model_cfg))
  base_shapes = jax.tree.map(lambda x: jnp.array(x.shape), nnx.state(abstract_base_model))

  abstract_model = nnx.eval_shape(lambda: instantiate_model(cfg.model))
  model_graphdef = nnx.graphdef(abstract_model)
  shapes = jax.tree.map(lambda x: jnp.array(x.shape), nnx.state(abstract_model))
  paths = jax.tree_util.tree_map_with_path(lambda path, p: '.'.join(str(x) for x in path), nnx.state(abstract_model))
  num_params = sum(prod(p.shape) for p in jax.tree_util.tree_leaves(nnx.state(abstract_model)))
  non_embed_params = sum((prod(p.shape) if cfg.model.V not in p.shape else 0) for p in jax.tree_util.tree_leaves(nnx.state(abstract_model)))

  print(f'Number of non-embedding parameters: {non_embed_params/1e9:.3g} B')
  print(f'Number of parameters: {num_params/1e9:.3g} B')
  print(f"Size of parameters: {num_params*4/1e9:.3g} GB")
  checkpoints_size = 2*cfg.opt.B*cfg.model.L*cfg.model.D*cfg.model.N
  print(f"Size of layer-only activations: {checkpoints_size/1e9:.3g} GB")


  # possibly auto-set T from fitted scale/exponent
  flop_per_token = get_flop_per_token(cfg.model)
  if cfg.total_flops:
     cfg.T = round(cfg.total_flops / (3*flop_per_token))
     print(f"Setting T to {cfg.T:.2g} from {cfg.total_flops} flops / {flop_per_token} flop per token")
  elif cfg.token_per_param:
    cfg.T = round(cfg.token_per_param * num_params)
    print(f"Setting T to {cfg.T:.2g} from {cfg.token_per_param} toks/param")
  elif cfg.scale is not None and cfg.exponent is not None:
    C = 1e15 * ((num_params / cfg.scale) ** cfg.exponent)
    cfg.T = round(C / (6 * num_params))
    print(f"Setting T to {cfg.T} from scale and exponent")
  else:
    assert cfg.T is not None, "Cannot determine T; please set one of total_flops, token_per_param, or scale/exponent."
  num_train_steps = cfg.T // tokens_per_step
  num_valid_steps = cfg.T_eval // tokens_per_step_eval
  print(f"Number of train steps: {num_train_steps} = {cfg.T} / {tokens_per_step}")
  print(f"Number of valid steps: {num_valid_steps} = {cfg.T_eval} / {tokens_per_step_eval}")

  if cfg.opt.warmup_tokens < 1:
    cfg.opt.warmup_tokens = round(cfg.opt.warmup_tokens * cfg.T)
  num_warmup_steps = cfg.opt.warmup_tokens // tokens_per_step
  print(f"Number of warmup steps: {num_warmup_steps} = {cfg.opt.warmup_tokens} / {tokens_per_step}")
  

  # Tie lr multiplier
  if cfg.opt.readout_lr_mult is None:
    cfg.opt.readout_lr_mult = cfg.opt.embed_lr_mult

  # convert half life to beta
  if cfg.opt.t1 is not None:
    cfg.opt.b1 = max(1 - tokens_per_step / cfg.opt.t1, 0)
  if cfg.opt.t2 is not None:
    cfg.opt.b2 = max(1 - tokens_per_step / cfg.opt.t2, 0)

  # convert normalized weight decay half life to wd
  if cfg.opt.wd_half_life is not None:
    if cfg.opt.wd_half_life < 0:
      cfg.opt.wd_half_life = 2**cfg.opt.wd_half_life
    # wd_half_life = 1/(wd * total_steps) => wd = 1 / (wd_half_life * total_steps)
    cfg.opt.weight_decay = 1 / (cfg.opt.wd_half_life * num_train_steps)
  elif cfg.opt.wdxD is not None:
    if cfg.opt.wdxD < 0:
      cfg.opt.wdxD = 2**cfg.opt.wdxD
    cfg.opt.weight_decay = float(f"{cfg.opt.wdxD / cfg.model.D:.4g}") # Keep four significant digits


  # Convert negative values to 2^x
  if cfg.opt.lr < 0:
    cfg.opt.lr = 2**cfg.opt.lr
  if cfg.opt.weight_decay < 0:
    cfg.opt.weight_decay = 2**cfg.opt.weight_decay

  if os.path.exists(f'{cfg.ds_path}/meta.pkl'):
    with open(f'{cfg.ds_path}/meta.pkl', 'rb') as f:
      meta = pickle.load(f)
      cfg.model.V = int(jnp.ceil(meta['vocab_size'] / 32)) * 32

  if cfg.opt.scale_eps is None:
    cfg.opt.scale_eps = cfg.opt.mup
    print(f'Setting scale_eps to {cfg.opt.scale_eps}')
  if cfg.opt.depth_mup is None:
    cfg.opt.depth_mup = cfg.opt.mup
    print(f'Setting depth_mup to {cfg.opt.depth_mup}')
  print(f"Enable fsdp: {cfg.model.fsdp_enabled}")
  print(f"Enable gradient checkpointing: {cfg.model.gradient_checkpointing}")

  ablation_actions = {
      "double_lr": lambda cfg: setattr(cfg.opt, "lr", cfg.opt.lr * 2),
      "half_lr": lambda cfg: setattr(cfg.opt, "lr", cfg.opt.lr / 2),
      "double_wd": lambda cfg: setattr(cfg.opt, "weight_decay", cfg.opt.weight_decay * 2),
      "half_wd": lambda cfg: setattr(cfg.opt, "weight_decay", cfg.opt.weight_decay / 2),
      # base model has 3713617920 training tokens
      "fix_warmup_ratio": lambda cfg: setattr(cfg.opt, "warmup_tokens", round(cfg.opt.warmup_tokens/3713617920*cfg.T)),
  }
  
  if cfg.custom_ablation_str:
      s = cfg.custom_ablation_str
      if s not in ablation_actions.keys():
        print(f"Ablation string {s} not recognized, skipping ablation")
      else:
        action = ablation_actions.get(s)
        if action:
            # print what’s about to change
            print(f"Applying ablation: {s}")
            before = {k: getattr(cfg.opt, k, None) for k in ["lr", "weight_decay", "warmup_tokens"]}
            action(cfg)
            after = {k: getattr(cfg.opt, k, None) for k in ["lr", "weight_decay", "warmup_tokens"]}
            print(f"Before: {before}\nAfter:  {after}")

  wandb_disabled = os.environ.get('WANDB_DISABLED', '').lower() in ('1', 'true', 'yes')
  if jax.process_index() == 0 and not wandb_disabled and cfg.wandb_mode != 'disabled':
    config = utils.flatten_dict(cfg)
    config['num_params'] = num_params
    config['non_embed_params'] = non_embed_params
    config['hardware'] = jax.devices()[0].device_kind.lower()
    config['device_count'] = jax.device_count()
    config['elr'] = (cfg.opt.lr * cfg.opt.weight_decay) ** 0.5
    run = wandb.init(project=cfg.wandb_project, config=config, mode=cfg.wandb_mode, tags=[cfg.wandb_tag], resume="allow", id=str(cfg.run_id) if cfg.run_id is not None else None)
  else:
     run = None


  if cfg.wandb_mode == "disabled":
     print("wandb_mode is set to disabled, so not saving checkpoints.")
     ckpt_path = None
  elif cfg.base_model_checkpoint_path is not None and cfg.ckpt:
    if jax.process_index() == 0:
      region = get_vm_region().replace("-", "_") # e.g. us_central1
      sweep_segment = (run.sweep_id if (run and run.sweep_id) else "no_sweep")
      run_segment = (run.id if run else time.strftime("%Y%m%d-%H%M%S"))
      ckpt_path_host0 = f'gs://{cfg.base_model_checkpoint_path}_{region}/{sweep_segment}/{run_segment}/'
      bucket_path = ckpt_path_host0.split("/")[2]
      if not bucket_exists(bucket_path):
        raise ValueError(f"Bucket gs://{bucket_path} doesn't exist")
    else:
      ckpt_path_host0 = ""

    # Broadcast the computed checkpoint path to all hosts
    ckpt_path = _broadcast_str_from_host0(ckpt_path_host0)
    print(f"[Process {jax.process_index()}]: checkpoint path is set to {ckpt_path}")
  else:
    print(f"Checkpoint path is {cfg.base_model_checkpoint_path} and ckpt is {cfg.ckpt}. Skip checkpointing.")
    ckpt_path = None


  # Mark preemptable process for sweep
  if run and run.sweep_id:
     run.mark_preempting()

  # Early termination
  if not cfg.opt.mup and cfg.opt.scale_eps:
      print("Running SP with eps scaling is pointless, so exiting...")
      if run:
        wandb.finish()
      exit()

  with mesh: ds_valid = jnp.stack([next(get_batch_test) for i in range(num_valid_steps)])

  # ------------------------------------------------------------------------
  # 1. tag every parameter with the optimiser name you want
  do_spectral_norm = cfg.opt.name.startswith('spectral-')
  if do_spectral_norm:
    cfg.opt.name = cfg.opt.name.replace('spectral-', '')
  def assign_optimizer(p, path_str):
      if cfg.opt.name.endswith('_adam'):
          return 'adam' if ('embed' in path_str or 'readout' in path_str) \
                        else cfg.opt.name.replace('_adam', '')
      return cfg.opt.name

  optimizers = jax.tree.map(assign_optimizer, shapes, paths)

  # ------------------------------------------------------------------------
  # 2. learning-rate per-parameter (µP-aware)
  N_scale = cfg.model.N / cfg.model.base_N # depth scaling factor
  # true if not embedding or readout
  is_bulk = jax.tree.map(lambda x: not ('embed' in x or 'readout' in x), paths)
  N_scale = jax.tree.map(lambda x: N_scale if x else 1, is_bulk)

  # Precompute normalized block counts per-parameter for use in LR and optimizers
  def compute_nb(base_shape, shape):
      base_din, base_dout = base_shape
      din, dout = shape
      # one block if exceeding max precond dim
      base_nb_in = jnp.ceil(base_din / cfg.opt.block_size) if base_din <= cfg.opt.max_precond_dim else 1
      base_nb_out = jnp.ceil(base_dout / cfg.opt.block_size) if base_dout <= cfg.opt.max_precond_dim else 1
      nb_in = jnp.ceil(din / cfg.opt.block_size) if din <= cfg.opt.max_precond_dim else 1
      nb_out = jnp.ceil(dout / cfg.opt.block_size) if dout <= cfg.opt.max_precond_dim else 1
      # normalize relative to base
      return nb_in / base_nb_in, nb_out / base_nb_out

  nb_trees = jax.tree.map(compute_nb, base_shapes, shapes)
  is_pair = lambda x: isinstance(x, tuple)
  nb_in_tree  = jax.tree_util.tree_map(lambda t: t[0], nb_trees, is_leaf=is_pair)
  nb_out_tree = jax.tree_util.tree_map(lambda t: t[1], nb_trees, is_leaf=is_pair)

  def assign_lr(base_shape, shape, N_scale, nb_in, nb_out, opt_name, path_str):
      base_lr             = cfg.opt.lr
      base_din, base_dout = base_shape
      din, dout           = shape

      din_rel  = din  / base_din
      dout_rel = dout / base_dout

      if   'embed'   in path_str: 
        base_lr *= cfg.opt.embed_lr_mult
      elif 'readout' in path_str: 
        base_lr *= cfg.opt.readout_lr_mult

      if not cfg.opt.mup or do_spectral_norm:
        return base_lr
      if not cfg.opt.depth_mup:
        N = 1
      else:
        N = N_scale

      if   opt_name == 'adam':                         mult = 1 / din_rel
      elif opt_name in ('muon', 'shampoo'):            mult = (dout_rel / din_rel) ** 0.5
      elif opt_name == 'soap':
        base_in_active  = base_din <= cfg.opt.max_precond_dim
        base_out_active = base_dout <= cfg.opt.max_precond_dim
        in_active  = din  <= cfg.opt.max_precond_dim
        out_active = dout <= cfg.opt.max_precond_dim
        assert in_active == base_in_active, "Precondition input active must be the same as base"
        assert out_active == base_out_active, "Precondition output active must be the same as base"
        assert in_active or out_active, "At least one side must be active"

        if in_active and out_active:
          mult = (dout_rel / din_rel) ** 0.5 * (nb_in * nb_out) ** -0.5
        elif in_active:
          mult = (din_rel * nb_in) ** -0.5
        else:
          mult = (dout_rel / nb_out) ** 0.5 / din_rel
      elif opt_name == 'shampoo2':                     mult = 1 / N
      elif opt_name.startswith('grafted_shampoo'):     mult = 1 / din_rel
      elif opt_name == "sgd":                          mult = dout_rel / din_rel * N
      else: raise ValueError(f'Unknown optimiser {opt_name}')

      return base_lr * mult

  lrs = jax.tree.map(assign_lr, base_shapes, shapes, N_scale, nb_in_tree, nb_out_tree, optimizers, paths)

  # ------------------------------------------------------------------------
  # 3. build optimiser transforms lazily so unused ones never compile
  B_scale = cfg.opt.B / cfg.opt.base_B

  factory = {
      'adam':   lambda *, din, dout, N, **_: scale_by_adam(
          b1=cfg.opt.b1 if cfg.opt.adam_b1 is None else cfg.opt.adam_b1, 
          b2=cfg.opt.b2 if cfg.opt.adam_b2 is None else cfg.opt.adam_b2, 
          eps=cfg.opt.eps if cfg.opt.adam_eps is None else cfg.opt.adam_eps,
          scale_eps=cfg.opt.scale_eps, B=B_scale, N=N, din=din, dout=dout),
      'muon':   lambda *, din, dout, N, **_: scale_by_muon(
          beta=cfg.opt.b1, eps=cfg.opt.eps, nesterov=True,
          scale_eps=cfg.opt.scale_eps, din=din, dout=dout, B=B_scale, N=N),
      'shampoo': lambda *, din, dout, nb_in, nb_out, N, **_: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, eps=cfg.opt.eps,
          freq=cfg.opt.freq, scale_eps=cfg.opt.scale_eps,
          B=B_scale, N=N, eigh=cfg.opt.eigh, block_size=cfg.model.block_size,
          din=din, dout=dout, nb_in=nb_in, nb_out=nb_out),
      'shampoo2': lambda *, din, dout, nb_in, nb_out, N, **_: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, eps=cfg.opt.eps,
          freq=cfg.opt.freq, kl=0.5, kr=0.5,
          scale_eps=cfg.opt.scale_eps, B=B_scale, N=N, eigh=True, block_size=cfg.model.block_size,
          din=din, dout=dout, nb_in=nb_in, nb_out=nb_out), # Newton is broken
      'grafted_shampoo': lambda *, din, dout, nb_in, nb_out, N, **_: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.adam_eps, matrix_eps=cfg.opt.matrix_eps, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim,
          freq=cfg.opt.freq, grafting=True,
          scale_eps=cfg.opt.scale_eps, B=B_scale, N=N, eigh=cfg.opt.eigh,
          din=din, dout=dout, nb_in=nb_in, nb_out=nb_out),
      'grafted_shampoo2': lambda *, din, dout, nb_in, nb_out, N, **_: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.adam_eps, matrix_eps=cfg.opt.matrix_eps, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim, 
          freq=cfg.opt.freq, kl=0.5, kr=0.5, grafting=True,
          scale_eps=cfg.opt.scale_eps, B=B_scale, N=N, eigh=cfg.opt.eigh,
          din=din, dout=dout, nb_in=nb_in, nb_out=nb_out),
      'sgd': lambda **_: scale_by_momentum(b1=cfg.opt.b1),
      'soap': lambda *, nb_in, nb_out, N, **_: scale_by_soap_bucketed(b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.eps, matrix_eps=cfg.opt.matrix_eps, freq=cfg.opt.freq, scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N, nb_in=nb_in, nb_out=nb_out, eigh=cfg.opt.eigh, eigh_warmup_steps=cfg.opt.eigh_warmup_steps, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim, bf16_momentum=cfg.bf16_momentum, atan2=False),
  }
  from optax.transforms._masking import MaskedNode
  def mask_pytree(pytree, mask_tree):
    return jax.tree.map(
        lambda m, p: p if m else MaskedNode(), mask_tree, pytree
    )
  def make_mask(labels, group):
    return jax.tree.map(lambda label: label == group, labels)

  labels_in_use   = frozenset(jax.tree_util.tree_leaves(optimizers))
  rel_scales = jax.tree.map(
     lambda s1, s2: jnp.array([s1[0]/s2[0], s1[1]/s2[1]]),
     shapes, base_shapes
  )
  din_tree = jax.tree.map(lambda rs: rs[0], rel_scales)
  dout_tree = jax.tree.map(lambda rs: rs[1], rel_scales)

  # mask pytree inputs per optimiser label
  transforms_dict = {}
  for name, make in factory.items():
      if name not in labels_in_use:
          continue
      mask = make_mask(optimizers, name)
      masked_args = dict(
          din=mask_pytree(din_tree, mask),
          dout=mask_pytree(dout_tree, mask),
          nb_in=mask_pytree(nb_in_tree, mask),
          nb_out=mask_pytree(nb_out_tree, mask),
          N=mask_pytree(N_scale, mask),
      )
      transforms_dict[name] = make(**masked_args)

  # single-label fast-path (skips optax.multi_transform completely)
  if len(labels_in_use) == 1:
      scale_by_optim = transforms_dict[next(iter(labels_in_use))]
  else:
      scale_by_optim = optax.multi_transform(
          transforms=transforms_dict,
          param_labels=optimizers
      )

  # ------------------------------------------------------------------------
  # 4. LR scaling & full optimiser
  scale_by_lr = optax.GradientTransformation(
      init   = lambda _: None,
      update = lambda upd, state, _: (jax.tree.map(
          lambda g, lr: g * lr, upd, lrs), state)
  )

  # Optional normalization by per-parameter spectral norm of the update.
  maybe_norm_by_spect = (
      utils.norm_by_spect(cfg.num_power_iter, seed=cfg.seed, emb_norm=cfg.opt.emb_norm, hidden_norm=cfg.opt.hidden_norm, readout_norm=cfg.opt.readout_norm)
      if cfg.num_power_iter > 0 and do_spectral_norm else optax.identity()
  )
  preconditioner = optax.chain(
    optax.clip_by_global_norm(cfg.opt.clip) 
        if cfg.opt.clip > 0 else optax.identity(),
    scale_by_optim,
    maybe_norm_by_spect,
    scale_by_lr,
    optax.transforms.add_decayed_weights(cfg.opt.weight_decay), # independent wd
  )

  schedule_fn = utils.get_scheduler(
      cfg.opt.schedule,
      cfg.opt.decay_frac,
      cfg.opt.warmup_tokens // tokens_per_step,
      num_train_steps
  )

  tx = optax.chain(
      preconditioner,
      optax.scale_by_schedule(schedule_fn),
      optax.scale(-1.0)
  )

  # Use eval_shape to avoid instantiating real optimizer buffers
  abstract_optimizer = nnx.eval_shape(lambda: nnx.Optimizer(abstract_model, tx))

  opt_graphdef, abstract_opt_state = nnx.split(abstract_optimizer)
  restore_specs = transformer.create_pspec_tree(abstract_opt_state)
  restore_specs = jax.tree.map(
    lambda v, spec: jax.ShapeDtypeStruct(
        v.shape, v.dtype, sharding=NamedSharding(mesh, spec)),
    abstract_opt_state, restore_specs
  )


  del base_shapes, shapes

  get_in_out = partial(data.get_in_out, task='regression' if cfg.ds_path == 'fourier' else 'ntp')
  loss_fn = optax.l2_loss if cfg.ds_path == 'fourier' else optax.softmax_cross_entropy_with_integer_labels

  @nnx.jit
  def batch_loss_fn(model, batch):
    x, y, weights = get_in_out(batch)
    logits = model(x).astype(jnp.float32) # float32 for stability
    losses = loss_fn(logits, y).mean()
    mean_loss = jnp.sum(losses * weights) / (weights.sum() + 1e-6)
    return mean_loss

  def _micro_loss_and_grad(model, batch):
      return nnx.value_and_grad(batch_loss_fn)(model, batch)

  @partial(jax.jit, donate_argnames=('opt_state'))
  def train_step(opt_state, micro_batches): #  micro_batches = list/tuple of length accum_steps   
    loss_sum, grads_sum = 0.0, None
    for batch in micro_batches:                    # python loop unrolled by jit, OK because accum_steps is small
        model = nnx.merge(model_graphdef, opt_state.model)
        loss, grads = _micro_loss_and_grad(model, batch)
        loss_sum += loss
        grads_sum = grads if grads_sum is None else jax.tree.map(jnp.add, grads_sum, grads)
    loss = loss_sum / accum_steps

    grads_mean = jax.tree.map(lambda g: g / accum_steps, grads_sum)

    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads_mean)
    opt_state = nnx.state(optimizer)
    return opt_state, loss
  

  def forward_loss_fn(param, batch):
    model = nnx.merge(model_graphdef, param)
    return batch_loss_fn(model, batch)

  @nnx.jit
  def eval_step(params, dataset):
      def _body(total_loss, batch):
          loss = forward_loss_fn(params, batch)
          return total_loss + loss, None
      total_loss, _ = jax.lax.scan(_body, 0.0, dataset)
      param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)]))
      return {"eval_loss": total_loss / len(dataset), "param_norm": param_norm}

  # training loop
  pending_train_metrics = None
  tau = 0
  prev_features = None
  prev_logits = None
  
  eval_steps = set(
      [int(i) for i in jnp.linspace(0, num_train_steps-1, cfg.num_evals)] + 
      ([] if cfg.no_geom_eval_step else [int(i) for i in jnp.geomspace(1, num_train_steps-1, 20)])
  )
  eval_steps.add(0)
  eval_steps.add(num_train_steps-1)
  if cfg.tracing:
    assert jax.process_count() == 1, "Only support single process tracing"
    assert cfg.benchmarking, "Expect benchmarking flag to be true"
  if cfg.benchmarking:
    print("Benchmarking flag is True. Will exit after 111 steps.")
    eval_steps = set([11, 111]) # benchmarking for 100 steps
    if cfg.tracing:
        eval_steps = set([11, 21]) # trace only 10 steps
    # disable checkpointing
    print("Disable checkpointing")
    ckpt_path = None

  # Creates checkpoint manager
  if ckpt_path is not None:
    mngr = ocp.CheckpointManager(
        ckpt_path,
        # ocp.AsyncCheckpointer(ocp.CompositeCheckpointHandler("opt_state", "prev")),
        options=ocp.CheckpointManagerOptions(save_interval_steps=cfg.checkpoint_frequency, max_to_keep=1)
    )
  else:
    mngr = None

  latest_ckpt_step = mngr.latest_step() if mngr is not None else None
  if latest_ckpt_step is None:
    start_step = -1
    opt_state = transformer.initialize_sharded_optimizer_state(cfg.model, tx, mesh, cfg.seed)
  else:
    start_step = latest_ckpt_step
    print(f"Resume training from step {start_step}")
    restore_args = {'opt_state': ocp.args.StandardRestore(restore_specs)}
    if cfg.checkpoint_features:
      restore_args.update({
          'prev_features': ocp.args.ArrayRestore(),
          'prev_logits': ocp.args.ArrayRestore(),
      })
    ckpt_state = mngr.restore(start_step, args=ocp.args.Composite(**restore_args))
    opt_state = ckpt_state['opt_state']
    if cfg.checkpoint_features:
      prev_features = ckpt_state['prev_features']
      prev_logits = ckpt_state['prev_logits']


#   pbar = tqdm(range(num_train_steps + 1)) # Add one more step for evaluating the final model

  last_time = None
  last_eval_step = None
  device_count = jax.device_count()
  device = jax.devices()[0]
  theoretical_flops = device_hardware_flops(device) * device_count
  profile_started = False


  with mesh:
    for step in range(num_train_steps + 1):
      # Fast forward to checkpoint
      if step <= start_step:
        next(get_batch_train)
        tau += cfg.opt.lr * schedule_fn(step)
        continue

      lr_t = cfg.opt.lr * schedule_fn(step)
      tokens_seen = step*tokens_per_step
      compute_spent = 3 * tokens_seen * flop_per_token
      compute_spent_approx = 6 * tokens_seen * num_params
      non_embed_compute = 6 * tokens_seen * non_embed_params

      # eval step at linearly spaced intervals
      if step in eval_steps:
        if last_eval_step is not None:
          elapsed_time = time.perf_counter() - last_time
          elapsed_step = step - last_eval_step
          consumed_tokens = elapsed_step * cfg.opt.B*cfg.model.L
          throughput = consumed_tokens / elapsed_time
          achieved_flops = 3 * consumed_tokens * flop_per_token
          flops_per_sec = achieved_flops / elapsed_time
          mfu = flops_per_sec / theoretical_flops

        wandb_eval_metrics = eval_step(opt_state.model, ds_valid)
        features, logits = nnx.merge(model_graphdef, opt_state.model).get_features_and_logits(get_in_out(ds_valid[0])[0])
        if prev_features is not None and prev_logits is not None:
          pending_feature_metrics = eval_features_and_logits(features, logits, prev_features, prev_logits)
        else:
          pending_feature_metrics = {'h': jnp.sqrt(jnp.mean(features**2)), 'f': jnp.sqrt(jnp.mean(logits**2))}
        wandb_eval_metrics |= pending_feature_metrics
        if device.platform == 'tpu':
            stats = jax.local_devices()[0].memory_stats()
            wandb_eval_metrics |= {
                "tpu_bytes_in_use": stats["bytes_in_use"] / 1e9,
                "tpu_peak_bytes_in_use": stats["peak_bytes_in_use"] / 1e9,
            }

        # only log if average over more than 100 steps
        if last_eval_step is not None:# and elapsed_step > 50:
            wandb_eval_metrics |= {"throughput": throughput, "flops_per_sec": flops_per_sec, "mfu": mfu}
            if cfg.benchmarking:
                if jax.process_index() == 0:
                    print("Successfully logged the MFU. Gracefully exit..")
                    if cfg.tracing:
                      jax.profiler.stop_trace()
                    if run:
                      wandb.log(wandb_eval_metrics)
                      wandb.finish()
                return 0

        prev_features = features
        prev_logits = logits
        wandb_eval_metrics |= {'step': step, 'tokens': tokens_seen, 'progress': tokens_seen / cfg.T, 'compute': compute_spent, 'compute_approx': compute_spent_approx, 'non_embed_compute': non_embed_compute, 'tau': tau, 'lr': lr_t}
        if run:
            wandb.log(wandb_eval_metrics)

        # Start computing throughput after the first 10 steps
        if step > start_step + 10:
            last_time = time.perf_counter()
            last_eval_step = step
            if not profile_started:
                if cfg.tracing:
                    jax.profiler.start_trace("traces/")
            profile_started = True

      # training step
      with jax.profiler.StepTraceAnnotation("train_step", step_num=step - 9):
        micro_batches = [next(get_batch_train) for _ in range(accum_steps)]
        if step == 0:
           print(f"[data sharding]: {micro_batches[0].sharding}")
        opt_state, loss = train_step(opt_state, micro_batches)

      if jax.process_index() == 0 and step % 1000 == 0:
        print(f'[step {step}]: L={loss:.2f}')

      # Checkpointing
      if mngr is not None:
        save_args = {
            'opt_state': ocp.args.StandardSave(opt_state),
        }
        if cfg.checkpoint_features:
            save_args.update({
                'prev_features': ocp.args.ArraySave(prev_features),
                'prev_logits': ocp.args.ArraySave(prev_logits),
            })
        mngr.save(step, args=ocp.args.Composite(**save_args))

    #   train_metrics = {'step': step, 'tokens': tokens_seen, 'train_loss': loss,
    #                     'compute': compute_spent, 'non_embed_compute': non_embed_compute,
    #                     'tau': tau, 'lr': lr_t}
      tau += lr_t

      # async logging
    #   if pending_train_metrics is not None:
      #   if cfg.wandb_project is not None: wandb.log(pending_train_metrics)
    #   pending_train_metrics = train_metrics

    # wandb.log(pending_train_metrics)
      if run and cfg.benchmarking:
        wandb.log({"step": step, "train_loss": loss})
    if run:
        wandb.finish()
    if mngr is not None:
        mngr.close()
        # gcs only needs to be cleaned once
        if jax.process_index() == 0:
            clean_gcs_path(ckpt_path)

    
