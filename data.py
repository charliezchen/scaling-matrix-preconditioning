import jax
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import glob
from functools import partial # Added
from typing import Iterator, Tuple, NamedTuple # Added

# ============================================================================
# Fourier Regression Task - Data Generation Functions
# ----------------------------------------------------------------------------

class FourierTargetParameters(NamedTuple):
    """Holds the fixed parameters defining the Fourier target function."""
    k_vectors: jax.Array
    weights: jax.Array
    phase_shifts: jax.Array

def _generate_fourier_target_parameters( # Renamed to avoid conflict if user has similar global func
    key: jax.Array,
    V_input_dim: int,
    N_fourier_modes: int,
    alpha_fourier_exponent: float
) -> FourierTargetParameters:
    """Generates the fixed parameters for a target function composed of Fourier modes."""
    key_mags, key_dirs, key_weights, key_phases = jax.random.split(key, 4)
    uniform_samples = jax.random.uniform(key_mags, (N_fourier_modes,), dtype=jnp.float32)
    k_magnitudes = (1 - uniform_samples) ** (-1.0 / alpha_fourier_exponent)
    random_directions = jax.random.normal(key_dirs, (N_fourier_modes, V_input_dim), dtype=jnp.float32)
    unit_directions = random_directions / jnp.linalg.norm(random_directions, axis=1, keepdims=True)
    k_vectors = jnp.round(k_magnitudes[:, None] * unit_directions)
    initial_weights = jax.random.normal(key_weights, (N_fourier_modes,), dtype=jnp.float32)
    normalized_weights = initial_weights / jnp.linalg.norm(initial_weights)
    phase_shifts = (jnp.pi / 2.0) * jax.random.bernoulli(key_phases, p=0.5, shape=(N_fourier_modes,))
    return FourierTargetParameters(
        k_vectors=k_vectors, weights=normalized_weights, phase_shifts=phase_shifts
    )

@jax.jit
def _calculate_fourier_target(x_batch: jax.Array,
                              fourier_params: FourierTargetParameters) -> jax.Array:
    """Computes the target values based on pre-defined Fourier parameters."""
    k_vectors, weights, phase_shifts = fourier_params.k_vectors, fourier_params.weights, fourier_params.phase_shifts
    batch_phases = 2 * jnp.pi * jnp.einsum('bv,nv->bn', x_batch, k_vectors) + phase_shifts[None, :]
    fourier_series_terms = jnp.sqrt(2) * jnp.cos(batch_phases)
    y_values = jnp.sum(weights[None, :] * fourier_series_terms, axis=1, keepdims=True)
    return y_values

def _make_fourier_loader(
    key_for_x_sampling: jax.Array,
    V_input_dim: int,
    B_batch_size: int,
    fourier_params: FourierTargetParameters,
) -> Iterator[jax.Array]:
    """Creates an infinite data loader for Fourier regression, yielding concatenated [x,y] JAX arrays."""
    generate_y_for_batch = partial(_calculate_fourier_target, fourier_params=fourier_params)
    current_rng_key = key_for_x_sampling
    while True:
        current_rng_key, x_sampling_key = jax.random.split(current_rng_key)
        x_batch = jax.random.uniform(
            x_sampling_key, (B_batch_size, V_input_dim), minval=-0.5, maxval=0.5, dtype=jnp.float32
        )
        y_batch = generate_y_for_batch(x_batch)
        # Concatenate x and y into a single array
        batch = jnp.concatenate([x_batch, y_batch], axis=1)
        yield batch

def make_dummy_ds_loader(seq_len, batch_size):
    """Creates a dummy data loader that returns batches of zeros."""
    
    # Set a reasonable number of tokens for the dummy dataset
    n_tokens = batch_size * seq_len * 100000  # Arbitrary large number
    
    def get_batch(idx):
        # Return a batch of zeros with shape [batch_size, seq_len]
        return np.ones((batch_size, seq_len), dtype=np.uint16)
    
    return get_batch, n_tokens

def _load_data_shard(file: Path):
    header = np.fromfile(file, dtype=np.int32, count=256)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = np.empty(num_tokens, dtype=np.uint16) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len: int, shuffle_data: bool, data_sharding=None):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if shuffle_data: np.random.shuffle(files)
    assert len(files) > 0, "No files found"
    # assert batch_size % world_size == 0
    # local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    try:
      tokens, pos = _load_data_shard(next(file_iter)), 0
      n_batches = len(tokens) // (batch_size*seq_len)
      tokens = tokens[: n_batches*batch_size*seq_len].reshape(n_batches, batch_size, seq_len)
      if shuffle_data: np.random.shuffle(tokens)
    except StopIteration:
      return
    while True:
        if pos >= n_batches:
            try:
              tokens, pos = _load_data_shard(next(file_iter)), 0
              n_batches = len(tokens) // (batch_size*seq_len)
              tokens = tokens[: n_batches*batch_size*seq_len].reshape(n_batches, batch_size, seq_len)
              if shuffle_data: np.random.shuffle(tokens)
            except StopIteration:
              return
        buf = tokens[pos]
        pos += 1

        arr = jax.numpy.asarray(buf) 
        if data_sharding is not None:
            arr = jax.lax.with_sharding_constraint(arr, data_sharding)

        yield arr


# Data Loader for smaller data
def make_small_ds_loader(ds_path, split, seq_len, batch_size, data_sharding):
    """note: we assume that the dataset on the disk is already shuffled!"""

    # get num. tokens
    data = np.memmap(f'{ds_path}/{split}.bin', dtype=np.uint8, mode='r')
    n_tokens = len(data)

    def get_batch():
        idx = 0
        while True:
          # read dataset
          # using np.memmap for each batch to avoid memory leak
          data = np.memmap(f'{ds_path}/{split}.bin', dtype=np.uint8, mode='r')

          # get batch
          max_idx = n_tokens // (batch_size * seq_len)
          idx = idx % max_idx  # wrap around if idx is out of bounds
          
          start_idx = batch_size*seq_len*idx + seq_len*np.arange(batch_size)
          token_idx = start_idx[:, None] + np.arange(seq_len)[None, :] # [batch, sequence]
          batch = data[token_idx]

          arr = jax.numpy.asarray(batch) 
          if data_sharding is not None:
            arr = jax.lax.with_sharding_constraint(arr, data_sharding)

          yield arr
        
          idx += 1

    return get_batch(), n_tokens

def make_ds_loader(ds_path: str, split: str, seq_len: int, batch_size: int, shuffle_data: bool, data_sharding):
    """
    Main dispatcher for creating data loaders.
    Returns (iterator, total_items_estimate).
    For Fourier task, total_items_estimate is a large arbitrary number.
    Fourier loader yields (x,y) JAX arrays. Others yield numpy arrays.
    """
    if ds_path == "fourier":
        # Hardcoded parameters for the Fourier regression task
        V_input_dim = 8  # Use seq_len as the input dimensionality for Fourier task
        N_FOURIER_MODES = 10_000   # Default number of Fourier modes
        ALPHA_FOURIER_EXPONENT = 1  # Default alpha exponent

        # Fixed key for generating shared Fourier parameters (ensures same target function)
        fourier_params_key = jax.random.PRNGKey(42)
        shared_fourier_params = _generate_fourier_target_parameters(
            key=fourier_params_key,
            V_input_dim=V_input_dim,
            N_fourier_modes=N_FOURIER_MODES,
            alpha_fourier_exponent=ALPHA_FOURIER_EXPONENT
        )

        # Fixed, distinct keys for x-value sampling for train and test/val splits
        if split == "train":
            key_for_x_sampling = jax.random.PRNGKey(43) # Fixed key for training x samples
        elif split == "val" or split == "test":
            key_for_x_sampling = jax.random.PRNGKey(44) # Fixed key for validation/testing x samples
        else:
            raise ValueError(f"Unknown split for Fourier task: {split}. Use 'train', 'val', or 'test'.")

        loader_iterator = _make_fourier_loader(
            key_for_x_sampling=key_for_x_sampling,
            V_input_dim=V_input_dim,
            B_batch_size=batch_size,
            fourier_params=shared_fourier_params
        )
        # Estimate for total items (samples) for the infinite Fourier loader
        arbitrary_num_batches = 1e16
        total_items_estimate = batch_size * arbitrary_num_batches
        return loader_iterator, total_items_estimate
    elif "fineweb" in ds_path:
      loader = distributed_data_generator(f'{ds_path}/fineweb_{split}_*.bin', batch_size, seq_len, shuffle_data)
      total_tokens = count_total_tokens(loader)
      return distributed_data_generator(f'{ds_path}/fineweb_{split}_*.bin', batch_size, seq_len, shuffle_data, data_sharding), total_tokens
    elif "c4" in ds_path:
      filename_pattern = f'{ds_path}/*_{split}_*.bin'
      total_tokens = fast_count_total_tokens(filename_pattern, batch_size, seq_len)
      return distributed_data_generator(filename_pattern, batch_size, seq_len, shuffle_data, data_sharding), total_tokens
    else:
       return make_small_ds_loader(ds_path, split, seq_len, batch_size, data_sharding)

# Keep the slow version of count_total_tokens to be backward compatible.
# Going through the data loader changes the internal seed of numpy, which
# will affect the data order.
def count_total_tokens(data_loader):
    total_tokens = 0
    for batch in data_loader:
        # batch is assumed to be a tensor of shape (B, L)
        total_tokens += batch.size
    return total_tokens

def fast_count_total_tokens(filename_pattern: str, batch_size: int, seq_len: int) -> int:
    """Estimate dataset size by reading shard headers instead of materialising batches."""
    total_tokens = 0
    tokens_per_batch = batch_size * seq_len
    files = sorted(glob.glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {filename_pattern}")
    for file in files:
        header = np.fromfile(file, dtype=np.int32, count=3)
        if header.size < 3:
            continue
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        num_tokens = int(header[2])
        total_tokens += (num_tokens // tokens_per_batch) * tokens_per_batch
    return total_tokens

def make_loader(ds_path, seq_len, batch_size, shuffle_data, split, data_sharding):
    if ds_path is None:
        return make_dummy_ds_loader(seq_len, batch_size)
    else:
        return make_ds_loader(ds_path, split, seq_len, batch_size, shuffle_data, data_sharding)


def get_in_out(batch: jax.Array, pad_id: int = 0, task: str = "ntp"):
    """Returns input, output, and weights for a batch of examples."""
    # Assumes input of the form <BOS> <IDs> <EOS> for eval.
    # in our datasets, <BOS> is 0, same as pad_id.
    # by masking out loss on pad_id, we also mask out loss on <BOS> as wanted.
    if task == "ntp":
        x = batch # [B, L]
        y = jnp.pad(x[:, 1:], ((0, 0), (0, 1)), constant_values=pad_id) # shift x by 1 along L axis
        weights = jnp.where(y != pad_id, 1, 0).astype(jnp.float32)
        return x, y, weights
    elif task == "regression":
        # regression task
        x, y = batch[:, :-1], batch[:, -1:]
        weights = jnp.ones_like(y)
        return x, y, weights
    else:
        raise ValueError(f"Unknown task: {task}. Use 'ntp' or 'regression'.")
