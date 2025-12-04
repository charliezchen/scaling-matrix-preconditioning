# mlp_model.py
import jax
import jax.numpy as jnp
from flax import nnx
from omegaconf import DictConfig # Assuming DictConfig is used for configuration
from jax.sharding import Mesh # For type hinting in create_sharded_model

# ============================================================================
# Initialisers (sharding-aware)
# ----------------------------------------------------------------------------
def fsdp_init(kind: str, model_cfg: DictConfig): # model_cfg is cfg.model from main config
    """
    Initializes weights for MLP layers, aware of FSDP partitioning if enabled.
    Args:
        kind (str): Type of layer/weights to initialize ("embedding", "mlp_kernel", "readout").
        model_cfg (DictConfig): The model's configuration (e.g., cfg.model from main script)
                                expected to contain 'fsdp_enabled' and 'embed_init_std'.
    """
    # Use with_partitioning if fsdp_enabled is True in the model_cfg, otherwise, it's a no-op.
    partition_fn = nnx.with_partitioning if model_cfg.fsdp_enabled else (lambda func, _: func)

    # Define standard initializers
    variance_scaling_init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    embedding_init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")
    zeros_init = jax.nn.initializers.zeros

    # Map kind to the appropriate initializer and partitioning spec
    if kind == "embedding":
        # Embedding layer: [VocabSize, HiddenDim], partitioned on HiddenDim if FSDP
        return partition_fn(embedding_init, (None, "data"))
    elif kind == "mlp_kernel":
        # MLP hidden layer kernel: [HiddenDim, HiddenDim], partitioned on HiddenDim (input) if FSDP
        return partition_fn(variance_scaling_init, ("data", None))
    elif kind == "readout":
        # Readout layer kernel: [HiddenDim, OutputDim], partitioned on HiddenDim (input) if FSDP
        return partition_fn(zeros_init, ("data", None))
    else:
        raise ValueError(f"Unknown initialization kind: {kind}")

# ============================================================================
# µP-aware, bias-free MLP Definition
# ----------------------------------------------------------------------------
class MLPBlock(nnx.Module):
    """A single block in the MLP, typically involving a linear transformation and activation."""
    def __init__(self, model_cfg: DictConfig, rngs: nnx.Rngs):
        """
        Args:
            model_cfg (DictConfig): The model's configuration (e.g., cfg.model).
                                   Expected to contain 'D' (hidden_dim), 'N' (depth),
                                   and 'depth_mup'.
            rngs (nnx.Rngs): NNX PRNG keys.
        """
        self.fc = nnx.Linear(
            in_features=model_cfg.D,
            out_features=model_cfg.D,
            use_bias=False,
            kernel_init=fsdp_init("mlp_kernel", model_cfg),
            rngs=rngs
        )
        # Explicitly zero-initialize hidden block kernels, as per user's confirmation.
        # This overrides the "mlp_kernel" (variance_scaling) initializer for these specific layers.
        nnx.update(self.fc.kernel, jnp.zeros_like(self.fc.kernel))

        # Depth scaling multiplier for µP
        self.depth_multiplier = 1.0 / (model_cfg.N / 3) if model_cfg.depth_mup else 1.0

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the MLP block with a residual connection."""
        # Residual connection: x + GELU(Linear(x)) * depth_multiplier
        transformed_x = self.fc(x)
        activated_x = jax.nn.gelu(transformed_x)
        return x + activated_x * self.depth_multiplier


class MLPRegression(nnx.Module):
    """Full MLP model for regression tasks."""
    def __init__(self, model_cfg: DictConfig, rngs: nnx.Rngs):
        """
        Args:
            model_cfg (DictConfig): The model's configuration (e.g., cfg.model).
                                   Expected to contain 'V' (input_dim), 'D' (hidden_dim),
                                   'N' (depth).
            rngs (nnx.Rngs): NNX PRNG keys.
        """
        # Embedding layer (input transformation)
        self.embed = nnx.Linear(
            in_features=model_cfg.V, # Input dimension
            out_features=model_cfg.D,
            use_bias=False,
            kernel_init=fsdp_init("embedding", model_cfg),
            rngs=rngs
        )

        # Hidden blocks: (N-1) blocks since N includes the initial embedding transformation
        # as one of the "transforming" layers in terms of depth counting for the multiplier.
        self.blocks = [MLPBlock(model_cfg, rngs) for _ in range(model_cfg.N)]

        # Readout layer (maps hidden dimension to a single output for regression)
        self.readout = nnx.Linear(
            in_features=model_cfg.D,
            out_features=1, # Single output for regression
            use_bias=False,
            kernel_init=fsdp_init("readout", model_cfg), # Zero-initialized
            rngs=rngs
        )

    def get_features(self, x: jax.Array) -> jax.Array:
        """Extracts features from the MLP before the final readout layer."""
        # Initial transformation and activation
        h = self.embed(x)
        # Pass through all MLP blocks
        for block in self.blocks:
            h = block(h)
        return h # These are the features input to the readout layer

    @nnx.jit
    def get_features_and_logits(self, x):
        h = self.get_features(x)
        return h, self.readout(h)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Performs a forward pass through the MLP."""
        features = self.get_features(x)
        return self.readout(features)


def create_sharded_model(model_cfg: DictConfig, mesh: Mesh, seed: int) -> MLPRegression:
    """
    Creates, initializes, and shards an MLPRegression model.
    Args:
        model_cfg (DictConfig): The model's configuration (e.g., cfg.model).
        mesh (Mesh): The JAX device mesh for sharding.
        seed (int): PRNG seed for model initialization.
    Returns:
        MLPRegression: The sharded MLP model instance.
    """
    @nnx.jit # JIT compile the initialization function for efficiency
    def _initialize_and_shard_model():
        # nnx.Rngs expects a mapping; 'params' is a common key for parameter initialization PRNGs.
        model_rngs = nnx.Rngs(params=jax.random.key(seed))
        
        # Instantiate the model
        model = MLPRegression(model_cfg, model_rngs)
        
        # Get the model's state (parameters) and their sharding specifications
        model_state = nnx.state(model)
        partition_specs = nnx.get_partition_spec(model_state)
        
        # Apply sharding constraints to the state and update the model
        sharded_state = jax.lax.with_sharding_constraint(model_state, partition_specs)
        nnx.update(model, sharded_state)
        
        return model

    # Execute the initialization and sharding within the provided device mesh context
    with mesh:
        sharded_model = _initialize_and_shard_model()
    return sharded_model