
# Core Config Parameters

We describe a few core config parameters related to our scaling parametrizations and general training setup.

## $\mu P$ Learning Rate Scaling
```bash
opt.mup=true
opt.scale_eps=true
opt.depth_mup=true
```

## 1/width Weight Decay Scaling
```bash
# To enable `1/width` weight decay scaling, set
opt.wdxD="optimal weight decay at base width * base width"
# For no weight decay scaling, set
opt.weight_decay=wd
```

## Perturb Optimal Hyperparameters

Set `custom_ablation_str` to one of the supported ablation strings: "double_lr", "half_lr", "double_wd", "half_wd", "fix_warmup_ratio".


## Model Checkpointing

We train our models in spot TPU machines, so it's necessary to save the intermediate models. We have the following config parameters to set up checkpointing.

- **ckpt:** whether to checkpoint the intermediate model during training
- **base_model_checkpoint_path:** location (e.g. Google Cloud Storage (GCS)) to save the model weights


## Reduce Memory Footprint
- **Smaller experiments**: Decrease context length (`model.L`), model width and depth (`model.D`/`model.N`), and/or batch size (`opt.B`)
- **Gradient Checkpointing**: Gradient checkpointing doesn't reduce the efficiency much, but it saves a decent amount of memory. Recommend always setting `model.gradient_checkpointing=True`.
- **Model Sharding**: Data sharding is turned on be default. For larger models, Fully Sharded Data Parallel (FSDP) or Tensor Parallelism (TP) may be required. You can turn on FSDP by setting `model.fsdp_enabled=True`. We find that enabling FSDP slows down the current implementation of SOAP and Shampoo significantly. We are working on a more efficienct implementation. TP can be enabled by setting `model.tp_dim` to be larger than 1. Note that the TP implementation is not optimized (e.g. missing efficient cross entropy fusion)
- **Gradient accumulation**: Set `opt.B_max` smaller than `opt.B` to accumulate gradients over multiple micro-batches.
