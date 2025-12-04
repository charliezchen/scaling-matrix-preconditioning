# FineWeb Experiments

## Quick Start

### Reproduce figures
```
python plots/plot_scaling_frontier.py --preset fig_6_panel_1_2
python plots/plot_scaling_frontier.py --preset fig_6_panel_3
python plots/plot_scaling_frontier.py --preset alternative_scaling
python plots/plot_scaling_frontier.py --preset tuned_tpp_panel_2_3

python plots/fit_tpp.py
python plots/hyper_ablations.py
```

## Example runs

python main.py -cn llama_fineweb opt.mup=True model.D=128 opt.name=muon




Tips:

- **µP + 1/D WD scaling**: Set `opt.mup=True`, `opt.scale_eps=True`, `opt.depth_mup=True`, and use `opt.wdxD` to automatically scale weight decay inversely with width.
- **Reduce memory**: Decrease context length (`model.L`) and/or batch size (`opt.B`). Enable `model.gradient_checkpointing=True` and/or `model.fsdp_enabled=True`.
- **Gradient accumulation**: Set `opt.B_max` smaller than `opt.B` to accumulate gradients over multiple micro-batches.
