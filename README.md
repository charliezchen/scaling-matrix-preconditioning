# FineWeb Experiments

Official codebase for Section 4 and Appendix H of **Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales**.

## Quick Start

### Reproduce figures
```bash
# Figure 6, left and middle figure
python plots/plot_scaling_frontier.py --preset fig_6_panel_1_2
# Figure 6, right figure
python plots/plot_scaling_frontier.py --preset fig_6_panel_3
# Figure 11
python plots/plot_scaling_frontier.py --preset alternative_scaling
# Figure 12
python plots/hyper_ablations.py
# Figure 13, middle and right figure
python plots/plot_scaling_frontier.py --preset tuned_tpp_panel_2_3
# Figure 13, left figure
python plots/fit_tpp.py
```

## Download dataset
To download the 30B tokens of fineweb dataset, run the following command
```
python cached_fineweb100B.py 300
```

To reproduce our runs, set `shuffle_data=true` (default), `seed=0` (default) and download exactly 30B tokens to ensure the validation set is the same set as we use for our experiments.


## Example runs

```bash
python main.py -cn llama_fineweb opt.mup=True model.D=128 opt.name=muon
```

All the sweep views in Weights&Biases can be found in Appendix H of the paper.

To see how to run the sweeps and understand some core config parameters, see [the README in sweep/](sweep/README.md)




Tips:

- **µP + 1/D WD scaling**: Set `opt.mup=True`, `opt.scale_eps=True`, `opt.depth_mup=True`, and use `opt.wdxD` to automatically scale weight decay inversely with width.
- **Reduce memory**: Decrease context length (`model.L`) and/or batch size (`opt.B`). Enable `model.gradient_checkpointing=True` and/or `model.fsdp_enabled=True`.
- **Gradient accumulation**: Set `opt.B_max` smaller than `opt.B` to accumulate gradients over multiple micro-batches.
