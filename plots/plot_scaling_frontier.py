#!/usr/bin/env python3
"""
Scaling frontier visualization for optimizer comparison.

Usage:
    python plots/plot_scaling_frontier.py --preset fig_6_panel_1_2
    python plots/plot_scaling_frontier.py --preset fig_6_panel_3
    python plots/plot_scaling_frontier.py --preset alternative_scaling
    python plots/plot_scaling_frontier.py --preset tuned_tpp_panel_2_3
    python plots/plot_scaling_frontier.py --list-presets

For more options:
    python plots/plot_scaling_frontier.py --help
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d

os.environ["TPU_VISIBLE_DEVICES"] = ""

# =============================================================================
# Constants
# =============================================================================

COMPUTE_COL = "compute"
WANDB_PROJECT = "llama_proj"
WANDB_BASE_FILTER = {"state": "finished", "config.wandb_tag": "llama_fineweb_v3"}
CONFIG_KEYS = ["opt/lr", "opt/name", "opt/mup", "model/D", "token_per_param", "custom_ablation_str"]
SUMMARY_KEYS = ["eval_loss", "compute", "non_embed_compute", "_runtime"]
HISTORY_KEYS = ["compute", "non_embed_compute", "tokens", "eval_loss"]
HISTORY_SAMPLES = 1_000

USE_MUP_BASE_MODEL_ONLY = True
BASE_MODEL_D = 512

OPTIMIZER_RUN_FILTERS = {
    "adam": {
        "mup": {"custom_sweep": "oct21_adam_mup_scaling"},
        "sp": {"custom_sweep": "oct21_adam_sp_scaling"},
        "lr_only": [
            {"custom_sweep": "oct23_adam_mup_lr_only"},
            {"run_id": "9c9b0f39"},
        ],
        "wd_only": {"custom_sweep": "oct31_adam_wd_scaling_only"},
        "co": {"run_id": ["b3e1d3d5", "c9a874c7", "a134b10f", "05d32060"]},
    },
    "muon": {
        "mup": {"custom_sweep": "oct21_muon_mup_scaling"},
        "sp": {"custom_sweep": "oct21_muon_sp_scaling"},
        "lr_only": [
            {"custom_sweep": "oct23_muon_mup_lr_only"},
            {"run_id": "baca9609"},
        ],
        "wd_only": {"custom_sweep": "oct31_muon_wd_scaling_only"},
        "inc_mup": {"custom_sweep": "oct23_muon_inc_mup_scaling_v2"},
        "co": {"id": ["115af832", "085bed41", "e4d1d100", "2zqcavvi"]},
    },
    "norm_muon": {
        "mup": {"custom_sweep": "dec18_spectral_muon_adam_scaling"},
    },
    "shampoo": {
        "mup": [
            {"custom_sweep": "nov28_shampoo_mup_with_abl", "custom_ablation_str": "None", "opt/block_size": 512},
            {"custom_sweep": "nov27_shampoo_xl", "opt/mup": True, "opt/block_size": 512},
        ],
        "sp": [
            {"custom_sweep": "nov24_shampoo_sp", "opt/block_size": 512},
            {"custom_sweep": "nov27_shampoo_xl", "opt/mup": False, "opt/block_size": 512},
        ],
        "lr_only": {"custom_sweep": "nov30_shampoo_no_wd_scaling"},
        "wd_only": {"custom_sweep": "nov30_shampoo_wd_only_scaling"},
    },
    "norm_shampoo": {
        "mup": [{"custom_sweep": "dec15_spectral_shampoo_scaling"},
                {"custom_sweep": "jan11_spectral_shampoo_xl"}],
    },
    "norm_soap": {
        "mup": [
            {"run_id": ["24614bf8", "2bd4aa55"]},
            {"custom_sweep": "nov28_soap_scaling_v2", "custom_ablation_str": "None", "opt/block_size": 512},
        ],
        "lr_only": [{"custom_sweep": "nov30_soap_no_wd_scaling"}],
        "wd_only": [{"custom_sweep": "nov30_soap_wd_only_scaling"}],
    },
    "soap": {
        "mup": {"id": ["b2222569", "d5a39aad", "d514d499", "61a7f710"]},
        "sp": {"id": ["b2222569", "9648b658", "4d1a516a", "td6nxv61"]},
        "lr_only": {"id": ["b2222569", "6d700075", "8751106d"]},
        "wd_only": {"id": ["b2222569", "f7bbf2a7", "87bef55b"]}
    },
}

COMPUTE_MULTIPLIER_SERIES = [("adam", "mup")]
COMPUTE_MULTIPLIER_BASELINES = defaultdict(lambda: ("adam", "mup"))
MODEL_SIZE_LABELS = ["190M", "380M", "640M", "1.4B"]

KIND_LABELS = {
    "sp": "SP",
    "mup": "$\\mu$P + 1/Width WD",
    "co": "Compute Optimal",
    "lr_only": "$\\mu$P only",
    "wd_only": "1/Width WD only",
    "inc_mup": "1/Width LR&WD",
}

DISPLAY_NAMES = {
    "muon": "Muon", 
    "norm_muon": "Muon (norm)", 
    "adam": "Adam", 
    "shampoo": "Shampoo", 
    "norm_shampoo": "Shampoo (norm)", 
    "norm_soap": "SOAP (norm)", 
    "soap": "SOAP"}

OPT_NAME_MAP = {
    "muon_adam": "muon",
    "spectral-muon_adam": "norm_muon",
    "adam": "adam",
    "grafted_shampoo2_adam": "shampoo",
    "spectral-grafted_shampoo2_adam": "norm_shampoo",
    "spectral-grafted_shampoo2_manual_adam": "norm_shampoo",
    "spectral-soap": "norm_soap",
    "soap": "soap",
    "soap_manual": "soap",
}

OPT_COLORS = {
    "adam": "#0083CB",
    "muon": "#F45A12",
    "norm_muon": "#D07D57",
    "shampoo": "#00A879",
    "norm_shampoo": "#66C5AAFF",
    "soap": "#C313C4",
    "norm_soap": "#CC79A7"
}

NORM_COLORS = {
    "no_norm": "#0083CB",
    "spectral_norm": "#F45A12",
}

NORM_LABELS = {
    "no_norm": "$\\mu$P",
    "spectral_norm": "Norm",
}

KIND_COLORS = {
    "sp": "#0072B2",
    "mup": "#D55E00",
    "co": "#009E73",
    "lr_only": "#E69F00",
    "wd_only": "#CC79A7",
    "inc_mup": "#56B4E9",
}


# =============================================================================
# Configuration Presets
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration for a plotting preset."""
    separate_plot: bool = False
    kinds_to_visualize: list = field(default_factory=lambda: ["mup", "sp"])
    optim_to_visualize: list = field(default_factory=lambda: ["adam", "muon"])
    legends_to_show: Optional[list] = None
    skip_pairs: Optional[list] = None
    plot_legend: bool = False
    legend_loss: Optional[str] = None
    legend_multiplier: Optional[str] = None
    legend_mode: Optional[str] = None
    legend_on_subplot: Optional[tuple[int, int]] = None
    color_mode: Optional[str] = None
    column_groups: Optional[list[tuple[str, list[str]]]] = None
    subplot_wspace: float = 0.15
    subplot_hspace: float = 0.15
    show_model_size: bool = True
    show_trajectories: bool = False
    y_min: float = 0.8
    y_max: float = 1.4
    markersize: int = 10
    edgewidth: int = 1
    loss_alpha_mup: float = 0.95
    loss_alpha_other: float = 0.6
    line_width_scale: float = 1.0


PRESETS = {
    "all_mup": PlotConfig(
        separate_plot=True,
        kinds_to_visualize=["mup"],
        optim_to_visualize=["adam", "shampoo", "norm_shampoo", "soap", "muon", "norm_muon", "norm_soap"],
        plot_legend=False,
        show_model_size=True,
        y_min=0.8,
        y_max=1.4,
    ),
    "fig_6_panel_1_2": PlotConfig(
        separate_plot=True,
        kinds_to_visualize=["sp", "mup"],
        optim_to_visualize=["adam", "shampoo", "muon", 'soap'],
        skip_pairs=[("adam", "sp")],
        plot_legend=True,
        legend_loss="optimizer",
        legend_multiplier="linestyle",
        show_model_size=True,
        y_min=0.8,
        y_max=1.4,
        line_width_scale=0.8,
    ),
    "fig_6_panel_1_2_norm": PlotConfig(
        separate_plot=True,
        kinds_to_visualize=["sp", "mup"],
        optim_to_visualize=["adam", "norm_shampoo", "norm_muon", 'norm_soap'],
        skip_pairs=[("adam", "sp")],
        plot_legend=True,
        show_model_size=True,
        y_min=0.8,
        y_max=1.4,
        line_width_scale=0.8,
    ),
    "fig_6_panel_3": PlotConfig(
        separate_plot=True,
        kinds_to_visualize=["sp", "lr_only", "wd_only", "inc_mup", "mup"],
        optim_to_visualize=["adam", "muon"],
        legends_to_show=["inc_mup", "wd_only", "lr_only"],
        skip_pairs=[("adam", "sp"), ("adam", "lr_only"), ("adam", "wd_only"), ("adam", "inc_mup")],
        plot_legend=True,
        show_model_size=True,
        y_min=0.8,
        y_max=1.4,
        line_width_scale=0.8,
    ),
    "alternative_scaling": PlotConfig(
        separate_plot=False,
        kinds_to_visualize=["inc_mup", "lr_only", "wd_only", "sp", "mup"],
        optim_to_visualize=["adam", "muon", "shampoo", "soap"],
        legends_to_show=None,
        skip_pairs=None,
        plot_legend=False,
        show_model_size=False,
        y_min=0.4,
        y_max=1.6,
        loss_alpha_mup=1.0,
        loss_alpha_other=1.0,
    ),
    "tuned_tpp_panel_2_3": PlotConfig(
        separate_plot=True,
        kinds_to_visualize=["co", "mup"],
        optim_to_visualize=["adam", "muon"],
        legends_to_show=None,
        skip_pairs=None,
        plot_legend=False,
        show_model_size=False,
        y_min=0.8,
        y_max=1.6,
    ),
    "compare_spectral_norm": PlotConfig(
        separate_plot=False,
        kinds_to_visualize=["mup"],
        optim_to_visualize=["soap", "norm_soap", "muon", "norm_muon", "shampoo", "norm_shampoo"],
        column_groups=[
            ("SOAP", ["soap", "norm_soap"]),
            ("Muon", ["muon", "norm_muon"]),
            ("Shampoo", ["shampoo", "norm_shampoo"]),
        ],
        plot_legend=False,
        legend_mode="norm",
        legend_on_subplot=(0, 0),
        color_mode="norm",
        show_model_size=False,
        y_min=0.8,
        y_max=1.6,
        subplot_wspace=0.05,
        subplot_hspace=0.08,
    ),
}


def is_norm_opt(opt_name: str) -> bool:
    return opt_name.startswith("norm_")


def resolve_color_mode(config: PlotConfig) -> str:
    if config.color_mode:
        return config.color_mode
    return "optimizer" if config.separate_plot else "kind"


def get_colors(config: PlotConfig) -> dict:
    """Get color mapping based on configuration."""
    colors = {}
    color_mode = resolve_color_mode(config)
    for opt in config.optim_to_visualize:
        for kind in config.kinds_to_visualize:
            key = (opt, kind)
            if color_mode == "optimizer":
                colors[key] = OPT_COLORS.get(opt, "#000000")
            elif color_mode == "kind":
                colors[key] = KIND_COLORS.get(kind, "#000000")
            elif color_mode == "norm":
                norm_key = "spectral_norm" if is_norm_opt(opt) else "no_norm"
                colors[key] = NORM_COLORS[norm_key]
            else:
                raise ValueError(f"Unknown color mode: {color_mode}")
    return colors


def get_linestyles(config: PlotConfig) -> dict:
    """Get linestyle mapping based on configuration."""
    if config.separate_plot:
        return {"sp": "--", "mup": "-", "lr_only": "-.", "wd_only": "-.", "co": ":", "inc_mup": "-."}
    return defaultdict(lambda: "-")


def get_markers(config: PlotConfig) -> dict:
    """Get marker mapping based on configuration."""
    if config.separate_plot:
        return {"sp": "v", "mup": "o", "co": "^", "lr_only": "d", "wd_only": "P", "inc_mup": "X"}
    return defaultdict(lambda: "o")


# =============================================================================
# Data Loading
# =============================================================================

def normalize_filter_dict(filter_dict: dict) -> dict:
    if "id" in filter_dict:
        return filter_dict
    filter = dict(WANDB_BASE_FILTER)
    for key in filter_dict:
        if isinstance(filter_dict[key], list):
            filter.update({f"config.{key}": {"$in": filter_dict[key]}})
        else:
            filter.update({f"config.{key}": filter_dict[key]})
    return filter


def filter_cache_key(filter_dict: dict) -> str:
    return json.dumps(filter_dict, sort_keys=True)


def norm_opt_name(raw: str) -> str:
    key = str(raw).lower()
    return OPT_NAME_MAP.get(key, key.split("_")[0])


def fetch_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    history = run.history(keys=HISTORY_KEYS, x_axis="step", samples=HISTORY_SAMPLES)
    frame = pd.DataFrame(history)
    return frame.sort_values(COMPUTE_COL).reset_index(drop=True)


def fetch_runs(filters: dict) -> pd.DataFrame:
    api = wandb.Api()
    rows = []
    runs = []
    if "id" in filters:
        assert len(filters.keys()) == 1, "When filtering by id, no other filters should be present"
        ids = filters.pop("id")
        if not isinstance(ids, list):
            ids = [ids]
        for run_id in ids:
            run = api.run(f"{WANDB_PROJECT}/{run_id}")
            runs.append(run)
    else:
        for run in api.runs(WANDB_PROJECT, filters=filters, order="-created_at"):
            runs.append(run)

    for run in runs:
        summary = run.summary._json_dict
        if not summary:
            continue
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        row = {key: config[key] for key in CONFIG_KEYS}
        for key in SUMMARY_KEYS:
            row[key] = summary[key]
        row["run_id"] = getattr(run, "id", None)
        row["history"] = fetch_history(run)
        rows.append(row)
    assert len(rows) > 0, f"No runs found with the specified filters {filters}"
    df = pd.DataFrame(rows)
    duplicates = df.duplicated(subset=CONFIG_KEYS, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate runs for keys:\n{df.loc[duplicates, CONFIG_KEYS]}")
    return df.replace({None: -1})


def apply_mup_base_policy(df: pd.DataFrame) -> pd.DataFrame:
    if not USE_MUP_BASE_MODEL_ONLY:
        return df
    target_mask = (df["kind"] != "co") & (df["kind"] != "mup") & (df["D"] == BASE_MODEL_D)
    df = df.loc[~target_mask]
    mup_base_rows = df[(df["D"] == BASE_MODEL_D) & (df["kind"] == "mup")]
    assert len(mup_base_rows) == len(df["opt"].unique()), \
        f"Expected one base mup run per optimizer, {mup_base_rows['opt'].unique()}"
    for kind in df["kind"].unique():
        if kind in ("co", "mup"):
            continue
        for _, row in mup_base_rows.iterrows():
            new_row = row.copy()
            new_row["kind"] = kind
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def load_all_runs(config: PlotConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fetch_cache: dict[str, pd.DataFrame] = {}

    optim_names = [o[0] for o in COMPUTE_MULTIPLIER_SERIES]
    kind_names = [o[1] for o in COMPUTE_MULTIPLIER_SERIES]

    for opt_name in set(config.optim_to_visualize + optim_names):
        for kind in set(config.kinds_to_visualize + kind_names):
            if opt_name not in OPTIMIZER_RUN_FILTERS or kind not in OPTIMIZER_RUN_FILTERS[opt_name]:
                print(f"No run filters configured for optimizer='{opt_name}' and kind='{kind}'")
                continue
            raw_filter = OPTIMIZER_RUN_FILTERS[opt_name][kind]
            if not isinstance(raw_filter, list):
                raw_filter = [raw_filter]
            sources = []
            for single_filter in raw_filter:
                normalized_filter = normalize_filter_dict(single_filter)
                print(normalized_filter)
                cache_key = filter_cache_key(normalized_filter)
                if cache_key not in fetch_cache:
                    fetch_cache[cache_key] = fetch_runs(normalized_filter)
                source = fetch_cache[cache_key]
                sources.append(source.copy())
            frame = pd.concat(sources, ignore_index=True)
            normalized_opt = frame["opt/name"].map(norm_opt_name)
            filtered = frame[normalized_opt == opt_name].copy()
            if kind in ("mup", "sp"):
                desired_flag = kind == "mup"
                filtered = filtered[filtered["opt/mup"].astype(bool) == desired_flag]
            if filtered.empty:
                raise ValueError(f"No runs found for optimizer='{opt_name}' and kind='{kind}'")
            filtered["kind"] = kind
            frames.append(filtered)

            print(f"Loaded {len(filtered)} runs for optimizer='{opt_name}' and kind='{kind}'")
            print(filtered["run_id"].tolist())

    if not frames:
        raise ValueError("No runs were loaded from the optimizer filter configuration")
    df = pd.concat(frames, ignore_index=True)
    df["hours"] = df["_runtime"] / 3600
    df["flops"] = df[COMPUTE_COL]
    df["opt"] = df["opt/name"].map(norm_opt_name)
    df["mup"] = df["opt/mup"].astype(bool)
    df["D"] = df["model/D"]
    df = apply_mup_base_policy(df)
    return df


# =============================================================================
# Compute Multiplier
# =============================================================================

def collect_multiplier_samples(df: pd.DataFrame) -> dict[tuple[str, str], interp1d]:
    required_pairs: set[tuple[str, str]] = set(COMPUTE_MULTIPLIER_SERIES)
    required_pairs.update(COMPUTE_MULTIPLIER_BASELINES.values())
    estimate_fns: dict[tuple[str, str], interp1d] = {}

    for opt_name, kind in sorted(required_pairs):
        mask = (df["opt"] == opt_name) & (df["kind"] == kind)
        subset = df.loc[mask]
        if subset.empty:
            raise ValueError(f"No runs found for compute multiplier pair {(opt_name, kind)}")
        valid = subset[(subset["flops"] > 0) & (subset["eval_loss"] > 0)]
        if len(valid) < 2:
            raise ValueError(f"Need at least two valid runs for {(opt_name, kind)} to build an interpolator")

        losses = valid["eval_loss"].to_numpy(dtype=float)
        compute = valid["flops"].to_numpy(dtype=float)
        sort_idx = np.argsort(losses)
        losses_sorted = losses[sort_idx]
        compute_sorted = compute[sort_idx]
        unique_losses, unique_idx = np.unique(losses_sorted, return_index=True)
        compute_unique = compute_sorted[unique_idx]
        if len(unique_losses) < 2:
            raise ValueError(f"Need at least two unique loss values for {(opt_name, kind)} to interpolate")
        interpolation = interp1d(
            np.log(unique_losses),
            np.log(compute_unique),
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        def fn(x, interp=interpolation):
            return np.exp(interp(np.log(x)))

        estimate_fns[(opt_name, kind)] = fn
        print(f"Built linear loss→compute interpolation for {(opt_name, kind)} using {len(unique_losses)} points")

    return estimate_fns


# =============================================================================
# Plotting
# =============================================================================

def configure_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 2.0,
            "lines.linewidth": 2,
            "grid.alpha": 0.3,
            "figure.figsize": (8, 6),
        },
    )
    sns.set_context("paper", font_scale=2.5)


def build_legend_handles(config: PlotConfig, mode: Optional[str] = None) -> list:
    handles = []
    if mode == "norm":
        return [
            Patch(facecolor=NORM_COLORS["no_norm"], edgecolor="none", label=NORM_LABELS["no_norm"]),
            Patch(facecolor=NORM_COLORS["spectral_norm"], edgecolor="none", label=NORM_LABELS["spectral_norm"]),
        ]
    if config.separate_plot:
        color_handles = [
            Patch(facecolor=OPT_COLORS[name], edgecolor="none", label=DISPLAY_NAMES[name])
            for name in config.optim_to_visualize[::-1]
            if config.legends_to_show is None or name in config.legends_to_show
        ]
        markers = get_markers(config)
        linestyles = get_linestyles(config)
        linestyle_handles = [
            Line2D(
                [0], [0],
                linestyle=linestyles[kind],
                color="black",
                linewidth=2 * config.line_width_scale,
                label=KIND_LABELS[kind],
                marker=markers[kind],
                markersize=config.markersize,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=config.edgewidth + 0.5,
            )
            for kind in config.kinds_to_visualize[::-1]
            if config.legends_to_show is None or kind in config.legends_to_show
        ]
        if mode is None or mode == "all":
            handles = color_handles + linestyle_handles
        elif mode == "optimizer":
            handles = color_handles
        elif mode == "linestyle":
            handles = linestyle_handles
        else:
            raise ValueError(f"Unknown legend mode: {mode}")
    else:
        handles = [
            Patch(facecolor=KIND_COLORS[kind], edgecolor="none", label=KIND_LABELS[kind])
            for kind in config.kinds_to_visualize[::-1]
            if config.legends_to_show is None or kind in config.legends_to_show
        ]
    return handles


def extract_model_size_ticks(group: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ordered = group.sort_values("D")
    ticks = ordered["flops"].to_numpy(dtype=float)
    count = min(len(ticks), len(MODEL_SIZE_LABELS))
    return ticks[:count], MODEL_SIZE_LABELS[:count]


def add_model_size_axis(ax: plt.Axes, group: pd.DataFrame) -> None:
    ticks, labels = extract_model_size_ticks(group)
    if len(ticks) == 0:
        return
    top_ax = ax.twiny()
    top_ax.set_xscale("log")
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks(ticks)
    top_ax.set_xticklabels(labels)
    top_ax.tick_params(axis="x", which="major", direction="in", pad=-32)
    top_ax.tick_params(axis="x", which="minor", direction="in", pad=-32)
    top_ax.xaxis.set_minor_locator(ticker.NullLocator())
    top_ax.grid(False, axis="x")


def plot_loss_vs_compute(df: pd.DataFrame, opt_to_axes: dict, config: PlotConfig) -> None:
    colors = get_colors(config)
    linestyles = get_linestyles(config)
    markers = get_markers(config)

    for kind in config.kinds_to_visualize:
        for opt in config.optim_to_visualize:
            group = df[(df["opt"] == opt) & (df["kind"] == kind)]
            pair = (opt, kind)
            if config.skip_pairs is not None and pair in config.skip_pairs:
                continue

            ax = opt_to_axes[opt]
            color = colors[pair]
            linestyle = linestyles[kind]
            marker = markers[kind]
            label = KIND_LABELS[kind]
            ordered = group.sort_values("flops")

            if config.show_trajectories:
                for _, row in ordered.iterrows():
                    history = row["history"]
                    ax.plot(
                        history[COMPUTE_COL],
                        history["eval_loss"],
                        lw=1.2 * config.line_width_scale,
                        alpha=0.4,
                        color=color,
                    )

            ax.plot(
                ordered["flops"],
                ordered["eval_loss"],
                linestyle=linestyle,
                color=color,
                alpha=config.loss_alpha_mup if kind == "mup" else config.loss_alpha_other,
                label=f"{DISPLAY_NAMES[opt]} ({label})",
                lw=(2.5 if kind == "mup" else 1.8) * config.line_width_scale,
                marker=marker,
                markeredgecolor="black",
                markeredgewidth=config.edgewidth,
                markersize=config.markersize if kind == "mup" else config.markersize + 1.5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    if config.separate_plot:
        compute_label = "Compute" if COMPUTE_COL == "compute" else "Non-embed compute"
        ax.set_xlabel(f"{compute_label} (FLOPs)")
        ax.set_ylabel("Loss")
        ax.set_xlim(2_700 * 1e15, 330_000 * 1e15)

    if config.plot_legend:
        legend_mode = config.legend_loss
        legend_handles = build_legend_handles(config, mode=legend_mode)
        legend = ax.legend(handles=legend_handles, loc="lower left", frameon=False, ncol=1, fontsize=20)
        ax.add_artist(legend)

    y_min, y_max = 2.6, 3.2
    ax.set_ylim(y_min - 0.1, y_max + 0.2)
    ax.set_yticks(np.linspace(y_min, y_max, num=round((y_max - y_min) / 0.2 + 1)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    if config.show_model_size:
        if config.separate_plot:
            main_ax = next(iter(opt_to_axes.values()))
            group = df[(df["opt"] == "adam") & (df["kind"] == "mup")]
            if group.empty:
                for opt in config.optim_to_visualize:
                    group = df[(df["opt"] == opt) & (df["kind"] == "mup")]
                    if not group.empty:
                        break
            if not group.empty:
                add_model_size_axis(main_ax, group)
        else:
            axis_to_opts = defaultdict(list)
            for opt, ax in opt_to_axes.items():
                axis_to_opts[ax].append(opt)
            for ax, opts in axis_to_opts.items():
                group = pd.DataFrame()
                for opt in opts:
                    group = df[(df["opt"] == opt) & (df["kind"] == "mup")]
                    if not group.empty:
                        break
                if not group.empty:
                    add_model_size_axis(ax, group)


def plot_compute_multiplier(
    df: pd.DataFrame,
    loss_to_compute: dict[tuple[str, str], interp1d],
    opt_to_ax: dict,
    config: PlotConfig,
) -> None:
    colors = get_colors(config)
    linestyles = get_linestyles(config)
    markers = get_markers(config)

    def lam_interp(sample: dict, func: interp1d) -> float:
        estimate = float(func(sample["eval_loss"]))
        return estimate / sample["flops"]

    df["FLOPS"] = df["flops"].map(lambda x: float(f"{x:.2g}"))

    for kind in config.kinds_to_visualize:
        for opt in config.optim_to_visualize:
            ax = opt_to_ax[opt]
            pair = (opt, kind)
            if config.skip_pairs is not None and pair in config.skip_pairs:
                continue
            color = colors[pair]
            ref_pair = COMPUTE_MULTIPLIER_BASELINES[pair]
            baseline_func = loss_to_compute[ref_pair]

            group = df[(df["opt"] == opt) & (df["kind"] == kind)]
            if group.empty:
                continue

            ordered = group.sort_values("D")
            xs = ordered["flops"].to_numpy(dtype=float)
            samples = [{"eval_loss": row["eval_loss"], "flops": row["flops"]} for _, row in ordered.iterrows()]
            ys = np.array([lam_interp(sample, baseline_func) for sample in samples], dtype=float)
            ax.plot(
                xs,
                ys,
                marker=markers[kind],
                linestyle=linestyles[kind],
                color=color,
                markersize=config.markersize,
                linewidth=(2 if kind == "mup" else 2.3) * config.line_width_scale,
                markeredgecolor="black",
                markeredgewidth=config.edgewidth,
            )

        ax.set_xscale("log")

        if config.separate_plot:
            compute_label = "Compute" if COMPUTE_COL == "compute" else "Non-embed compute"
            ax.set_xlabel(f"{compute_label} (FLOPs)")
            ax.set_ylabel("Compute multiplier")
            ax.set_xlim(2_700 * 1e15, 330_000 * 1e15)
            ax.set_ylim(config.y_min - 0.15, config.y_max + 0.20)
            ax.set_yticks(np.linspace(config.y_min, config.y_max, num=round((config.y_max - config.y_min) / 0.2 + 1)))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.yaxis.set_minor_locator(ticker.NullLocator())

    if config.show_model_size:
        if config.separate_plot:
            main_ax = next(iter(opt_to_ax.values()))
            group = df[(df["opt"] == "muon") & (df["kind"] == "mup")]
            if group.empty:
                for opt in config.optim_to_visualize:
                    group = df[(df["opt"] == opt) & (df["kind"] == "mup")]
                    if not group.empty:
                        break
            if not group.empty:
                add_model_size_axis(main_ax, group)
        else:
            axis_to_opts = defaultdict(list)
            for opt, ax in opt_to_ax.items():
                axis_to_opts[ax].append(opt)
            for ax, opts in axis_to_opts.items():
                group = pd.DataFrame()
                for opt in opts:
                    group = df[(df["opt"] == opt) & (df["kind"] == "mup")]
                    if not group.empty:
                        break
                if not group.empty:
                    add_model_size_axis(ax, group)

    if config.plot_legend:
        legend_mode = config.legend_multiplier
        legend_handles = build_legend_handles(config, mode=legend_mode)
        legend = ax.legend(handles=legend_handles, loc="lower left", frameon=False, ncol=1, fontsize=20)
        ax.add_artist(legend)


def create_legend_fig(config: PlotConfig, output_dir: Path) -> None:
    legend_handles = build_legend_handles(config)
    legend_fig = plt.figure()
    legend = legend_fig.legend(
        handles=legend_handles,
        ncol=int(np.ceil(len(legend_handles))),
        frameon=False,
    )
    legend_path = output_dir / f"{COMPUTE_COL}_legend.pdf"
    legend_fig.savefig(legend_path, bbox_extra_artists=[legend], bbox_inches="tight")
    plt.close(legend_fig)
    print(f"Saved legend to {legend_path}")


def generate_separate_plots(df: pd.DataFrame, loss_to_compute: dict, config: PlotConfig, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8/1.1, 6/1.1), dpi=300)
    plot_loss_vs_compute(df, {opt: ax for opt in config.optim_to_visualize}, config)
    fig.tight_layout()
    path = output_dir / f"{COMPUTE_COL}_vs_loss.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved loss vs compute plot to {path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8/1.1, 6/1.1), dpi=300)
    plot_compute_multiplier(df, loss_to_compute, {opt: ax for opt in config.optim_to_visualize}, config)
    fig.tight_layout()
    path = output_dir / f"{COMPUTE_COL}_multiplier.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved compute multiplier plot to {path}")
    plt.close(fig)


def generate_combined_plot(df: pd.DataFrame, loss_to_compute: dict, config: PlotConfig, output_dir: Path) -> None:
    groups = config.column_groups
    opt_num = len(groups) if groups else len(config.optim_to_visualize)
    fig, axs = plt.subplots(2, opt_num, figsize=(6 * opt_num / 1.5, 12 / 1.5), dpi=300, sharex=True, sharey="row")

    baseline_pair = COMPUTE_MULTIPLIER_BASELINES[COMPUTE_MULTIPLIER_SERIES[0]]
    baseline_group = df[(df["opt"] == baseline_pair[0]) & (df["kind"] == baseline_pair[1])]
    if baseline_group.empty:
        baseline_flops = np.sort(df["flops"].unique())
    else:
        baseline_flops = np.sort(baseline_group["flops"].unique())

    opt_to_ax_0 = {}
    opt_to_ax_1 = {}
    if groups:
        for i, (title, opts) in enumerate(groups):
            ax0 = axs[0, i]
            ax1 = axs[1, i]
            ax0.set_title(title)
            for opt in opts:
                opt_to_ax_0[opt] = ax0
                opt_to_ax_1[opt] = ax1

            ax1.plot(
                baseline_flops,
                [1 for _ in baseline_flops],
                linestyle="-",
                color="black",
                lw=1.5 * config.line_width_scale,
                marker="o",
                markersize=config.markersize,
            )
    else:
        for i, opt in enumerate(config.optim_to_visualize):
            opt_to_ax_0[opt] = axs[0, i]
            opt_to_ax_1[opt] = axs[1, i]
            axs[0, i].set_title(DISPLAY_NAMES[opt])

            axs[1, i].plot(
                baseline_flops,
                [1 for _ in baseline_flops],
                linestyle="-",
                color="black",
                lw=1.5 * config.line_width_scale,
                marker="o",
                markersize=config.markersize,
            )

    plot_loss_vs_compute(df, opt_to_ax_0, config)
    plot_compute_multiplier(df, loss_to_compute, opt_to_ax_1, config)

    axs[0, 0].set_ylabel("Loss")
    axs[1, 0].set_ylabel("Compute multiplier")
    compute_label = "Compute" if COMPUTE_COL == "compute" else "Non-embed compute"
    fig.text(0.5, 0.02, f"{compute_label} (FLOPs)", ha="center")

    axs[1, 0].set_ylim(config.y_min, config.y_max)

    legend = None
    if config.legend_on_subplot is not None:
        row, col = config.legend_on_subplot
        legend_handles = build_legend_handles(config, mode=config.legend_mode)
        legend = axs[row, col].legend(
            handles=legend_handles,
            loc="lower left",
            frameon=False,
        )
    elif config.plot_legend:
        legend_handles = build_legend_handles(config, mode=config.legend_mode)
        legend_handles += [Patch(facecolor="black", edgecolor="none", label="Baseline")]
        legend = fig.legend(
            handles=legend_handles,
            ncol=int(np.ceil(len(legend_handles) / 2)),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            frameon=False,
        )

    fig.tight_layout()
    fig.subplots_adjust(wspace=config.subplot_wspace, hspace=config.subplot_hspace)
    path = output_dir / f"{COMPUTE_COL}_whole.pdf"
    extra_artists = [legend] if legend is not None else None
    fig.savefig(path, bbox_extra_artists=extra_artists, bbox_inches="tight")
    print(f"Saved combined plot to {path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scaling frontier plots for optimizer comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plots/plot_scaling_frontier.py --preset fig_6_panel_1_2
  python plots/plot_scaling_frontier.py --preset alternative_scaling --output-dir ./my_figures
  python plots/plot_scaling_frontier.py --list-presets
        """,
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: ./figures/cr)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )
    parser.add_argument(
        "--save-cache",
        type=str,
        default=None,
        help="Save fetched data to this pickle file for faster reruns",
    )
    parser.add_argument(
        "--load-cache",
        type=str,
        default=None,
        help="Load data from this pickle file instead of fetching from wandb",
    )
    return parser.parse_args()


def list_presets() -> None:
    print("Available presets:\n")
    for name, config in PRESETS.items():
        print(f"  {name}:")
        print(f"    separate_plot: {config.separate_plot}")
        print(f"    optimizers: {config.optim_to_visualize}")
        print(f"    kinds: {config.kinds_to_visualize}")
        print()


def main() -> None:
    args = parse_args()

    if args.list_presets:
        list_presets()
        return

    config = PRESETS[args.preset]
    print(f"Using preset: {args.preset}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    configure_style()

    # Load data
    if args.load_cache:
        print(f"Loading cached data from {args.load_cache}")
        df = pickle.load(open(args.load_cache, "rb"))
    else:
        print("Fetching data from wandb...")
        df = load_all_runs(config)

    if args.save_cache:
        print(f"Saving data cache to {args.save_cache}")
        pickle.dump(df, open(args.save_cache, "wb"))

    loss_to_compute = collect_multiplier_samples(df)
    for key in sorted(loss_to_compute):
        print(f"Computed interpolation for {key}")

    create_legend_fig(config, output_dir)

    if config.separate_plot:
        generate_separate_plots(df, loss_to_compute, config, output_dir)
    else:
        generate_combined_plot(df, loss_to_compute, config, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
