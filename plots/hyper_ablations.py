from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import wandb
except ImportError:  # pragma: no cover - optional at authoring time
    wandb = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "plots" / "figures"
WANDB_PROJECT = "llama_proj"
WANDB_TAG = "llama_fineweb_v3"
TARGET_WIDTHS = [512, 768, 1024]
SCALED_WIDTHS = [w for w in TARGET_WIDTHS if w != 512]
LOSS_ERR_ABS = {
    "adam": 0.003608574407,
    "muon": 0.0001796470658,
    "shampoo": 0.004046358352,
    "soap": 0.003427401829,
}

# REL_ERR = 0.0009484638263
ABLATION_ORDER = ["Ours", "Double LR", "Half LR", "Double WD", "Half WD", "Fix warmup ratio"]
PALETTE = {
    "Ours": "#4a4a4a",
    "Double LR": "#3f7fbf",
    "Half LR": "#77a8d8",
    "Double WD": "#4c9d70",
    "Half WD": "#8bc9a4",
    "Fix warmup ratio": "#c07d57",
}


@dataclass(frozen=True)
class FilterSpec:
    sweeps: Tuple[str, ...] = ()
    filters: Dict[str, object] = field(default_factory=dict)

    def build_query(self, tag: str) -> Dict[str, object]:
        query: Dict[str, object] = {"state": "finished", "config.wandb_tag": tag}
        query.update(self.filters)
        if self.sweeps and "config.custom_sweep" not in query:
            query["config.custom_sweep"] = {"$in": list(self.sweeps)}
        return query


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    display_name: str
    raw_names: Tuple[str, ...]
    scaled: Optional[FilterSpec] = None
    base: Optional[FilterSpec] = None
    baseline: Optional[Dict[str, float]] = None

    def iter_filters(self, tag: str) -> List[Dict[str, object]]:
        queries: List[Dict[str, object]] = []
        for spec in (self.scaled, self.base):
            if spec is None:
                continue
            query = spec.build_query(tag)
            if "config.opt/name" not in query:
                query["config.opt/name"] = {"$in": list(self.raw_names)}
            queries.append(query)
        return queries


OPTIMIZER_SPECS: Dict[str, OptimizerSpec] = {
    "adam": OptimizerSpec(
        name="adam",
        display_name="Adam",
        raw_names=("adam",),
        scaled=FilterSpec(
            sweeps=("oct21_adam_mup_scaling", "oct22_adam_mup_scaling_abl"),
            filters={"config.model/D": {"$in": SCALED_WIDTHS}},
        ),
        base=FilterSpec(
            sweeps=("oct21_adam_small_lr_wd", "oct21_adam_small_lr_b2_warmup"),
            filters={
                "config.model/D": 512,
                "config.opt/b2": 0.98,
                "config.opt/warmup_tokens": 742_723_584,
                "config.opt/weight_decay": {"$in": [0.0001, 0.0002, 0.0004]},
                "config.opt/lr": {"$in": [0.002, 0.004, 0.008]},
            },
        ),
        baseline={
            "width": 512,
            "lr": 0.004,
            "beta2": 0.98,
            "warmup_tokens": 742_723_584,
            "weight_decay": 0.0002,
        },
    ),
    "muon": OptimizerSpec(
        name="muon",
        display_name="Muon",
        raw_names=("muon_adam",),
        scaled=FilterSpec(
            sweeps=("oct21_muon_mup_scaling", "oct22_muon_mup_scaling_abl"),
            filters={"config.model/D": {"$in": SCALED_WIDTHS}},
        ),
        base=FilterSpec(
            sweeps=("oct21_muon_lr_wd",),
            filters={
                "config.model/D": 512,
                "config.opt/adam_b2": 0.98,
                "config.opt/warmup_tokens": 11_140_854,
                "config.opt/weight_decay": {"$in": [0.00016, 0.00032, 0.00064]},
                "config.opt/lr": {"$in": [0.004, 0.008, 0.016]},
                "config.opt/embed_lr_mult": 1.6,
            },
        ),
        baseline={
            "width": 512,
            "lr": 0.008,
            "adam_beta2": 0.98,
            "warmup_tokens": 11_140_854,
            "weight_decay": 0.00032,
            "embed_lr_mult": 1.6,
            "readout_lr_mult": 1.6,
        },
    ),
    "soap": OptimizerSpec(
        name="soap",
        display_name="SOAP",
        raw_names=("spectral-soap", "soap"),
        scaled=FilterSpec(
            sweeps=("nov28_soap_scaling_v2",),
            filters={"config.model/D": {"$in": SCALED_WIDTHS}},
        ),
        base=FilterSpec(
            sweeps=("nov26_soap_lr_wd",),
            filters={
                "config.model/D": 512,
                "config.opt/warmup_tokens": 23_210_112,
                "config.opt/block_size": 512,
            },
        ),
        baseline={
            "width": 512,
            "lr": 0.032,
            "warmup_tokens": 23_210_112,
            "weight_decay": 0.0002,
            "block_size": 512,
        },
    ),
    "shampoo": OptimizerSpec(
        name="shampoo",
        display_name="Shampoo",
        raw_names=("grafted_shampoo2_adam",),
        scaled=FilterSpec(
            sweeps=("nov28_shampoo_mup_with_abl",),
            filters={"config.model/D": {"$in": SCALED_WIDTHS}},
        ),
        base=FilterSpec(
            sweeps=("nov28_shampoo_mup_with_abl",),
            filters={
                "config.model/D": 512,
                "config.opt/b1": 0.95,
                "config.opt/b2": 0.98,
                "config.opt/warmup_tokens": 185_680_896,
                # "config.opt/weight_decay": {"$in": [0.0002, 0.0001, 0.0004]},
                # "config.opt/lr": {"$in": [0.002, 0.001, 0.004]},
                "config.opt/block_size": 512,
                # "config.opt/embed_lr_mult": 32,
                "config.opt/readout_lr_mult": 32,
            },
        ),
        baseline={
            "width": 512,
            "lr": 0.002,
            "beta1": 0.95,
            "beta2": 0.98,
            "warmup_tokens": 185_680_896,
            "weight_decay": 0.0002,
            "block_size": 512,
            # "embed_lr_mult": 32,
            "readout_lr_mult": 32,
        },
    ),
}

OPT_NAME_MAP = {raw: name for name, spec in OPTIMIZER_SPECS.items() for raw in spec.raw_names}
DISPLAY_NAMES = {name: spec.display_name for name, spec in OPTIMIZER_SPECS.items()}
BEST_BASELINES: Dict[str, Dict[str, float]] = {
    name: spec.baseline for name, spec in OPTIMIZER_SPECS.items() if spec.baseline is not None
}


def configure_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.5,
            "lines.linewidth": 2,
            "grid.alpha": 0.3,
            "figure.figsize": (7.5, 5.5),
        },
    )
    sns.set_context("paper", font_scale=2)


def norm_opt_name(raw: str) -> str:
    key = str(raw).lower()
    return OPT_NAME_MAP.get(key, key.split("_")[0])


def close_enough(a: float, b: float, *, rel: float = 1e-3, abs_tol: float = 1e-12) -> bool:
    # return a==b
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= max(abs_tol, rel * abs(b))




def same_family(row: pd.Series, baseline: Dict[str, float]) -> bool:
    if int(row["width"]) != int(baseline["width"]):
        return False
    checks = [
        ("warmup_tokens", 1),
        ("beta2", 1e-4),
        ("adam_beta2", 1e-4),
        ("beta1", 1e-4),
        ("embed_lr_mult", 1e-3),
        ("readout_lr_mult", 1e-3),
        ("block_size", 1),
    ]
    for key, tol in checks:
        target = baseline.get(key)
        if target is None:
            continue
        if not close_enough(row.get(key, np.nan), target, rel=0, abs_tol=tol):
            return False
    return True

mapping = {
    "double_lr": "Double LR",
    "half_lr": "Half LR",
    "double_wd": "Double WD",
    "half_wd": "Half WD",
    "fix_warmup_ratio": "Fix warmup ratio",
    "None": "Ours"
}

def infer_ablation(row: pd.Series) -> Optional[str]:
    if row.get("custom_ablation_str") in mapping:
        return mapping[row.get("custom_ablation_str")]
    
    opt = row.get("opt_normalized")
    baseline = BEST_BASELINES.get(opt)
    if baseline.get('width', -1) != row.get('width', -2):
        return "Ours"

    lr = row.get("lr", np.nan)
    wd = row.get("weight_decay", np.nan)
    lr_base = baseline.get("lr", np.nan)
    wd_base = baseline.get("weight_decay", np.nan)

    if baseline.get('width', -1) == row.get('width', -2):
        if close_enough(lr, lr_base) and close_enough(wd, wd_base):
            return "Ours"
        if close_enough(lr, lr_base * 2) and close_enough(wd, wd_base):
            return "Double LR"
        if close_enough(lr, lr_base / 2) and close_enough(wd, wd_base):
            return "Half LR"

        if close_enough(lr, lr_base) and close_enough(wd, wd_base * 2):
            return "Double WD"
        if close_enough(lr, lr_base) and close_enough(wd, wd_base / 2):
            return "Half WD"

        if row.get("custom_ablation_str") == "fix_warmup_ratio":
            return "Fix warmup ratio"
    return None


def dedupe_by_ablation(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby(["opt_normalized", "ablation", "width"])["eval_loss"].idxmin()
    return df.loc[idx].reset_index(drop=True)


def ensure_baseline_presence(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee an 'Ours' row for each width/opt so percent deltas are anchored."""
    df = df.copy()
    # Prefer exact baselines where we know the target hypers (e.g., base widths)
    for opt, base in BEST_BASELINES.items():
        width = base.get("width")
        if width is None:
            continue
        mask = (df["opt_normalized"] == opt) & (df["width"] == width)
        if not mask.any():
            continue
        if (df.loc[mask, "ablation"] == "Ours").any():
            continue

        lr_base = base.get("lr", np.nan)
        wd_base = base.get("weight_decay", np.nan)
        candidates = df.loc[mask].copy()
        candidates = candidates[
            candidates.apply(
                lambda r: close_enough(r.get("lr", np.nan), lr_base, rel=0, abs_tol=lr_base * 0.01)
                and close_enough(r.get("weight_decay", np.nan), wd_base, rel=0, abs_tol=max(1e-6, wd_base * 0.1)),
                axis=1,
            )
        ]
        if candidates.empty:
            continue

        def score(row: pd.Series) -> float:
            lr_delta = abs(row.get("lr", np.nan) - base.get("lr", np.nan))
            wd_delta = abs(row.get("weight_decay", np.nan) - base.get("weight_decay", np.nan))
            warm_delta = abs(row.get("warmup_tokens", np.nan) - base.get("warmup_tokens", np.nan))
            return float(lr_delta) + float(wd_delta) * 10 + float(warm_delta) / 1e9

        idx = candidates.apply(score, axis=1).idxmin()
        df.loc[idx, "ablation"] = "Ours"

    # Fallback: ensure every (opt, width) has a baseline, pick best loss if missing
    for (opt, width), group in df.groupby(["opt_normalized", "width"]):
        if (group["ablation"] == "Ours").any():
            continue
        idx = group["eval_loss"].idxmin()
        df.loc[idx, "ablation"] = "Ours"
    return df


def add_percent_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for (opt, width), group in df.groupby(["opt_normalized", "width"]):
        base = group[group["ablation"] == "Ours"]
        if base.empty:
            continue
        base_loss = base["eval_loss"].iloc[0]
        err_pct = LOSS_ERR_ABS[opt]*100 if LOSS_ERR_ABS.get(opt) is not None else np.nan
        enriched = group.copy()
        enriched["loss_increase_pct"] = (enriched["eval_loss"] / base_loss - 1) * 100
        enriched["err_pct"] = err_pct
        rows.append(enriched)
    if not rows:
        raise ValueError("No groups contained a baseline to anchor deltas.")
    return pd.concat(rows, ignore_index=True)


def draw_bars(ax: plt.Axes, bars: pd.DataFrame, palette: Dict[str, tuple]) -> plt.Axes:
    xs = np.sort(bars["width"].unique())
    hues = [h for h in ABLATION_ORDER if h != "Ours" and h in bars["ablation"].unique()]
    w = 0.8 / max(len(hues), 1)
    off = 0.4 - w / 2

    for j, hue in enumerate(hues):
        bh = bars.loc[bars["ablation"].eq(hue)].set_index("width").reindex(xs)
        x0 = np.arange(len(xs)) - off + j * w
        vals = bh["loss_increase_pct"].to_numpy()
        errs = bh["err_pct"].to_numpy()
        mask = ~np.isnan(vals)
        if not mask.any():
            continue
        ax.bar(x0[mask], vals[mask], yerr=errs[mask], width=w, label=hue, capsize=3, color=palette[hue])

    ax.set_xticks(np.arange(len(xs)))
    labels = []
    for width in xs:
        suffix = ""
        if int(width) == 512:
            suffix = " (tuned)"
        elif int(width) in (768, 1024):
            suffix = " (scaled)"
        labels.append(f"{int(width)}{suffix}")
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model width")
    ax.set_ylabel("Loss increase")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.set_ylim(-0.2, 1.0)
    ax.axhline(0, color="#555555", linewidth=1.5, linestyle="-")
    return ax


def get_runs(
    filters: dict,
    *,
    project: str,
    config_keys: Iterable[str],
    summary_keys: Iterable[str],
    dedupe_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    if wandb is None:
        raise RuntimeError("wandb is required to download runs from the API.")

    print(filters)

    api = wandb.Api()
    runs = api.runs(project, filters=filters, order="-created_at")
    summaries, configs, names = [], [], []
    for run in runs:
        summaries.append(run.summary._json_dict)
        configs.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        names.append(run.name)

    runs_df = pd.DataFrame({"summary": summaries, "config": configs, "name": names})
    runs_df = runs_df[runs_df["summary"].apply(lambda s: s != {})]

    for key in config_keys:
        runs_df[key] = runs_df["config"].apply(lambda cfg: cfg.get(key, -1))
    for key in summary_keys:
        runs_df[key] = runs_df["summary"].apply(lambda summary: summary.get(key, -1))

    runs_df = runs_df.drop(columns=["summary", "config", "name"])
    runs_df = runs_df.replace({None: -1})

    if dedupe_keys:
        idx = runs_df.groupby(dedupe_keys)["eval_loss"].idxmin()
        runs_df = runs_df.loc[idx]
    return runs_df.reset_index(drop=True)


def build_filters(tag: str) -> List[dict]:
    filters: List[dict] = []
    for spec in OPTIMIZER_SPECS.values():
        filters.extend(spec.iter_filters(tag))
    return filters


def load_runs(project: str, tag: str) -> pd.DataFrame:
    config_keys = [
        "opt/name",
        "custom_ablation_str",
        "model/D",
        "T",
        "num_params",
        "run_id",
        "opt/lr",
        "opt/b1",
        "opt/b2",
        "opt/adam_b2",
        "opt/warmup_tokens",
        "opt/weight_decay",
        "opt/embed_lr_mult",
        "opt/readout_lr_mult",
        "opt/block_size",
    ]
    summary_keys = ["eval_loss", "compute", "_runtime"]

    frames = [
        get_runs(
            filters,
            project=project,
            config_keys=config_keys,
            summary_keys=summary_keys,
            dedupe_keys=[
                "opt/name",
                "custom_ablation_str",
                "model/D",
                "opt/lr",
                "opt/b1",
                "opt/warmup_tokens",
                "opt/weight_decay",
                "opt/b2",
                "opt/block_size",
                # "opt/adam_b2",
                # "opt/embed_lr_mult",
                # "opt/readout_lr_mult",
            ],
        )
        for filters in build_filters(tag)
    ]
    df = pd.concat(frames, ignore_index=True).reset_index(drop=True)

    df = df.rename(
        columns={
            "opt/name": "opt_name",
            "model/D": "width",
            "opt/lr": "lr",
            "opt/b1": "beta1",
            "opt/b2": "beta2",
            "opt/adam_b2": "adam_beta2",
            "opt/warmup_tokens": "warmup_tokens",
            "opt/weight_decay": "weight_decay",
            "opt/embed_lr_mult": "embed_lr_mult",
            "opt/readout_lr_mult": "readout_lr_mult",
            "opt/block_size": "block_size",
        }
    )
    df["opt_normalized"] = df["opt_name"].apply(norm_opt_name)
    # breakpoint()
    df["ablation"] = df.apply(infer_ablation, axis=1)
    # df = ensure_baseline_presence(df)
    df = df.sort_values("width", ascending=False)
    print(df[df['opt_normalized']=='muon'][['width', 'ablation', 'custom_ablation_str', 'lr', 'weight_decay', 'eval_loss']])
    df = df[~((df["opt_normalized"] == "adam") & (df["width"] == 512) & (df["ablation"] == "Fix warmup ratio"))]
    df = df[df["ablation"].notna()]
    df = df[df["width"].isin(TARGET_WIDTHS)]
    df = dedupe_by_ablation(df)
    df["petaflops"] = df["compute"] / 1e15
    df["hours"] = df["_runtime"] / 3600
    df = add_percent_deltas(df)
    return df


def plot_ablation(bars: pd.DataFrame, opt_name: str) -> plt.Figure:
    fig, ax = plt.subplots()
    palette = dict(PALETTE)
    subset = bars[bars["opt_normalized"] == opt_name]
    draw_bars(ax, subset, palette)
    title = DISPLAY_NAMES.get(opt_name, opt_name.title())
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_legend(handles: List[plt.Artist], labels: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 1.2))
    ax.axis("off")
    ax.legend(handles, labels, loc="center", frameon=False, ncol=len(labels), fontsize=24)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ablation figures without the notebook.")
    parser.add_argument("--project", default=WANDB_PROJECT, help="Weights & Biases project name.")
    parser.add_argument("--tag", default=WANDB_TAG, help="WANDB tag used to filter runs.")
    parser.add_argument("--output-dir", default=str(FIG_DIR), help="Directory to write figure PDFs.")
    args = parser.parse_args()

    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bars = load_runs(project=args.project, tag=args.tag)

    handles: Optional[List[plt.Artist]] = None
    labels: Optional[List[str]] = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_opts = list(bars["opt_normalized"].unique())
    opt_order: List[str] = []
    seen: set[str] = set()
    for name in OPTIMIZER_SPECS:
        if name in available_opts:
            opt_order.append(name)
            seen.add(name)
    for name in available_opts:
        if name not in seen:
            opt_order.append(name)

    for opt in opt_order:
        fig = plot_ablation(bars, opt)
        path = output_dir / f"ablation_{opt}.pdf"
        fig.savefig(path, bbox_inches="tight")
        if handles is None:
            handles, labels = fig.axes[0].get_legend_handles_labels()
        plt.close(fig)

    if handles and labels:
        leg = plot_legend(handles, labels)
        leg.savefig(output_dir / "ablation_legend.pdf", bbox_inches="tight")
        plt.close(leg)


if __name__ == "__main__":
    main()
