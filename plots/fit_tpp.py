import colorsys
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import colors as mpl_colors
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, LogLocator, NullLocator
from scipy.optimize import curve_fit

os.environ["TPU_VISIBLE_DEVICES"] = ""

DEFAULT_TPP = 20.0
LIGHTEN_STRENGTH = 0.3
FIG_SIZE = (8.0, 6.0)
DPI = 350


@dataclass
class PlotSpec:
    project: str
    filters: Dict
    config_keys: List[str]
    summary_keys: List[str]
    optimizer: str
    color: str
    dedupe: str = "first"
    flop_limit: float = 1e20
    per_width_best: bool = False


@dataclass
class SpecResult:
    spec: PlotSpec
    raw: pd.DataFrame
    plot_frame: pd.DataFrame
    best: pd.DataFrame
    base: pd.DataFrame


def set_plot_style():
    """Configure a clean, publication-style plotting theme."""
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
            "figure.figsize": FIG_SIZE,
        },
    )
    sns.set_context("paper", font_scale=2)


def lighten_color(color, factor):
    """Lighten a color toward white by the requested factor (0.0 keeps the original color)."""
    rgb = mpl_colors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0.0, min(1.0, l + (1.0 - l) * factor))
    return colorsys.hls_to_rgb(h, l, s)


def geometric_mean(values):
    """Compute the geometric mean of positive values, returning None if unavailable."""
    positives = [v for v in values if v > 0]
    if not positives:
        return None
    logs = np.log(np.asarray(positives, dtype=float))
    return float(np.exp(logs.mean()))


def fit_scaling_law(df, optimizer_name):
    """
    Fit L(T,P) = a*T^(-alpha) + b*P^(-beta) + c for a given optimizer.

    Args:
        df: DataFrame with columns T (tokens), num_params (P), and eval_loss (L)
        optimizer_name: Name of the optimizer for logging

    Returns:
        Fitted parameters (a, alpha, b, beta, c) and their covariance matrix
    """
    df_fit = df[(df["T"] > 0) & (df["num_params"] > 0) & (df["eval_loss"] > 0)].copy()
    if len(df_fit) < 5:
        print(f"Not enough data points for {optimizer_name}: {len(df_fit)}")
        return None, None

    T = df_fit["T"].values
    P = df_fit["num_params"].values
    L = df_fit["eval_loss"].values

    def scaling_law(data, a, alpha, b, beta, c):
        T, P = data
        return a * T ** (-alpha) + b * P ** (-beta) + c

    p0 = [1.0, 0.2, 1e8, 0.2, L.min() * 0.9]
    bounds = ([1e-6, 1e-6, 1e-6, 1e-6, 0], [np.inf, 1.0, np.inf, 1.0, L.min()])

    try:
        popt, pcov = curve_fit(
            scaling_law,
            (T, P),
            L,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        a, alpha, b, beta, c = popt
        perr = np.sqrt(np.diag(pcov))

        print(f"\n{optimizer_name} Scaling Law Fit:")
        print(f"L(T,P) = {a:.4e}*T^(-{alpha:.4f}) + {b:.4e}*P^(-{beta:.4f}) + {c:.4f}")
        print("Parameter uncertainties:")
        print(f"  a = {a:.4e} ± {perr[0]:.4e}")
        print(f"  alpha = {alpha:.4f} ± {perr[1]:.4f}")
        print(f"  b = {b:.4e} ± {perr[2]:.4e}")
        print(f"  beta = {beta:.4f} ± {perr[3]:.4f}")
        print(f"  c = {c:.4f} ± {perr[4]:.4f}")

        L_pred = scaling_law((T, P), *popt)
        ss_res = np.sum((L - L_pred) ** 2)
        ss_tot = np.sum((L - L.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((L - L_pred) ** 2))
        print(f"  R^2 = {r2:.6f}")
        print(f"  RMSE = {rmse:.6f}")

        return popt, pcov

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error fitting {optimizer_name}: {exc}")
        return None, None


def fetch_runs(project, filters, config_keys, summary_keys, dedupe="error"):
    api = wandb.Api()
    rows = []
    for run in api.runs(project, filters=filters, order="-created_at"):
        summary = run.summary._json_dict
        if not summary:
            continue
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        row = {key: config.get(key, -1) for key in config_keys}
        for key in summary_keys:
            row[key] = summary.get(key, -1)
        row["created_at"] = getattr(run, "created_at", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    dedupe_keys = [k for k in config_keys if k != "run_id"]

    if dedupe == "error":
        duplicates = (
            df.duplicated(subset=dedupe_keys, keep=False)
            if dedupe_keys
            else pd.Series(False, index=df.index)
        )
        if duplicates.any():
            print("Duplicate keys:")
            print(df.loc[duplicates, dedupe_keys])
            raise AssertionError("Don't expect duplicated runs")
    elif dedupe == "best":
        if dedupe_keys:
            idx = df.groupby(dedupe_keys)["eval_loss"].idxmin()
            df = df.loc[idx]
    elif dedupe == "first":
        start = pd.to_datetime(df["created_at"])
        df = df.assign(_start_time=start).sort_values("_start_time")
        if dedupe_keys:
            df = df.drop_duplicates(subset=dedupe_keys, keep="first")
        else:
            df = df.head(1)
        df = df.drop(columns=["_start_time"])
    else:
        raise ValueError(f"Unknown dedupe option: {dedupe}")

    return df.reset_index(drop=True)


def prepare_plot_frame(df, flop_limit, per_width_best):
    df_plot = df[df["compute"] < flop_limit].copy()
    df_plot["FLOPs"] = df_plot["compute"].apply(lambda x: float(f"{x:.1g}"))
    df_plot["Width"] = df_plot["model/D"]
    df_plot["Loss"] = df_plot["eval_loss"]
    df_plot["TPP"] = df_plot["T"] / df_plot["num_params"]

    if per_width_best:
        df_plot = df_plot.loc[df_plot.groupby(["FLOPs", "Width"])["Loss"].idxmin()]

    return df_plot


def summarize_by_flops(df_plot, optimizer):
    best_idx = df_plot.groupby("FLOPs")["Loss"].idxmin()
    base_idx = (
        df_plot.groupby("FLOPs")["TPP"]
        .apply(lambda series: (series - DEFAULT_TPP).abs().idxmin())
        .values
    )

    best = df_plot.loc[best_idx].copy()
    base = df_plot.loc[base_idx].copy()
    best["optimizer"] = optimizer
    base["optimizer"] = optimizer

    result_cols = ["optimizer", "Width", "Loss", "TPP"]
    if "run_id" in best.columns:
        result_cols.append("run_id")
    print(best[result_cols])

    return best, base


def process_spec(spec: PlotSpec):
    raw = fetch_runs(
        project=spec.project,
        filters=spec.filters,
        config_keys=spec.config_keys,
        summary_keys=spec.summary_keys,
        dedupe=spec.dedupe,
    )
    print(f"Get {len(raw)} runs for {spec.optimizer}")

    raw["petaflops"] = raw["compute"] / 1e15
    raw["hours"] = raw["_runtime"] / 3600

    plot_frame = prepare_plot_frame(raw, spec.flop_limit, spec.per_width_best)
    plot_frame["optimizer"] = spec.optimizer

    agg_map = {"Loss": "mean", "optimizer": "first"}
    if "run_id" in plot_frame.columns:
        agg_map["run_id"] = "first"
    plot_frame = (
        plot_frame.groupby(["FLOPs", "TPP", "Width"], as_index=False)
        .agg(agg_map)
    )

    best, base = summarize_by_flops(plot_frame, spec.optimizer)
    return SpecResult(spec=spec, raw=raw, plot_frame=plot_frame, best=best, base=base)


def plot_combined(results, out_path):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    opt_colors = {result.spec.optimizer: result.spec.color for result in results}

    # Keep only TPP values that appear for all optimizers to ensure fair comparison.
    common_tpps = None
    for result in results:
        tpps = {round(float(v), 6) for v in result.plot_frame["TPP"] if np.isfinite(v)}
        common_tpps = tpps if common_tpps is None else common_tpps & tpps
    if common_tpps is None:
        common_tpps = set()
    for result in results:
        mask = result.plot_frame["TPP"].apply(lambda v: round(float(v), 6) in common_tpps)
        result.plot_frame = result.plot_frame.loc[mask].reset_index(drop=True)

    all_flops = sorted(
        {flops for result in results for flops in result.plot_frame["FLOPs"].unique()}
    )
    flops_min = min(all_flops) if all_flops else 0.0
    flops_max = max(all_flops) if all_flops else 1.0

    def flops_factor(flops):
        if flops_max == flops_min:
            return 0.0
        return (flops - flops_min) / (flops_max - flops_min)

    positive_tpps = []
    for result in results:
        positive_rows = result.plot_frame[
            (result.plot_frame["TPP"] > 0) & result.plot_frame["Loss"].notna()
        ]
        if positive_rows.empty:
            continue
        positive_tpps.extend(positive_rows["TPP"].tolist())

    if positive_tpps:
        axis_x_min = min(positive_tpps)
        axis_x_max = max(positive_tpps)
        log_axis_min = np.log10(axis_x_min)
        log_axis_max = np.log10(axis_x_max)
    else:
        axis_x_min = axis_x_max = None
        log_axis_min = log_axis_max = None

    parabola_points = []

    for result in results:
        spec = result.spec
        for flops, group in result.plot_frame.groupby("FLOPs"):
            factor = flops_factor(flops) * LIGHTEN_STRENGTH
            flops_color = lighten_color(spec.color, factor)
            ordered = (
                group[(group["TPP"] > 0) & group["Loss"].notna()]
                .sort_values("TPP")
                .copy()
            )
            if ordered.empty:
                continue

            x = ordered["TPP"].to_numpy()
            y_raw = ordered["Loss"].to_numpy()
            log_x = np.log10(x)
            deg = min(2, len(log_x) - 1)
            log_fit = None
            x_fit = None
            if deg >= 1:
                coeffs = np.polyfit(log_x, y_raw, deg)
                poly = np.poly1d(coeffs)
                log_min = log_axis_min if log_axis_min is not None else log_x.min()
                log_max = log_axis_max if log_axis_max is not None else log_x.max()
                log_fit = np.linspace(log_min, log_max, 400)
                x_fit = np.power(10.0, log_fit)
            else:
                poly = np.poly1d([y_raw[0]])

            candidate_min = log_axis_min if log_axis_min is not None else log_x.min()
            candidate_max = log_axis_max if log_axis_max is not None else log_x.max()
            x_candidates = [candidate_min, candidate_max]
            if poly.order >= 2:
                a, b = poly.c[0], poly.c[1]
                if a != 0:
                    vertex_log_x = -b / (2 * a)
                    if candidate_min <= vertex_log_x <= candidate_max:
                        x_candidates.append(vertex_log_x)

            best_log_x = min(x_candidates, key=lambda lv: poly(lv))
            best_loss = poly(best_log_x)
            if not np.isfinite(best_loss) or best_loss <= 0:
                continue

            y = (y_raw / best_loss) - 1.0
            if log_fit is not None and x_fit is not None:
                y_fit = np.maximum((poly(log_fit) / best_loss) - 1.0, 0.0)
                ax.plot(
                    x_fit,
                    y_fit,
                    color=flops_color,
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=2,
                )
            best_x = 10.0 ** best_log_x
            parabola_points.append((spec.optimizer, best_x, 0.0, flops_color))
            ax.scatter(
                x,
                y,
                s=90,
                color=flops_color,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

    for _, x_star, y_star, color in parabola_points:
        ax.scatter(
            x_star,
            y_star,
            s=300,
            marker="*",
            facecolors=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )

    optimizer_minima = {}
    for optimizer_name, x_star, _, _ in parabola_points:
        optimizer_minima.setdefault(optimizer_name, []).append(x_star)

    ax.set_xlabel("Tokens per Parameter")
    ax.set_ylabel("Relative Loss Increase")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.2f}%"))
    ax.set_xscale("log")
    geo_means = {name: geometric_mean(vals) if vals else None for name, vals in optimizer_minima.items()}
    # base_ticks = [2.0, 4.0, 16.0, 32.0]
    # tick_values = sorted({*base_ticks, *(gm for gm in geo_means.values() if gm)})
    base_ticks = [2.0, 4.0, 8.0, 16.0, 32.0]
    tick_values = [2.0, 4.0, 8.0, 16.0, 32.0]
    def format_tick(val: float) -> str:
        return f"{int(val)}" if val in base_ticks else f"{val:.1f}"
    tick_labels = [format_tick(value) for value in tick_values]
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", axis="x", linestyle="-", color="#b3b3b3", linewidth=0.7)
    ax.grid(True, which="major", axis="y", linestyle="-", color="#d9d9d9", linewidth=0.7)
    ax.grid(False, which="minor")
    for opt_name, gm in geo_means.items():
        if gm is None:
            continue
        ax.axvline(
            gm,
            color=opt_colors.get(opt_name, "#333333"),
            linestyle="--",
            linewidth=1.1,
            zorder=1.5,
        )


    legend_handles = []
    seen_opt = set()
    for result in results:
        if result.spec.optimizer in seen_opt:
            continue
        seen_opt.add(result.spec.optimizer)
        label = result.spec.optimizer
        gm = geo_means.get(result.spec.optimizer)
        if gm is not None:
            label = f"{label} ({gm:.1f})"
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markersize=10,
                color=result.spec.color,
                label=label,
            )
        )

    limited_flops = all_flops[:3]
    legend_handles.extend(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor=lighten_color("#4d4d4d", flops_factor(flops) * LIGHTEN_STRENGTH),
            markeredgecolor="none",
            label=f"{flops} FLOPs",
        )
        for flops in limited_flops
    )
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markersize=16,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            label="Parabola Min",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),  # inside axes: top-right corner
        ncol=1,
        frameon=True,
        columnspacing=1.0,
        handletextpad=0.6,
        borderpad=0.6,
        # fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_view(view_name, specs, out_path):
    print(f"\nRunning view: {view_name}")
    results = [process_spec(spec) for spec in specs]

    print("\n" + "=" * 80)
    print("SCALING LAW FITTING")
    print("=" * 80)
    for result in results:
        fit_scaling_law(result.raw, result.spec.optimizer)

    plot_combined(results, out_path)


def main():
    set_plot_style()

    adam_color = "#F45A12"
    muon_color = "#0083CB"

    project = "llama_proj"
    summary_keys = ["eval_loss", "compute", "_runtime"]

    def build_filters(tag, custom_sweeps, opt_name=None):
        filters = {
            "state": "finished",
            "config.wandb_tag": tag,
            "config.custom_sweep": {"$in": custom_sweeps},
        }
        if opt_name:
            filters["config.opt/name"] = opt_name
        return filters

    def make_spec(optimizer, color, filters, config_keys, dedupe, per_width_best):
        return PlotSpec(
            project=project,
            filters=filters,
            config_keys=config_keys,
            summary_keys=summary_keys,
            dedupe=dedupe,
            flop_limit=1e20,
            per_width_best=per_width_best,
            optimizer=optimizer,
            color=color,
        )

    adam_filters = build_filters(
        tag="llama_fineweb_v3",
        custom_sweeps=[
            "oct23_adam_small_tpp",
            "oct23_adam_medium_tpp",
            "oct21_adam_mup_scaling",
            "oct25_adam__tpp",
        ],
    )
    muon_filters = build_filters(
        tag="llama_fineweb_v3",
        custom_sweeps=[
            "oct21_muon_tpp",
            "oct21_muon_medium_tpp_v2",
            "oct24_muon_large_tpp",
            "oct21_muon_mup_scaling",
        ],
    )

    adam_config_keys = [
        "opt/lr",
        "opt/name",
        "opt/mup",
        "model/D",
        "model/V",
        "token_per_param",
        "total_flops",
        "T",
        "num_params",
        "run_id",
    ]
    muon_config_keys = [
        "opt/name",
        "opt/mup",
        "model/D",
        "model/V",
        "T",
        "num_params",
        "run_id",
    ]

    view_presets = {
        "full": {
            "specs": [
                make_spec(
                    optimizer="Adam",
                    color=adam_color,
                    filters=adam_filters,
                    config_keys=adam_config_keys,
                    dedupe="first",
                    per_width_best=False,
                ),
                make_spec(
                    optimizer="Muon",
                    color=muon_color,
                    filters=muon_filters,
                    config_keys=muon_config_keys,
                    dedupe="best",  # Muon has runs with suboptimal hypers
                    per_width_best=True,
                ),
            ],
            "out_path": "./plots/figures/tpp_tuning.pdf",
        },
    }

    for view_name, view in view_presets.items():
        run_view(view_name=view_name, specs=view["specs"], out_path=view["out_path"])


if __name__ == "__main__":
    main()
