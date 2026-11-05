"""VISUALISE: monitoring charts across time from gold.model_monitoring -> datamart/gold/monitoring_charts/*.png

Run once per DAG run (charts always reflect everything known up to that run):
    python -m utils.model_visualize --snapshotdate YYYY-MM-DD
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from utils import config as C
from utils.ml_common import load_registry, read_gold, to_date

BG, INK, MUTED, GRID = "#F6F3EC", "#1B1D22", "#6E6B64", "#E2DCCF"
S1, S2 = "#2A6CB0", "#C46A1C"           # series colours (validated: CVD-safe on the paper background)
AMBER, RED = "#B8860B", "#B3261E"      # reserved for thresholds only

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def _threshold(ax, y, color, label, style):
    ax.axhline(y, color=color, lw=1.2, ls=style, zorder=1)
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=9, color=color)


def _promotions(ax, reg, x_min, x_max):
    for m in reg["models"]:
        if m.get("promoted_on"):
            d = pd.Timestamp(m["promoted_on"])
            if x_min <= d <= x_max:
                ax.axvline(d, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
                ax.annotate(f"{m['algorithm'].replace('_', ' ')}\nlive {d:%b %Y}", xy=(d, 1.0),
                            xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                            va="top", fontsize=8.5, color=MUTED)


def _fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def plot_performance(ax, mon, reg):
    p = mon.dropna(subset=["auc"])
    ax.plot(p["cohort"], p["auc"], color=S1, lw=2, marker="o", ms=7, mec=BG, mew=1.5, label="AUC (matured cohort)")
    _threshold(ax, C.AUC_AMBER, AMBER, f"amber {C.AUC_AMBER}", "--")
    _threshold(ax, C.AUC_RED, RED, f"red {C.AUC_RED}", ":")
    if not p.empty:
        last = p.iloc[-1]
        ax.annotate(f"{last['auc']:.3f}", (last["cohort"], last["auc"]), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    lo = min(0.6, (p["auc"].min() - 0.03) if not p.empty else 0.6)
    ax.set_ylim(lo, max(0.9, (p["auc"].max() + 0.03) if not p.empty else 0.9))
    ax.set_title("Performance: AUC by application cohort", pad=12)
    ax.set_ylabel("AUC")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10)


def plot_stability(ax, mon):
    s = mon.dropna(subset=["psi_score"])
    ax.plot(s["cohort"], s["psi_score"], color=S1, lw=2, marker="o", ms=7, mec=BG, mew=1.5, label="Score PSI")
    ax.plot(s["cohort"], s["max_feature_psi"], color=S2, lw=2, marker="s", ms=7, mec=BG, mew=1.5,
            label="Worst top-5 feature PSI")
    _threshold(ax, C.PSI_AMBER, AMBER, f"amber {C.PSI_AMBER}", "--")
    _threshold(ax, C.PSI_RED, RED, f"red {C.PSI_RED}", ":")
    ax.set_ylim(0, max(0.3, s[["psi_score", "max_feature_psi"]].max().max() * 1.15 if not s.empty else 0.3))
    ax.set_title("Stability: PSI vs model reference", pad=12)
    ax.set_ylabel("PSI")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10)


def plot_calibration(ax, mon):
    s = mon.dropna(subset=["mean_pd"])
    p = mon.dropna(subset=["default_rate"])
    ax.plot(s["cohort"], s["mean_pd"], color=S1, lw=2, marker="o", ms=7, mec=BG, mew=1.5, label="Mean predicted PD")
    ax.plot(p["cohort"], p["default_rate"], color=S2, lw=2, marker="s", ms=7, mec=BG, mew=1.5,
            label="Actual default rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(0.5, s["mean_pd"].max() * 1.2 if not s.empty else 0.5))
    ax.set_title("Calibration: predicted vs actual", pad=12)
    ax.set_ylabel("rate")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10)


def visualise(run_date):
    mon = read_gold("model_monitoring")
    if mon.empty:
        print(f"{run_date}: no monitoring results yet")
        return
    # early runs have stability results only; performance columns appear once the first cohort's labels mature
    expected = ["auc", "default_rate", "mean_pd", "psi_score", "max_feature_psi"]
    mon = mon.reindex(columns=[*mon.columns, *[c for c in expected if c not in mon.columns]])
    mon["cohort"] = pd.to_datetime(mon["snapshot_date"])
    mon = mon.sort_values("cohort")
    reg = load_registry()
    os.makedirs(C.CHARTS, exist_ok=True)
    x_min, x_max = mon["cohort"].min() - pd.Timedelta(days=20), mon["cohort"].max() + pd.Timedelta(days=20)

    panels = [("performance", lambda ax: plot_performance(ax, mon, reg)),
              ("stability", lambda ax: plot_stability(ax, mon)),
              ("calibration", lambda ax: plot_calibration(ax, mon))]
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        fn(ax)
        _promotions(ax, reg, x_min, x_max)
        ax.set_xlim(x_min, x_max)
        _fmt_dates(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(C.CHARTS, f"{name}.png"), dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for ax, (_, fn) in zip(axes, panels):
        fn(ax)
        _promotions(ax, reg, x_min, x_max)
        ax.set_xlim(x_min, x_max)
        _fmt_dates(ax)
    fig.suptitle(f"Credit default model monitoring (as of {run_date})", x=0.01, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(C.CHARTS, "monitoring_dashboard.png"), dpi=150)
    plt.close(fig)
    print(f"charts written to {C.CHARTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    visualise(to_date(ap.parse_args().snapshotdate))
