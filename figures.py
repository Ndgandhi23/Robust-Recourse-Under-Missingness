#!/usr/bin/env python3
"""
Generate all evaluation figures for the paper.

Reads cross-validation results from pickles saved by run_eval_cv.py.
Supports multiple datasets — auto-discovers results/*/results_cv.pkl.

Figures produced (saved to results/figures/):
  1. fig_robustness_curves.png  — Robustness vs εtarget (Retrain, Ellipsoid, AWP)
  2. fig_summary_table.png      — Summary metrics table (mean +/- SE)
  3. fig_scatter.png             — Per-person robustness vs L2 scatter
  4. fig_lof.png                 — LOF plausibility per condition
  5. fig_tradeoff.png            — Robustness vs proximity trade-off (means + SE)

With multiple datasets, each figure becomes multi-panel (one column per dataset).

Usage:
    python figures.py                                       # auto-discover all
    python figures.py --results results/diabetes/results_cv.pkl   # one dataset
    python figures.py --results results/diabetes/results_cv.pkl results/fico/results_cv.pkl
"""

import argparse
import pathlib
import pickle
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ── style ──────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── constants ──────────────────────────────────────────────────────────

CONDS = ["ours", "baseline", "no-robust", "no-reveal"]

COLORS = {
    "ours":      "#2ca02c",
    "baseline":  "#ff7f0e",
    "no-robust": "#d62728",
    "no-reveal": "#1f77b4",
}

LABELS = {
    "ours":      "Ours",
    "baseline":  "Baseline (no Rashomon)",
    "no-robust": "No robustness",
    "no-reveal": "No reveal",
}

EVALUATORS = [
    ("retrain",   "Retrain R"),
    ("ellipsoid", "Ellipsoid R"),
    ("awp",       "AWP R"),
]


# ── data loading ──────────────────────────────────────────────────────

def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def discover_results(results_dir="results"):
    """Find all results/{dataset}/results_cv.pkl files, sorted by dataset name."""
    p = pathlib.Path(results_dir)
    if not p.exists():
        return []
    return sorted(p.glob("*/results_cv.pkl"))


def load_all_datasets(paths):
    """Load multiple result pickles. Returns list of (display_name, data) tuples."""
    datasets = []
    for p in paths:
        data = load_results(p)
        # get display name: prefer explicit key, fall back to filename
        display = data.get("display_name")
        if not display:
            # extract from parent directory: results/diabetes/results_cv.pkl -> Diabetes
            display = p.parent.name.replace("_", " ").title()
        datasets.append((display, data))
    return datasets


# ── helpers ────────────────────────────────────────────────────────────

def _mean_se(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=0) / np.sqrt(len(a)))


def _agg_curve(folds, cond, evaluator, etarget_grid):
    et_valid, means, ses = [], [], []
    for i, et in enumerate(etarget_grid):
        vals = []
        for f in folds:
            r = f["curves"][cond][evaluator][i]["robustness"]
            if r is not None:
                vals.append(r)
        if vals:
            m, se = _mean_se(vals)
            et_valid.append(et)
            means.append(m)
            ses.append(se)
    return np.array(et_valid), np.array(means), np.array(ses)


def _agg_metric(folds, cond, key):
    vals = []
    for f in folds:
        s = f["summaries"][cond]
        if s["nf"] > 0 and key in s:
            vals.append(s[key])
    return _mean_se(vals)


def _has_raw(folds):
    return "raw" in folds[0] and len(folds[0]["raw"]) > 0


# ── Figure 1: Robustness curves ───────────────────────────────────────

def fig_robustness_curves(datasets, outdir):
    """Grid: rows = evaluators, columns = datasets."""
    n_ds = len(datasets)
    n_ev = len(EVALUATORS)
    fig, axes = plt.subplots(n_ev, n_ds,
                             figsize=(5 * n_ds, 4 * n_ev),
                             squeeze=False, sharey=True)

    for col, (ds_name, data) in enumerate(datasets):
        folds = data["folds"]
        etarget_grid = data["etarget_grid"]

        for row, (ev_key, ev_title) in enumerate(EVALUATORS):
            ax = axes[row][col]
            for cond in CONDS:
                et, m, se = _agg_curve(folds, cond, ev_key, etarget_grid)
                if len(et) == 0:
                    continue
                ax.plot(et, m, "o-", color=COLORS[cond], label=LABELS[cond],
                        markersize=5, linewidth=2)
                ax.fill_between(et, m - se, m + se, color=COLORS[cond], alpha=0.15)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(left=0)

            if row == 0:
                ax.set_title(ds_name, fontsize=13)
            if col == 0:
                ax.set_ylabel(ev_title)
            if row == n_ev - 1:
                ax.set_xlabel(r"Target $\varepsilon$")

    axes[0][0].legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Robustness vs. Model Multiplicity Level (4-fold CV)",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out = outdir / "fig_robustness_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 2: Summary table ───────────────────────────────────────────

def fig_summary_table(datasets, outdir):
    """One table per dataset, stacked vertically."""
    metrics = [
        ("feas_rate", "Feasible"),
        ("cost",      r"Cost $\downarrow$"),
        ("nom",       "Nominal"),
        ("rr",        r"Retrain R $\uparrow$"),
        ("awp",       "AWP"),
        ("lof",       "LOF"),
        ("l2",        r"L2 $\downarrow$"),
    ]

    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1,
                             figsize=(11, 2.2 * n_ds),
                             squeeze=False)

    for ds_idx, (ds_name, data) in enumerate(datasets):
        ax = axes[ds_idx][0]
        folds = data["folds"]

        col_labels = [label for _, label in metrics]
        row_labels = [LABELS[c] for c in CONDS]
        row_colors = [to_rgba(COLORS[c], 0.15) for c in CONDS]

        cell_text = []
        for cond in CONDS:
            row = []
            for key, _ in metrics:
                m, se = _agg_metric(folds, cond, key)
                if np.isnan(m):
                    row.append("--")
                else:
                    row.append(f"{m:.3f}\u00b1{se:.3f}")
            cell_text.append(row)

        ax.axis("off")
        ax.set_title(f"{ds_name} (4-fold CV, mean \u00b1 SE)",
                     fontsize=13, pad=12)

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e6e6e6")

        for i, cond in enumerate(CONDS):
            table[i + 1, -1].set_facecolor(row_colors[i])
            table[i + 1, -1].set_text_props(weight="bold")
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(row_colors[i])

    fig.tight_layout()
    out = outdir / "fig_summary_table.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 3: Per-person scatter ──────────────────────────────────────

def fig_scatter(datasets, outdir):
    """One subplot per dataset."""
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    any_data = False
    for col, (ds_name, data) in enumerate(datasets):
        folds = data["folds"]
        ax = axes[0][col]

        if not _has_raw(folds):
            ax.text(0.5, 0.5, "No per-person data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ds_name)
            continue

        any_data = True
        for cond in CONDS:
            l2_vals, rr_vals = [], []
            for f in folds:
                for r in f["raw"][cond]:
                    if r["feasible"]:
                        l2_vals.append(r["l2"])
                        rr_vals.append(r["rr"])

            if not l2_vals:
                continue

            ax.scatter(l2_vals, rr_vals, color=COLORS[cond], label=LABELS[cond],
                       alpha=0.45, s=30, edgecolors="none", zorder=2)
            ax.scatter(np.mean(l2_vals), np.mean(rr_vals),
                       color=COLORS[cond], s=150, edgecolors="black",
                       linewidths=1.2, zorder=3, marker="D")

        ax.set_xlabel("L2 Proximity")
        ax.set_title(ds_name)
        if col == 0:
            ax.set_ylabel("Retrain Robustness")
            ax.legend(loc="lower left", framealpha=0.9)

    if not any_data:
        plt.close(fig)
        print("  SKIP fig_scatter (no per-person data)")
        return

    fig.tight_layout()
    out = outdir / "fig_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 4: LOF plausibility ────────────────────────────────────────

def fig_lof(datasets, outdir):
    """One subplot per dataset."""
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4), squeeze=False)

    for col, (ds_name, data) in enumerate(datasets):
        folds = data["folds"]
        ax = axes[0][col]

        x_pos = np.arange(len(CONDS))
        bar_means, bar_ses = [], []

        for cond in CONDS:
            m, se = _agg_metric(folds, cond, "lof")
            bar_means.append(m)
            bar_ses.append(se)

        ax.bar(x_pos, bar_means, yerr=bar_ses,
               color=[COLORS[c] for c in CONDS],
               edgecolor="black", linewidth=0.5,
               capsize=4, error_kw={"linewidth": 1.5})

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([LABELS[c] for c in CONDS], fontsize=8, rotation=20, ha="right")
        ax.set_title(ds_name)

        ymin = min(bar_means) - 0.1
        ymax = max(bar_means) + max(bar_ses) + 0.1
        ax.set_ylim(max(0, ymin), ymax)

        if col == 0:
            ax.set_ylabel("LOF Score")

    fig.tight_layout()
    out = outdir / "fig_lof.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── Figure 5: Robustness-proximity trade-off ──────────────────────────

def fig_tradeoff(datasets, outdir):
    """One subplot per dataset."""
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5), squeeze=False)

    for col, (ds_name, data) in enumerate(datasets):
        folds = data["folds"]
        ax = axes[0][col]

        for cond in CONDS:
            l2_m, l2_se = _agg_metric(folds, cond, "l2")
            rr_m, rr_se = _agg_metric(folds, cond, "rr")

            if np.isnan(l2_m):
                continue

            ax.errorbar(l2_m, rr_m, xerr=l2_se, yerr=rr_se,
                        fmt="o", color=COLORS[cond], label=LABELS[cond],
                        markersize=12, capsize=5, linewidth=2,
                        markeredgecolor="black", markeredgewidth=0.8)

        ax.set_xlabel("L2 Proximity")
        ax.set_title(ds_name)
        if col == 0:
            ax.set_ylabel("Retrain Robustness")
            ax.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout()
    out = outdir / "fig_tradeoff.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate all evaluation figures from CV results.")
    parser.add_argument("--results", type=str, nargs="+", default=None,
                        help="Paths to result pickles (auto-discovered from results/ if omitted)")
    args = parser.parse_args()

    if args.results:
        pkl_paths = [pathlib.Path(p) for p in args.results]
    else:
        pkl_paths = discover_results()

    if not pkl_paths:
        print("ERROR: no result pickles found.")
        print("  Run run_eval_cv.py first, or pass --results explicitly.")
        sys.exit(1)

    for p in pkl_paths:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    print(f"Loading {len(pkl_paths)} result file(s):")
    for p in pkl_paths:
        print(f"  {p}")

    datasets = load_all_datasets(pkl_paths)

    for ds_name, data in datasets:
        folds = data["folds"]
        etarget_grid = data["etarget_grid"]
        has_raw = _has_raw(folds)
        print(f"  {ds_name}: {len(folds)} folds, "
              f"{len(etarget_grid)} etarget levels, "
              f"per-person data: {'yes' if has_raw else 'no'}")

    outdir = pathlib.Path("results") / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures ...")
    fig_robustness_curves(datasets, outdir)
    fig_summary_table(datasets, outdir)
    fig_scatter(datasets, outdir)
    fig_lof(datasets, outdir)
    fig_tradeoff(datasets, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
