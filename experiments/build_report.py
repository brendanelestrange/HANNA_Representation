"""Aggregate reports/metrics/*.json into a comparison table + figure.

Groups runs by descriptor_set, aggregates across seeds (mean +/- std), and emits:
  - reports/summary.csv            (one row per descriptor_set)
  - reports/summary_raw.csv        (one row per run)
  - reports/figures/comparison_mae.png
  - stdout markdown table (pasted into REPORT.md)

Robust to partial results — only reads whatever JSONs exist.
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(ROOT, "reports", "metrics")
FIG_DIR = os.path.join(ROOT, "reports", "figures")

# display order + friendly labels
ORDER = ["none", "curated_only", "full_only", "curated", "full"]
LABEL = {
    "none": "ChemBERTa only (baseline)",
    "curated": "ChemBERTa + curated (18)",
    "full": "ChemBERTa + full (~217)",
    "curated_only": "curated only (no BERT)",
    "full_only": "full only (no BERT)",
}
METRIC_KEYS = [
    ("test_mean_system_MAE", "mean-sys MAE"),
    ("test_median_system_MAE", "median-sys MAE"),
    ("test_overall_MAE", "overall MAE"),
    ("test_mean_system_MSE", "mean-sys MSE"),
]


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(METRICS_DIR, "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def agg(runs):
    by_set = defaultdict(list)
    for r in runs:
        by_set[r["descriptor_set"]].append(r)
    rows = []
    for dset in ORDER:
        if dset not in by_set:
            continue
        group = by_set[dset]
        row = {
            "descriptor_set": dset,
            "label": LABEL.get(dset, dset),
            "embedding_dim": group[0]["embedding_dim"],
            "n_params": group[0]["n_params"],
            "n_seeds": len(group),
            "seeds": sorted(g["seed"] for g in group),
        }
        for key, _ in METRIC_KEYS:
            vals = [g[key] for g in group if g.get(key) is not None]
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        best_eps = [(g.get("loss_history") or {}).get("best_epoch") for g in group]
        best_eps = [e for e in best_eps if e]
        row["mean_best_epoch"] = float(np.mean(best_eps)) if best_eps else None
        row["mean_wall_sec"] = float(np.mean([g["wall_time_sec"] for g in group]))
        rows.append(row)
    return rows, by_set


def fmt(mean, std):
    if mean is None:
        return "—"
    if std and std > 0:
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f}"


def markdown_table(rows):
    hdr = "| Config | emb dim | params | seeds | mean-sys MAE | median-sys MAE | overall MAE |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['embedding_dim']} | {r['n_params']:,} | {r['n_seeds']} | "
            f"{fmt(r['test_mean_system_MAE_mean'], r['test_mean_system_MAE_std'])} | "
            f"{fmt(r['test_median_system_MAE_mean'], r['test_median_system_MAE_std'])} | "
            f"{fmt(r['test_overall_MAE_mean'], r['test_overall_MAE_std'])} |"
        )
    return "\n".join(lines)


def write_csvs(rows, runs):
    with open(os.path.join(ROOT, "reports", "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "label", "emb_dim", "n_params", "n_seeds",
                    "mean_sys_MAE", "mean_sys_MAE_std",
                    "median_sys_MAE", "median_sys_MAE_std",
                    "overall_MAE", "overall_MAE_std", "mean_best_epoch", "mean_wall_sec"])
        for r in rows:
            w.writerow([r["descriptor_set"], r["label"], r["embedding_dim"], r["n_params"], r["n_seeds"],
                        r["test_mean_system_MAE_mean"], r["test_mean_system_MAE_std"],
                        r["test_median_system_MAE_mean"], r["test_median_system_MAE_std"],
                        r["test_overall_MAE_mean"], r["test_overall_MAE_std"],
                        r["mean_best_epoch"], r["mean_wall_sec"]])
    with open(os.path.join(ROOT, "reports", "summary_raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suffix", "config", "seed", "emb_dim", "mean_sys_MAE",
                    "median_sys_MAE", "overall_MAE", "best_epoch", "wall_sec"])
        for r in sorted(runs, key=lambda x: (x["descriptor_set"], x["seed"])):
            w.writerow([r["suffix"], r["descriptor_set"], r["seed"], r["embedding_dim"],
                        r["test_mean_system_MAE"], r["test_median_system_MAE"],
                        r["test_overall_MAE"], (r.get("loss_history") or {}).get("best_epoch"),
                        r["wall_time_sec"]])


def make_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skip figure: {e})")
        return
    labels = [r["label"] for r in rows]
    means = [r["test_mean_system_MAE_mean"] for r in rows]
    stds = [r["test_mean_system_MAE_std"] for r in rows]
    meds = [r["test_median_system_MAE_mean"] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, means, w, yerr=stds, capsize=4, label="mean-system MAE", color="#3b6fb0")
    ax.bar(x + w / 2, meds, w, label="median-system MAE", color="#b06f3b")
    base = next((r["test_mean_system_MAE_mean"] for r in rows if r["descriptor_set"] == "none"), None)
    if base is not None:
        ax.axhline(base, ls="--", lw=1, color="gray", label="baseline mean-sys MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Test MAE (ln γ)")
    ax.set_title("HANNA representation sweep: ChemBERTa ± RDKit descriptors")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "comparison_mae.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    runs = load_runs()
    if not runs:
        print("No metrics JSONs found yet.")
        return
    rows, _ = agg(runs)
    print(f"\n{len(runs)} runs across {len(rows)} configs\n")
    print(markdown_table(rows))
    write_csvs(rows, runs)
    make_figure(rows)
    print("\nwrote reports/summary.csv, reports/summary_raw.csv")


if __name__ == "__main__":
    main()
